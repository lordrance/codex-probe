# CodexProbe

CodexProbe records every LLM call that the [OpenAI Codex CLI](https://github.com/openai/codex) makes: the complete request, the complete streamed response, and the tool calls. It is a small local proxy that Codex reaches through its `base_url` setting. It forwards request and response bodies unchanged, so with the same Codex settings, Codex behaves the same with or without it. A config change, not a code change, sends Codex to another backend, such as a local Qwen model.

It is a solution to [Task 7](https://github.com/STAR-Computing/ProgrammingTask/blob/master/07_codex_llm_call_recorder.md) of the STAR-Computing programming tasks, tested with Codex CLI 0.164.0 on Windows 11.

## How it works

```text
              request                              request
 Codex CLI  ──────────▶  CodexProbe (127.0.0.1)  ──────────▶  model backend:
            ◀──────────            │             ◀──────────  OpenAI, Ollama or a mock
           streamed reply          │            streamed reply
                                   ▼
                      one JSON record per LLM call
```

- **Transparent:** request and response bodies are forwarded byte for byte. A streamed (SSE) reply is passed on chunk by chunk as it arrives, and parsed from a copy.
- **Complete:** each call is saved with the raw HTTP request and response, every stream event, the rebuilt output (messages, tool calls, reasoning summaries), token usage and timing. Credentials are redacted.
- **Organized by session:** one JSONL file per Codex session, which `load_session()` reads back in order.
- **Config-only backend switch:** the backend is the `upstream` entry of CodexProbe's config, so switching needs no code change.

## Results

| Run | Result |
|---|---|
| 349 offline tests, in a fresh install from a clean copy | **PASS** |
| Real Codex → CodexProbe → a scripted mock backend (not a model): a small coding task | **PASS**. Every recorded body is byte-identical to what the mock received and sent |
| Real Codex → CodexProbe → OpenAI, signed in with ChatGPT: a fixed one-word reply, and a coding task | **PASS**. The coding task took 3 LLM calls, all recorded in full. The first reply run failed on a bug, since fixed: the backend's stream had no `Content-Type` |
| Real Codex → CodexProbe → a local Qwen on Ollama: a fixed one-word reply | **PASS**. The request reached the local model byte for byte, and the reply was recorded in full |

The OpenAI and Qwen runs used earlier versions of the quickstart, and were not repeated on the final code. [VALIDATION.md](VALIDATION.md) has the commands, outputs and evidence.

## Quick start

No account or model is needed. You need Python 3.12 or newer, and Node.js with npm for the Codex CLI. The commands are for Windows PowerShell, the only platform tested.

**1. Install and test,** in this folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe -m pytest -q tests          # 349 passed, 1 deselected
```

**2. Install the Codex CLI 0.164.0** into a folder of its own, which leaves any global Codex alone, and choose a folder for Codex's settings:

```powershell
npm.cmd install --prefix "<tools folder>" @openai/codex@0.164.0
$codex = "<tools folder>\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe"
& $codex --version                                    # codex-cli 0.164.0
$codexHome = "<a folder for Codex's settings>"        # not your ~\.codex; reuse it for every run
```

**3. Run the demo.** The real Codex runs a small coding task through CodexProbe, against a local mock backend. The mock is not a model: it replays a scripted solution, so Codex writes `calc.py` and a unit test, runs the test, and answers.

```powershell
.venv\Scripts\python.exe examples\quickstart.py --mode mock --codex $codex --codex-home $codexHome
```

The quickstart prints each recorded call, then checks the run from its own evidence. The end of a run:

```text
LLM calls recorded by CodexProbe (3):
  seq 0  model codexprobe-mock  completed  tool calls: exec_command (call-1)  first byte 29.7 ms, total 33.1 ms
  seq 1  model codexprobe-mock  completed  tool calls: exec_command (call-2)  first byte 2.9 ms, total 6.1 ms
  seq 2  model codexprobe-mock  completed  tool calls: none  first byte 2.7 ms, total 5.4 ms
…
PASS  proxied run: logged bodies equal the mock's bytes: all 3 request and response bodies are identical, byte for byte
RESULT: PASS
```

**4. Read the log.** It is in `runs\quickstart-<time>\codexprobe\<run>\`, which the quickstart prints as `Log directory`:

```powershell
.venv\Scripts\python.exe -c "import sys; from codex_probe import load_session; [print(c['seq'], c['outcome'], c['derived']['item_types']) for c in load_session(sys.argv[1])]" "<log directory>"
```

For the run above, it prints `0 completed ['function_call']`, `1 completed ['function_call']` and `2 completed ['message']`.

## Using it with OpenAI

Sign in with ChatGPT once, into a new empty folder; this opens the sign-in page in the browser. Then run a one-word reply and the coding task through CodexProbe:

```powershell
$openaiHome = "<a new, empty folder>"
New-Item -ItemType Directory -Force $openaiHome | Out-Null
$env:CODEX_HOME = $openaiHome; & $codex login; Remove-Item Env:CODEX_HOME
.venv\Scripts\python.exe examples\quickstart.py --mode openai --codex $codex --codex-home $openaiHome --task text
.venv\Scripts\python.exe examples\quickstart.py --mode openai --codex $codex --codex-home $openaiHome
```

Signed in with ChatGPT, Codex sends its requests to `https://chatgpt.com/backend-api/codex`, and they count against the plan's Codex usage. No API key is used. The quickstart never reads Codex's `auth.json`, and passes its settings as `codex exec -c` overrides instead of writing a `config.toml` into that folder.

## Using it with a local Qwen model

**Serve the model with Ollama 0.40.10.** The qwen mode refuses other versions. The tested build is the portable `ollama-windows-amd64.zip` from [Ollama's v0.40.10 release](https://github.com/ollama/ollama/releases/tag/v0.40.10); below, `ollama` means the `ollama.exe` it contains. In a second PowerShell window:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NO_CLOUD = "1"           # no cloud models and no server-side web search
ollama serve
```

**Then, back in the first window** (where `$codex` and `$codexHome` are set), in this folder, build the model and run Codex against it:

```powershell
ollama pull qwen2.5-coder:1.5b
ollama create codexprobe-qwen2.5-coder:1.5b -f examples/ollama/Modelfile
.venv\Scripts\python.exe examples\quickstart.py --mode qwen --codex $codex --codex-home $codexHome --task text
```

The qwen mode first checks that Ollama 0.40.10 is running and that the model sets `num_ctx 16384`. Otherwise it stops with BLOCKED and runs nothing. It writes the Codex settings marked *qwen* under [Codex config](#codex-config).

**Hardware.** The 1.5B model with a 16K context took about 1.6 GB of GPU memory, and about 1 GB of disk. It was tested on an RTX 3050 Laptop GPU with 4 GB. That GPU is why the 1.5B model was used: the task's example, Qwen2.5-Coder-7B, is about a 4.7 GB download even in its 4-bit build (Ollama's model library; not measured here).

**The [Modelfile](examples/ollama/Modelfile)** reuses the weights of `qwen2.5-coder:1.5b` and changes three things:
1. Codex sends its instructions as `developer` messages, which the base template drops without an error ([ollama#18305](https://github.com/ollama/ollama/issues/18305)). The new template adds them to the system prompt.
2. Tool calls are asked for as a bare JSON object instead of inside `<tool_call>` tags. Ollama parses a reply that starts with a JSON object as a tool call.
3. `num_ctx 16384`. On the test GPU, Ollama's default context was 4096 tokens, and according to Ollama's source (not measured here), a longer prompt loses its start. One Codex request was over 5,100 tokens.

## Codex config

This is the `config.toml` that the qwen mode writes into the settings folder, after a marker comment line. The mock mode writes the same file without the lines and tables marked *qwen*, and with `model = "codexprobe-mock"`.

```toml
model = "codexprobe-qwen2.5-coder:1.5b"   # the model Codex asks the backend for
model_provider = "codexprobe"
web_search = "disabled"
model_context_window = 16384              # qwen: the Modelfile's num_ctx
model_reasoning_summary = "none"          # qwen: minimal tool set

[model_providers.codexprobe]
name = "CodexProbe"                       # not "OpenAI", which turns on OpenAI-only code paths
base_url = "http://127.0.0.1:<port>/v1"   # the URL that start() returned
wire_api = "responses"
request_max_retries = 0                   # so that every request the backend sees was sent once
stream_max_retries = 0
stream_idle_timeout_ms = 600000           # qwen: Ollama sends nothing while it reads the prompt

[features]
unbounded_connection_retries = false
plugins = false
apps = false
view_image = false                        # qwen: minimal tool set
tool_suggest = false                      # qwen: minimal tool set
goals = false                             # qwen: minimal tool set

[agents]                                  # qwen: minimal tool set
enabled = false

[tools.experimental_request_user_input]   # qwen: minimal tool set
enabled = false

[skills]                                  # qwen: minimal tool set
include_instructions = false

[analytics]
enabled = false

[windows]
sandbox = "unelevated"                    # without it, codex exec rejected every command on the test machine
```

The *minimal tool set* lines and tables leave Codex only `exec_command` and `write_stdin`, which keeps each request small. For OpenAI with a ChatGPT sign-in, the openai mode passes the unmarked settings except `model` as `codex exec -c` overrides, so Codex uses its default model, and adds `requires_openai_auth = true` and `supports_websockets = false` to the provider.

## Using the API

The task's snippet, here with the OpenAI backend that Codex reaches with a ChatGPT sign-in:

```python
from codex_probe import ProxyRecorder, load_session

config = {
    "upstream": {"base_url": "https://chatgpt.com/backend-api/codex", "auth": {"mode": "passthrough"}},
    "log_dir": "runs",
}

recorder = ProxyRecorder(config)
endpoint = recorder.start()   # e.g. "http://127.0.0.1:8135/v1"
# point model_providers.<name>.base_url in Codex's config.toml at `endpoint`,
# then run Codex normally (e.g. `codex exec "<task prompt>"`)
calls = recorder.stop()       # one dict per LLM call, in order
same = load_session(recorder.run_dir)   # the same list, read back from disk
```

For the local Qwen, `upstream` is `{"base_url": "http://127.0.0.1:11434/v1", "auth": {"mode": "none"}}`, and nothing else changes.

**Config fields.** A config is a dict, or the path to a JSON file holding one. Unknown keys are rejected.

| Field | Default | Meaning |
|---|---|---|
| `upstream.base_url` | required | The real backend. A request to `/v1/<path>` goes to `<base_url>/<path>`. |
| `upstream.auth` | `{"mode": "passthrough"}` | `mode` is `passthrough` (forward Codex's own credentials), `none` (send none, for local servers) or `env` (send `Authorization: Bearer` with the key in the environment variable named by `env_var`). |
| `upstream.name` | the host of `base_url`, with its port if the URL names one | A label written into every record. |
| `upstream.wire_api` | `"responses"` | The only value Codex 0.164.0 supports (see below). |
| `listen.host`, `listen.port` | `"127.0.0.1"`, `0` | Loopback only. Port 0 picks a free port. |
| `log_dir` | required | Each recorder writes one run folder under it. |
| `seed` | `0` | Only used to derive each record's `call_id`. Never sent upstream. |

## What a captured call looks like

A run folder holds `run.json` (the config, versions and times), one `session-<sha256 of the session id>.jsonl` per Codex session with one line per call, `no-session.jsonl` for calls without a session id, and `other-requests.jsonl` for other requests, such as `GET /v1/models`.

A real Codex call is large. Each request carries Codex's instructions (as `developer` messages), the tool definitions, and the whole conversation so far, including every earlier tool call and its output; in the mock coding task, each request body was about 42 KB. Below is one record, shortened. It is synthetic, from a mock backend and a hand-written request, because real records hold complete prompts.

```json
{
  "kind": "llm_call", "seq": 0, "call_id": "call_621ce506746f1eb2",
  "session_key": "019a0000-0000-7000-8000-000000000001", …,
  "upstream": {"name": "mock", "url": "http://127.0.0.1:59322/v1/responses", "peer": "127.0.0.1:59322",
               "changes": ["request: Host set to the upstream host", …]},
  "request": {
    "method": "POST", "path": "/v1/responses", …,
    "headers": [["content-type", "application/json"], ["authorization", "[redacted]"], …],
    "body": {"text": "{\"model\": \"gpt-test\", \"input\": [ … ], \"stream\": true, …}", "json": { … }}
  },
  "response": {
    "source": "upstream", "status": 200,
    "headers": [["content-type", "text/event-stream"], …],
    "body": {"text": "event: response.created\ndata: {…}\n\n … event: response.completed\ndata: {…}\n\n"}
  },
  "stream": {
    "events": [{"index": 0, "event": "response.created", "type": "response.created", "json": {…}}, …],
    "reconstruction": {
      "outcome": "completed", "terminal_event": "response.completed", "response_id": "resp-1", "usage": {…}, …,
      "items": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello from the mock."}], …}]
    }
  },
  "timing": {"started_at": "2026-10-24T22:27:14.563Z", "first_byte_ms": 51.081, "total_ms": 53.475, …},
  "outcome": "completed", …,
  "derived": {"request_bytes": 346, "response_bytes": 495, "event_count": 3, "item_types": ["message"], "tool_calls": [], …}
}
```

- `request` and `response` are the raw HTTP exchange. Bodies are kept byte for byte: as text, or as base64 if they are not UTF-8. This layer is the authoritative one.
- `stream` holds every parsed SSE event, and the output items rebuilt from them: messages, `function_call` and `custom_tool_call` items, reasoning, and types it does not know.
- `derived` holds figures computed from the layers above.
- `outcome` is `completed`, `incomplete` or `failed` when a stream ended with that terminal event; for a reply that is not a stream, `completed` means the whole body arrived. Otherwise it says what happened: `stream_cut`, `client_disconnected`, `recorder_stopped`, `upstream_http_error` or `proxy_error`.

## How streaming stays transparent

- Each chunk from the backend goes to Codex first, and only then to the parser, which works on its own copy. An event that is not valid JSON is recorded as a parse error. Measured: with events sent 200 ms apart, each one reached the client before the next was sent, with at most 1.63 ms added.
- The proxy never retries and has no read timeout, so it never cuts off a slow local model. Codex's own timeouts decide.
- CodexProbe changes headers only where HTTP requires it (hop-by-hop headers dropped, `Host` set, a chunked request body sent with a `Content-Length`) and where the auth mode says so, and the record lists each change. Each hop also does its own framing: a streamed reply reaches Codex with `Transfer-Encoding: chunked`, set by the proxy's HTTP server.
- In the recorded runs, OpenAI's ChatGPT backend streamed without a `Content-Type` header. Such a response is parsed as a stream only when its request was a `POST /v1/responses` that asked for `"stream": true`. The body itself is never sniffed.
- The outcome comes from the terminal event that CodexProbe parsed itself. Codex hangs up as soon as it sees `response.completed`, and the call still counts as `completed`.

## Responses vs. Chat Completions

- **Chat Completions** (`POST /v1/chat/completions`) takes a list of `messages`, and streams `choices[].delta` chunks; tool calls arrive as `tool_calls` fragments.
- **Responses** (`POST /v1/responses`) takes typed `input` items, such as messages with roles including `developer`, `function_call`, `function_call_output` and reasoning. It streams typed events, such as `response.output_item.done` and `response.completed`.

The task text says local servers speak Chat Completions. But Codex dropped Chat Completions in version 0.95.0, and in 0.164.0 `wire_api = "responses"` is the only accepted value. Ollama serves `/v1/responses` itself, since v0.13.3. So CodexProbe forwards Codex's requests unchanged and translates nothing. These facts come from the Codex and Ollama sources and release notes.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q tests          # 349 passed, 1 deselected; no network, model or Codex needed
```

They cover config validation; byte-for-byte passthrough of JSON, SSE and HTTP errors; SSE split at every byte position; live relay and slow backends; a client that hangs up before the reply, mid-stream or after `response.completed`, and `stop()` before the reply or mid-stream; the log schema, redaction and `load_session`; and the quickstart's checks, each one shown to fail on bad evidence. The deselected test runs the real Codex against the mock:

```powershell
$env:CODEXPROBE_CODEX = $codex; $env:CODEXPROBE_CODEX_HOME = $codexHome
.venv\Scripts\python.exe -m pytest -m e2e tests\test_quickstart.py
```

## Quickstart options

| Option | Meaning |
|---|---|
| `--mode mock\|openai\|qwen` | The backend. Required. |
| `--codex <path>` | The Codex CLI 0.164.0. Default: `codex` on PATH. |
| `--codex-home <dir>` | Codex's settings folder (CODEX_HOME). mock and qwen write `config.toml` there; default: a new temporary folder. On Windows, pass the same folder every time: according to Codex's source, each new one can leave another sandbox permission entry on `%TEMP%`. openai requires its own signed-in folder, with no `config.toml`. |
| `--task code\|text` | openai and qwen: the coding task (default), or a fixed one-word reply. |
| `--log-dir <dir>` | Default: `runs/` in this folder. |
| `--timeout <s>` | Default: 600 s; 3600 s for qwen. |
| `--ollama-url <url>` | Default: `http://127.0.0.1:11434/v1`. |

Exit codes: 0 PASS, 1 FAIL, 2 bad arguments, 3 BLOCKED. BLOCKED means that something the run needs is missing or wrong, such as the Codex or Ollama version, the model, the settings folder or the sign-in; nothing is faked.

## Project layout

| Path | Contents |
|---|---|
| `src/codex_probe/proxy.py` | `ProxyRecorder`: forwarding, live relay, outcomes, records |
| `src/codex_probe/streaming.py` | The SSE parser and the Responses stream reassembler |
| `src/codex_probe/storage.py` | The record schema, redaction, the JSONL writer, `load_session` |
| `src/codex_probe/config.py` | Config loading and validation |
| `src/codex_probe/_server.py` | The background HTTP server (uvicorn) |
| `src/codex_probe/mock_server.py` | A scriptable mock of the Responses API |
| `src/codex_probe/e2e_validation.py` | The checker for real Codex runs against the mock; `scripts/verify_e2e.py` is its command line |
| `src/codex_probe/codex_run.py` | Codex's config, and a guarded `codex exec` |
| `examples/quickstart.py` | The mock, openai and qwen demos |
| `examples/ollama/Modelfile` | The derived Qwen model |
| `tests/` | The test suite |

## Limitations

- Tested on one Windows 11 machine. macOS and Linux were not tested.
- OpenAI was tested only through a ChatGPT sign-in, not with an API key. To use a `base_url`, Codex needs a custom provider, which talks HTTP and SSE rather than WebSockets. So a capture is not identical to the traffic of Codex's built-in OpenAI provider.
- Logs hold complete prompts, which may include workspace files. With OpenAI, the response headers also carry account details, such as the plan and usage. Credential headers are redacted, but keep logs private.
- CodexProbe ignores `HTTP_PROXY` and `HTTPS_PROXY` (httpx `trust_env=False`), so it cannot reach OpenAI on a network that requires an outbound proxy. This was not tested.

## Prior art

According to Codex's source, Codex 0.164.0 has a hidden `codex responses-api-proxy --dump-dir <dir>`. It dumps each `POST /v1/responses`, but keeps a stream as one raw string, without parsing or sessions, refuses every other path, and replaces Codex's credentials with its own API key. Codex can also write a rollout trace (`CODEX_ROLLOUT_TRACE_ROOT`), which has no raw SSE bytes or HTTP headers. CodexProbe adds byte-exact records of every request under `/v1`, live SSE parsing and reassembly, explicit outcomes for cut or abandoned streams, per-session logs, and a config-only backend switch.

## References

- [Task 7: Codex LLM call recorder](https://github.com/STAR-Computing/ProgrammingTask/blob/master/07_codex_llm_call_recorder.md)
- [OpenAI Codex CLI](https://github.com/openai/codex), tag `rust-v0.164.0`; the [Responses API](https://platform.openai.com/docs/api-reference/responses); [Codex authentication](https://developers.openai.com/codex/auth)
- [Ollama](https://github.com/ollama/ollama), tag `v0.40.10`; [qwen2.5-coder](https://ollama.com/library/qwen2.5-coder)
- [VALIDATION.md](VALIDATION.md): the commands, outputs and evidence behind the results above

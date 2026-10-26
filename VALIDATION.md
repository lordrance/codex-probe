# Validation

This file records the runs behind the results below, and what came out. Every result comes from a command that was actually run on the test machine, and every number is copied from that run's output or logs. The raw run logs are not published, because they hold complete prompts and account headers. Run folders are named by their start time, for example `quickstart-openai-20261024-195557`.

## Summary

| What was tested | Result | Where |
|---|---|---|
| Install into a fresh venv from a clean copy, and the offline test suite | **PASS**: 349 passed, 1 deselected | Phases 1 and 5 |
| Real Codex → CodexProbe → a scripted mock backend: a coding task | **PASS**: every recorded body is byte-identical to the mock's own copy | Phase 2 |
| Real Codex, direct vs. through CodexProbe | **PASS**: the same request methods and paths, input item types and roles, tools, tool calls and files | Phase 2 |
| Real Codex → CodexProbe → OpenAI, signed in with ChatGPT: a fixed one-word reply | **PASS**, after one recorded FAIL that exposed a bug, and its fix | Phase 3 |
| Real Codex → CodexProbe → OpenAI: a coding task | **PASS**: 3 calls, all recorded; the files were written, and their test passes | Phase 3 |
| Real Codex → CodexProbe → a local Qwen on Ollama: a fixed one-word reply | **PASS**: the request reached the local model byte for byte, and the reply was recorded in full | Phase 4 |

## Environment

| Item | Value |
|---|---|
| Machine | Windows 11 Home (10.0.22000), x64; NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB; 16 GB of RAM |
| Python | 3.12.4 |
| Dependencies | `httpx==0.28.1` and `uvicorn==0.54.0`; for the tests, `pytest==9.1.1` and `pytest-timeout==2.4.0` |
| Codex CLI | 0.164.0: the npm package `@openai/codex@0.164.0`, installed into a folder of its own with Node v24.14.0 and npm 10.5.0. The global Codex and the default `%USERPROFILE%\.codex` were not used |
| Ollama | 0.40.10: the portable `ollama-windows-amd64.zip` from the GitHub release, whose sha256 matched the release's `sha256sum.txt` |
| Model | `qwen2.5-coder:1.5b`: 986,048,576 bytes, Q4_K_M, Apache-2.0 |

## Phase 1: the package, offline

`pytest -q tests` needs no network, model or Codex. It has 349 tests. Many of them are negative controls: bad input that a check must reject.

| File | Tests | What they check |
|---|---|---|
| `test_config.py` | 66 | Loading from a dict and from a JSON file, the defaults, and the rejection paths |
| `test_streaming.py` | 35 | The SSE parser and the Responses reassembler. A stream split at every byte position, fed one byte at a time, and split 200 random ways must parse the same as in one piece; also with CRLF and CR line endings. Comments, multi-line `data:`, every terminal event, and partial output for cut streams |
| `test_storage.py` | 62 | Bodies round-trip byte for byte: text, binary, JSON, a BOM, and 100 random byte strings. Credential headers are redacted whatever their case, and no secret value reaches the disk. One file per session. `load_session` rejects damaged files, naming the file and line |
| `test_mock_server.py` | 17 | The mock reproduces the SSE of Codex 0.164.0's own test helpers byte for byte. Scripted turns, timing and failure injection, and its own capture |
| `test_proxy.py` | 37 | Real HTTP through `ProxyRecorder` to the mock: byte-for-byte passthrough of SSE, JSON, non-UTF-8 bodies and HTTP errors; the exact header changes; the three auth modes; two configs that record two different backends with no code change; the live relay; slow upstreams; a client that hangs up before the reply, mid-stream or after `response.completed`, and `stop()` before the reply or mid-stream; the logs |
| `test_e2e_validation.py` | 18 | The checker for real Codex runs, with negative controls: a missing file, a model that only says it is done, seven kinds of damage to the log or the mock's capture, an altered function call, an extra file |
| `test_quickstart.py` | 114, plus 1 `e2e` | The quickstart's configs, its refusals (BLOCKED) and its checks, each check failing on its own negative control. The `e2e` test runs the real Codex against the mock and is left out by default |

For several fixes, the previous code was put back temporarily, to confirm that the new tests fail on it.

**Measured with the mock as the upstream** (`pytest tests/test_proxy.py -s`, over loopback):
- **Live relay.** Five events were sent 200 ms apart. Each reached the client through the proxy before the mock sent the next. The delay added per event, over three runs, was at most 1.63 ms. One value came out at −0.06 ms, because the mock takes its timestamp after its `send()` returns.
- **Slow upstream.** 6.2 s of silence before the response headers, then 6.2 s between two events. The stream arrived complete and was recorded as `completed`, with `first_byte_ms` 6213 and `total_ms` 12424. Both gaps are longer than httpx's default read timeout of 5 s.

## Phase 2: real Codex against a mock backend

The backend here is a local, scripted mock of the Responses API, not a model: it replays a fixed solution. These runs show that the real Codex CLI, CodexProbe and a Responses server work together, and that the recording is exact.

Codex ran as follows, with stdin closed and a new, empty workspace:

```
codex exec --json --skip-git-repo-check --strict-config -s workspace-write -c allow_login_shell=false -C <workspace> -o <file> "<prompt>"
```

Its `config.toml` is below. Only the port changed between runs, except that the first two runs (demo 1, and the first demo 2) had no `[windows]` section. The quickstart also writes a marker comment as the first line.

```toml
model = "codexprobe-mock"
model_provider = "codexprobe"
web_search = "disabled"

[model_providers.codexprobe]
name = "CodexProbe"
base_url = "http://127.0.0.1:<port>/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0

[features]
unbounded_connection_retries = false
plugins = false
apps = false

[analytics]
enabled = false

[windows]
sandbox = "unelevated"
```

The demos and the control experiment were run with a Windows-only development script that is not part of this repository. To repeat them, use `examples/quickstart.py --mode mock`, which runs demo 3's task with the seven checks listed below, and `scripts/verify_e2e.py`, whose `--direct-*` options compare a direct run with a proxied one.

| Run | Result |
|---|---|
| Demo 1, a fixed one-word reply (`demo1-20261024-084422`) | **PASS**. One call, with both bodies identical to the mock's bytes |
| Demo 2, one shell command, before `[windows] sandbox` was set (`demo2-20261024-084511`) | **FAIL**: Codex rejected the command as "blocked by policy", yet exited with 0 and answered `DONE`. The file check caught it |
| Demo 2 with `sandbox = "unelevated"` (`demo2-20261024-084630`) | **PASS** |
| Demo 3, the coding task (`demo3-20261024-090032`) | **PASS** |
| Control experiment, direct vs. through CodexProbe (`demo3-compare-20261024-090853`) | **PASS** |
| `examples/quickstart.py --mode mock` (`quickstart-20261024-092948`) | **PASS**, exit code 0 |
| The `e2e` test, run on its own (`pytest -m e2e tests\test_quickstart.py`) | **PASS** |

**The Windows sandbox.** Without a `[windows] sandbox` key, Codex 0.164.0 logged `Windows sandbox selection: disabled` on the test machine. `codex exec`, which cannot ask for approval, then rejected every command, while its exit code and final message still looked like success. The unelevated sandbox needs no administrator rights. In these runs it added permission (ACL) entries for its sandbox identities to each workspace, and once to `%TEMP%`; they stay after the run.

**The coding task (demo 3).** The mock scripts three turns:
1. an `exec_command` that runs an `apply_patch` adding `calc.py` and `test_calc.py`;
2. an `exec_command` that runs the tests with `python -m unittest -v`;
3. the final answer, which the mock sends only after both tool outputs have come back.

There were three `POST /v1/responses`, all `completed`. Every recorded request and response body is identical, byte for byte, to what the mock saved itself. Codex sent back both tool outputs: `Success. Updated the following files: A calc.py, A test_calc.py`, and `Process exited with code 0 … Ran 1 test … OK`. The checker then ran the tests itself: `Ran 1 test … OK`.

**The control experiment.** The same scripted task ran twice: once with Codex's `base_url` pointing straight at the mock, and once through CodexProbe. Everything else was identical. These were the same in both runs:
- the method and path of every request: 3 × `POST /v1/responses`;
- the input item types and roles, and the tool names, in every request;
- the function calls Codex sent back (name, call_id, arguments), and the call_ids of their outputs;
- every file in the two workspaces, byte for byte. Both test runs passed.

The raw request bodies were not identical: each proxied body was 8 bytes longer (41,955, 42,738 and 43,334 bytes direct, against 41,963, 42,746 and 43,342 through the proxy), because each run has its own session id and workspace path. No normalisation rule was needed.

**The quickstart's mock mode.** Codex exited with 0 after 2.6 s. Three calls were recorded, all `completed`. All seven checks passed: the exit code; `calc.py` (32 bytes) and `test_calc.py` (186 bytes) exist; their test passes when run again; the mock answered all 3 requests as scripted; both tool calls got their output back; the log holds 3 `completed` calls; and all 3 request and response bodies equal the mock's bytes.

## Phase 3: real Codex against OpenAI, through a ChatGPT sign-in

**The sign-in.** The user signed in with ChatGPT once, on OpenAI's own page, into a new, dedicated CODEX_HOME, the folder where Codex keeps its settings and sign-in (`codex login`). `codex login status` then reported `Logged in using ChatGPT`. Codex's `auth.json` was never opened or copied, and no API key was used. With this sign-in, Codex sends its model requests to `https://chatgpt.com/backend-api/codex`.

**The settings.** The openai mode passes the provider settings as `codex exec -c` overrides, and writes nothing into that CODEX_HOME:
- the provider `codexprobe`, named `CodexProbe`, with `base_url` set to the recorder;
- `wire_api = "responses"`, `requires_openai_auth = true` and `supports_websockets = false`;
- no retries; web search, plugins, apps and analytics off;
- `windows.sandbox = "unelevated"`.

CodexProbe forwards Codex's own credentials unchanged (`passthrough`) and redacts them in its log.

| Run | Result |
|---|---|
| Direct pre-check: Codex → OpenAI, without CodexProbe (`direct-20261024-192428`) | **PASS**: exit code 0, the reply `OPENAI_DIRECT_OK`, no tool |
| One-word reply through CodexProbe, first run (`quickstart-openai-20261024-193329`) | **FAIL**: see below |
| The same command after the fix (`quickstart-openai-20261024-195010`) | **PASS** |
| The coding task through CodexProbe (`quickstart-openai-20261024-195557`) | **PASS**, in 3 calls |

The direct pre-check rules out an account or backend problem. Its usage was 11,704 input tokens (9,856 of them cached) and 8 output tokens.

**The first reply run failed, and that exposed a bug.** Codex got the whole answer through the proxy (`OPENAI_PROXY_OK`), and the raw body was recorded in full: 96,289 bytes, 13 SSE events from `response.created` to `response.completed`. But the backend's HTTP 200 response had **no `Content-Type` header**, and CodexProbe then parsed only responses marked `text/event-stream`. So it parsed nothing, and when Codex hung up after `response.completed`, the call was recorded as `client_disconnected`. Fed offline into CodexProbe's own parser, the stored bytes gave `completed`, with the right message.

**The fix** is in `src/codex_probe/proxy.py` (`_is_responses_stream`). A 2xx response with no `Content-Type` at all is now also parsed as a stream, when its request was a `POST /v1/responses` whose JSON body has `"stream": true`. The body is never sniffed, an explicit other `Content-Type` is respected, and nothing that is forwarded changes. Seven new tests in `tests/test_proxy.py` cover the rule, and each of three wrong versions of it fails at least one of them. The same command, run once more, passed: 13 events parsed live, the outcome `completed`, the reply `OPENAI_PROXY_OK`, and usage of 11,720 input and 8 output tokens.

**The coding task.** The prompt: `Write calc.py with add(a, b) and a unittest test for it in test_calc.py, run the tests, and report.` The quickstart exited with 0 after 23 s.

| seq | Status | Parsed live | Outcome | The model's output |
|---|---|---|---|---|
| 0 | 200 | 76 events, `response.completed` | `completed` | A message, and one `exec` call that looked for files with `rg`. It exited with 1, because the workspace was empty |
| 1 | 200 | 224 events, `response.completed` | `completed` | One `exec` call: an `apply_patch` adding `calc.py` and `test_calc.py`, then `python -m unittest -v`, which exited with 0 (`Ran 1 test … OK`) |
| 2 | 200 | 54 events, `response.completed` | `completed` | The final message |

- **Tools.** The model was `gpt-6.1-sol`. In these runs, Codex offered its tools differently from the mock runs: they were listed inside the input, in an `additional_tools` item, and the model used one free-text tool, `exec`, whose input is a short script. So the model's calls are recorded as `custom_tool_call` items rather than `function_call` items. The scripts ran Codex's `exec_command` (seq 0 and 1) and `apply_patch` (seq 1).
- **History.** Each request carries the conversation so far, and every tool output appears under the same `call_id` in the next request.
- **Usage.** The per-call usage that CodexProbe parsed was 11,735/81, 11,912/227 and 12,233/49 input/output tokens. The sums, 35,880 and 357, equal the totals Codex itself reported.
- **Files.** The workspace held `calc.py` (32 bytes) and `test_calc.py` (265 bytes). The quickstart ran the tests again itself: 1 test, OK.
- **Headers.** Every response carried an `x-oai-request-id` set by OpenAI's server, and none had a `Content-Type`, so all three were parsed through the new rule.
- **The log.** `load_session()` on the run folder equals the list `stop()` returned. In every request, `authorization` and `chatgpt-account-id` are recorded as `[redacted]`, and a scan of the run's files found no token and no unredacted credential.
- **Other requests.** Codex's two `GET /v1/models` were forwarded, and logged separately in `other-requests.jsonl`.

## Phase 4: real Codex against a local Qwen, through Ollama

**Setup.** Ollama 0.40.10 ran on `127.0.0.1:11434`, with `OLLAMA_NO_CLOUD=1`. Its log reported the RTX 3050 Laptop GPU with 3.2 GiB of 4.0 GiB available, and a default context of 4096 tokens (`default_num_ctx=4096`). `ollama pull qwen2.5-coder:1.5b` was the only model download. The derived model was built from `examples/ollama/Modelfile` with `ollama create`, which reused the base weights.

**Direct checks against Ollama's `/v1/responses`,** with no Codex and no proxy:
- **Developer messages.** A passphrase was given only in a `developer` message. The base model could not repeat it: `input_tokens` was 40, because the base template dropped the message. The derived model repeated it, with `input_tokens` 65.
- **Text reply.** `QWEN_LOCAL_OK`, with the first byte after 216 ms and the whole reply after 261 ms, once loaded. The very first load from disk took 46 s.
- **Memory.** `ollama ps` showed `1.6 GB  100% GPU  CONTEXT 16384`. The GPU had 3,007 MiB of its 4,096 MiB in use, about 1,460 MiB of it by other programs.

**Codex settings for the local model.** The qwen mode writes the Phase 2 settings with `model = "codexprobe-qwen2.5-coder:1.5b"`, plus `model_context_window = 16384`, `stream_idle_timeout_ms = 600000`, and a minimal tool set that leaves Codex only `exec_command` and `write_stdin`: `model_reasoning_summary = "none"`, `view_image`, `tool_suggest` and `goals` off, `[agents] enabled = false`, `[tools.experimental_request_user_input] enabled = false` and `[skills] include_instructions = false`. Codex accepted every key under `--strict-config`.

**Codex → CodexProbe → Ollama: a fixed one-word reply** (`phase3-qwen-text-20261024-110701`; the folder name keeps an earlier phase number). The command:

```
python examples\quickstart.py --mode qwen --codex <codex 0.164.0> --codex-home <settings folder> --log-dir <run folder>
```

This run was made before the quickstart had its `--task` option. The qwen mode then ran only this check, which is now `--task text`. The prompt was `Reply with QWEN_LOCAL_OK. Do not use tools.`

The result: exit code 0, `RESULT: PASS`. Codex exited with 0 after 4.3 s, and replied `QWEN_LOCAL_OK`. One call was recorded: `completed`, no tool call, first byte after 3,623 ms (which includes loading the model) and 3,692 ms in total, with 5,132 input and 6 output tokens.

The evidence that the call really reached the local model:

| Evidence | What was found |
|---|---|
| The request body | The 25,289 bytes that CodexProbe recorded are byte-identical, by sha256, to the body Ollama logged itself (`OLLAMA_DEBUG_LOG_REQUESTS`). It was the only `/v1/responses` request Ollama received |
| The timing | Ollama's access log has the same request: status 200, 3.64 s. CodexProbe recorded 3.69 s |
| The peer | `127.0.0.1:11434`, at the URL `http://127.0.0.1:11434/v1/responses` |
| The request | model `codexprobe-qwen2.5-coder:1.5b`; tools `exec_command` and `write_stdin` only, the minimal tool set |
| The response | HTTP 200, 14,891 bytes, ending with a complete `response.completed`. Its id, `resp_587219`, has Ollama's form |
| Ollama's side | Its log shows the derived model loaded, and `ollama ps` showed `codexprobe-qwen2.5-coder:1.5b  23a7228ddd20  1.6 GB  100% GPU  16384` |
| No command ran | Codex's events hold one `agent_message` and no command, and the workspace stayed empty |

## Phase 5: the published code, from a clean copy

On 2026-10-25 the published code was committed, and `git archive` exported that commit into a new, empty folder: 24 files, with no `.venv`, `runs/`, cache or credential. In that folder (`publish-20261025-195530`):

| Step | Command | Result |
|---|---|---|
| venv | `python -m venv .venv`, with Python 3.12.4 | exit code 0 |
| install | `.venv\Scripts\python.exe -m pip install -e ".[test]"` | exit code 0. It installed anyio 4.15.1, certifi 2026.10.23, click 8.5.0, colorama 0.4.6, h11 0.16.0, httpcore 1.0.9, httpx 0.28.1, idna 3.20, iniconfig 2.3.1, packaging 26.3, pluggy 1.7.0, Pygments 2.21.0, pytest 9.1.1, pytest-timeout 2.4.0, typing_extensions 4.16.0 and uvicorn 0.54.0 |
| tests | `.venv\Scripts\python.exe -m pytest -q tests` | `349 passed, 1 deselected in 95.59s` |
| mock demo | `.venv\Scripts\python.exe examples/quickstart.py --mode mock --codex <codex 0.164.0> --codex-home <settings folder> --timeout 600` | exit code 0, `RESULT: PASS` |
| the README's log reader, on that run's log | the command in the README's Quick start, step 4 | `0 completed ['function_call']`, `1 completed ['function_call']`, `2 completed ['message']` |
| the README's API snippet, offline | the snippet, with `log_dir` in a temporary folder | `start()` returned an endpoint; `load_session()` returned the same list as `stop()` |
| the README's two upstream configs | `load_config` on each | both accepted; default names `chatgpt.com` and `127.0.0.1:11434` |

In the mock demo, Codex exited with 0 after 2.1 s. Three calls were recorded, all `completed`, and all seven checks passed, including the byte-for-byte equality of all 3 request and response bodies with the mock's copy. The output of every step is saved in that folder.

After this run, only `README.md` and `VALIDATION.md` changed.

**Which code the real runs used.** The two OpenAI runs after the fix (`quickstart-openai-20261024-195010` and `quickstart-openai-20261024-195557`) used the code with that fix. The later changes do not touch the path those runs took: the openai mode now refuses a CODEX_HOME that holds a `config.toml`, before anything runs; and some comments, docstrings and tests changed. The Qwen run of Phase 4 predates the `--task` option. Neither real backend was run again on the final code.

## Not tested

- macOS and Linux, and any other machine: every run, including the clean copies, was on the test machine.
- OpenAI through an API key (`https://api.openai.com/v1`).
- A global `npm install -g` of Codex, and two of the quickstart's defaults: `codex` on PATH, and a new temporary CODEX_HOME.
- A network that needs an outbound HTTP proxy.

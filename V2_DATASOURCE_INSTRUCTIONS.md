# V2 Datasource Authoring Instructions

*Version: 2.0*
*Last updated: 2026-03-17*

This document is the canonical source for creating Subjective v2 datasources. It contains everything an LLM or developer needs to generate a fully working v2 datasource plugin for any technology.

Use it when implementing a new Subjective datasource manually or when prompting an LLM to generate one.

---

## 1) Purpose

Write a Subjective v2 datasource that works correctly with:

- the connection editor (connection fields displayed in the UI)
- the Send dialog (on-demand chat messaging)
- the pipeline editor (input/output port wiring)
- the datasource launcher (subprocess execution)
- the v2 pipeline runner (batch + streaming + event-driven execution)

---

## 2) Plugin Folder Structure

Every datasource plugin lives in its own folder under the plugins directory. The folder name **must** follow the convention `subjective_<technology>_datasource`.

```
subjective_<technology>_datasource/
├── Subjective<Technology>DataSource.py   # Main datasource class (required)
├── pyproject.toml                         # Dependencies and metadata (required)
├── icon.svg                               # Icon displayed in the UI (required)
└── README.md                              # Optional description
```

**Naming rules:**
- Folder: `subjective_<technology>_datasource` (lowercase, underscores)
- Python file: `Subjective<Technology>DataSource.py` (PascalCase, must end with `DataSource.py`)
- Class inside the file: `Subjective<Technology>DataSource` (must match filename without `.py`)
- The framework discovers plugins by scanning for folders matching `subjective_*_datasource` and files matching `*DataSource.py` that contain the expected class name.

### pyproject.toml template

```toml
[project]
name = "subjective_<technology>_datasource"
version = "0.1.0"
description = "Subjective data source plugin: <Technology>."
requires-python = ">=3.12"
dependencies = [
  "subjective-abstract-data-source-package @ git+https://github.com/PabloBorda/subjective-abstract-data-source-package.git",
  "brainboost-data-source-logger-package @ git+https://github.com/PabloBorda/brainboost_data_source_logger_package.git",
  "brainboost-configuration-package @ git+https://github.com/PabloBorda/brainboost_configuration_package.git",
  # Add technology-specific dependencies here, e.g.:
  # "anthropic>=0.28.0",
  # "openai>=1.0.0",
  # "requests>=2.28.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The framework uses `uv` to install plugin dependencies in an isolated virtual environment based on `requires-python`. Each datasource subprocess runs in this environment.

---

## 3) API Version Auto-Detection

You do **NOT** need to set `_subjective_api_version` manually. The base class `__init_subclass__` hook detects v2 automatically based on which methods your class defines:

- If your class defines `connection_schema`, `run`, `stream`, or `handle_message` → auto-detected as **v2**
- If your class defines `get_connection_data`, `fetch`, `get_icon`, or `_process_message` → stays as **v1**

**Enforcement:** If detected as v2, the framework raises `TypeError` at class definition time if `run()` or `connection_schema()` is missing. This catches errors immediately, not at runtime.

---

## 4) Required V2 Contract

Every v2 datasource **must**:

1. Inherit from `SubjectiveDataSource` (from `subjective_abstract_data_source_package`).
2. Define `__init__(self, **kwargs)` and call `super().__init__(**kwargs)` as the first line.
3. Read saved connection values from `self._connection` (set by the framework from persisted connection data).
4. Implement `connection_schema()` as a `@classmethod` — declares the connection editor fields.
5. Implement `run(self, request: dict) -> dict` — the main execution logic.

Every v2 datasource **should**:

6. Implement `request_schema()` as a `@classmethod` when the datasource accepts per-request inputs (pipeline input ports).
7. Implement `output_schema()` as a `@classmethod` when the datasource exposes stable output fields (pipeline output ports).
8. Implement `icon()` as a `@classmethod` that reads `icon.svg` from the plugin folder.

---

## 5) What `super().__init__(**kwargs)` Gives You

After calling `super().__init__(**kwargs)`, the following attributes are available:

| Attribute | Type | Source | Description |
|-----------|------|--------|-------------|
| `self._connection` | `dict` | Persisted connection data | Connection-specific fields (API keys, URLs, model names). Populated from the connection editor form. |
| `self._config` | `dict` | Framework runtime config | Contains `connection_name`, `context_dir`, `TARGET_DIRECTORY`, `output_dir`, `request_data`, `ds_connection_tmp_space`. |
| `self.input_dir` | `str` (property) | `self._config["input_dir"]` | Directory where the pipeline runner places files routed from upstream nodes. |
| `self.output_dir` | `str` (property) | `self._config["output_dir"]` | Directory for context output / persistent results. |
| `self.scratch_dir` | `str` (property) | `self._config["scratch_dir"]` | Temporary workspace for intermediate files. |
| `self.connection_name` | `str` (property) | `self._config["connection_name"]` | The user-assigned connection name. |
| `self.params` | `dict` | Empty for v2 | Legacy attribute. Empty `{}` for v2 datasources. Do not rely on it. |

**Important:** The framework instantiates v2 datasources with `ds_class(connection=connection_data, config=runtime_config)`. The base class `__init__` routes this to `_init_v2()` which populates all the attributes above. You never need to handle these arguments directly — just use `**kwargs`.

---

## 6) Schema Declarations

### 6.1 connection_schema() — Connection Editor Fields

Declares persistent fields shown in the connection editor. These are stored with the connection and passed to the constructor via `self._connection`.

```python
@classmethod
def connection_schema(cls):
    return {
        "api_key": {
            "type": "password",
            "label": "API Key",
            "required": True,
            "placeholder": "sk-...",
        },
        "model": {
            "type": "select",
            "label": "Model",
            "options": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "default": "gpt-4o-mini",
        },
        "api_base_url": {
            "type": "url",
            "label": "API Base URL",
            "default": "https://api.openai.com/v1",
        },
    }
```

**Supported field types:**

| Type | Description | Extra attributes |
|------|-------------|------------------|
| `text` | Single-line text input | `placeholder`, `default` |
| `password` | Masked text input (for secrets) | `placeholder` |
| `url` | URL input | `placeholder`, `default` |
| `email` | Email input | `placeholder` |
| `number` | Numeric input | `default`, `min`, `max`, `step` |
| `int` | Integer input | `default`, `min`, `max` |
| `select` | Dropdown selector | `options` (list of strings), `default` |
| `textarea` | Multi-line text | `placeholder`, `default`, `rows` |
| `checkbox` / `bool` | Boolean toggle | `default` (True/False) |
| `file_path` | File picker | `placeholder` |
| `folder_path` | Folder picker | `placeholder` |

**Common attributes for all types:** `label` (display name), `required` (bool), `default`, `placeholder`, `description` (help text).

**Forbidden fields** — the framework manages these, never include them:
`connection_name`, `server`, `server_ip`, `ip`, `ip_address`, `ds_class`, `ds_type`, `start_at_startup`, `is_local_folder`

### 6.2 request_schema() — Per-Run Input Ports

Declares what the datasource accepts as input when `run(request)` is called. In the pipeline editor, these become input ports that can be wired from upstream nodes.

```python
@classmethod
def request_schema(cls):
    return {
        "prompt": {
            "type": "text",
            "label": "Prompt",
            "required": True,
        },
        "max_tokens": {
            "type": "int",
            "label": "Max Tokens",
            "default": 1024,
        },
    }
```

The `request` dict passed to `run()` will contain these keys with values either from the pipeline wiring or from user input.

### 6.3 output_schema() — Output Ports

Declares the stable keys that `run()` returns. In the pipeline editor, these become output ports that downstream nodes can wire to.

```python
@classmethod
def output_schema(cls):
    return {
        "response": {
            "type": "text",
            "label": "Response Text",
        },
        "model": {
            "type": "text",
            "label": "Model Used",
        },
        "usage_tokens": {
            "type": "int",
            "label": "Tokens Used",
        },
    }
```

**Rule:** The dict returned by `run()` must include all keys declared in `output_schema()`.

### 6.4 actions() — Multi-Action Datasources (Optional)

For datasources that support multiple distinct operations (e.g., a 3D API with "preview" and "refine" actions), declare an `actions()` classmethod. Each action has its own request and output schema:

```python
@classmethod
def actions(cls):
    return {
        "preview": {
            "label": "Generate Preview",
            "request": {
                "prompt": {"type": "text", "label": "Prompt", "required": True},
            },
            "output": {
                "task_id": {"type": "text", "label": "Task ID"},
                "status": {"type": "text", "label": "Status"},
            },
        },
        "refine": {
            "label": "Refine Model",
            "request": {
                "preview_task_id": {"type": "text", "label": "Preview Task ID", "required": True},
                "texture_prompt": {"type": "text", "label": "Texture Prompt"},
            },
            "output": {
                "task_id": {"type": "text", "label": "Refined Task ID"},
                "model_url": {"type": "text", "label": "Model URL"},
            },
        },
    }
```

When `actions()` is defined, `request_schema()` and `output_schema()` automatically merge all action schemas (unless you override them). The pipeline editor shows a `selected_action` field for the node.

---

## 7) Execution Patterns

### 7.1 Batch Datasource (most common)

A datasource that receives a request, does work, and returns a result.

```python
def run(self, request: dict) -> dict:
    prompt = request.get("prompt", "")
    # ... call external API ...
    return {
        "response": result_text,
        "model": self.model,
        "usage_tokens": token_count,
    }
```

- Called once per pipeline execution (or once per iteration if upstream provides arrays).
- Must return a dict whose keys match `output_schema()`.
- Must be JSON-serializable when practical.

### 7.2 Streaming Datasource

A datasource that runs indefinitely, emitting events over time (folder monitors, real-time feeds, tickers).

```python
def supports_streaming(self) -> bool:
    return True

def stream(self, request: dict):
    """Yield events indefinitely. Each yield triggers downstream nodes."""
    while True:
        event = self._poll_for_event()
        if event:
            yield {
                "path": event.path,
                "event_type": event.type,
                "timestamp": time.time(),
            }
        time.sleep(1.0)
```

- The pipeline runner starts streaming nodes in **daemon threads**.
- Each `yield` triggers execution of all downstream batch nodes with the yielded data wired through ports.
- **Blocking is fine** — each datasource runs in its own OS process.
- `stream()` must also have `connection_schema()`, `request_schema()`, and `output_schema()` like any other datasource.

### 7.3 On-Demand Chat Datasource

A datasource that stays alive and responds to user messages sent via the virtual glass or the Send dialog.

```python
def supports_chat(self) -> bool:
    return True

def handle_message(self, message: str, files: list | None = None) -> Any:
    """Process a single chat message and return the response."""
    if not self._connection.get("api_key"):
        return {"error": "Missing API key"}

    # ... call LLM API with message ...
    return {"response": response_text, "model": self.model}

def run(self, request: dict) -> dict:
    """Initial execution — just return ready status."""
    return {"status": "ready"}
```

**Lifecycle for on-demand datasources:**
1. Framework calls `run(request)` — return immediately with a status.
2. Framework detects `supports_chat() == True` and sets up a Redis chat listener.
3. User sends messages via virtual glass or Send dialog → `handle_message()` is called.
4. Responses are published back via Redis to the caller.

**File attachments:** When the user sends files alongside a message, they arrive in the `files` parameter as a list of dicts with keys: `name`, `path`, `size`, `mime_type`, and either `text` (for text files) or `data_base64` (for binary files).

---

## 8) Reading Connection Values

Always read connection values from `self._connection` in `__init__`:

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.api_key = self._connection.get("api_key", "")
    self.model = self._connection.get("model", "default-model")
    self.api_base_url = self._connection.get("api_base_url", "https://api.example.com")
```

**Why `self._connection` and not `self.params`?**
- `self._connection` contains persisted connection data (API keys, URLs, models) from the connection editor.
- `self.params` is empty `{}` for v2 datasources — it's a v1 legacy attribute.
- The framework explicitly separates connection metadata from request data. Connection data goes to the constructor; request data goes to `run()`.

---

## 9) Temp Storage

If your datasource needs to write intermediate files (downloads, caches, working data) that should not be treated as final output:

```python
tmp_dir = self.get_connection_temp_dir()
# Returns a path like: <tmp_root>/<connection_name>/
# The directory is created automatically.
```

This is scoped per-connection, isolated from other connections, and cleaned up separately from context output.

---

## 10) Icon

Provide an `icon.svg` file in the plugin folder. Implement `icon()` to load it:

```python
@classmethod
def icon(cls):
    icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
    if os.path.exists(icon_path):
        with open(icon_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""
```

The SVG is displayed in the connection list and pipeline editor. Keep it simple (24x24 viewport recommended).

---

## 11) Important Rules

- Do **NOT** put framework-managed fields in `connection_schema()`. See the forbidden fields list in section 6.1.
- Do **NOT** add legacy v1 methods (`fetch()`, `get_connection_data()`, `get_icon()`, `_process_message()`). These trigger v1 detection and break the v2 contract.
- Do **NOT** implement progress tracking, context-output writing, or subscriber wiring inside the datasource. The v2 framework handles these externally.
- **DO** return a dict from `run()` whose keys match `output_schema()`.
- **DO** return plain Python data that is JSON-serializable.
- **DO** handle errors gracefully — return error dicts rather than raising exceptions when practical, so the pipeline runner can log and continue.

---

## 12) Complete Examples

### 12.1 Batch API Datasource

```python
import os
from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveExampleApiDataSource(SubjectiveDataSource):
    """Batch datasource that calls an external API."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self._connection.get("api_key", "")
        self.model = self._connection.get("model", "default")
        self.api_base_url = self._connection.get("api_base_url", "https://api.example.com")

    @classmethod
    def connection_schema(cls):
        return {
            "api_key": {
                "type": "password",
                "label": "API Key",
                "required": True,
                "placeholder": "your-api-key",
            },
            "model": {
                "type": "select",
                "label": "Model",
                "options": ["default", "large", "fast"],
                "default": "default",
            },
            "api_base_url": {
                "type": "url",
                "label": "API Base URL",
                "default": "https://api.example.com",
            },
        }

    @classmethod
    def request_schema(cls):
        return {
            "prompt": {
                "type": "text",
                "label": "Prompt",
                "required": True,
            },
        }

    @classmethod
    def output_schema(cls):
        return {
            "response": {"type": "text", "label": "Response"},
            "model": {"type": "text", "label": "Model Used"},
        }

    @classmethod
    def icon(cls):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def run(self, request: dict) -> dict:
        prompt = request.get("prompt", "")
        if not self.api_key:
            return {"error": "Missing API key", "response": "", "model": self.model}

        BBLogger.log(f"Calling API with model={self.model}, prompt={prompt[:50]}...")

        import requests
        resp = requests.post(
            f"{self.api_base_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "prompt": prompt},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "response": data.get("text", ""),
            "model": self.model,
        }
```

### 12.2 Streaming Datasource (Folder Monitor)

```python
import os
import time
from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveExampleMonitorDataSource(SubjectiveDataSource):
    """Streaming datasource that watches a folder for new files."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.watch_path = self._connection.get("watch_path", "")
        self.poll_interval = float(self._connection.get("poll_interval", 2.0))

    @classmethod
    def connection_schema(cls):
        return {
            "watch_path": {
                "type": "folder_path",
                "label": "Watch Folder",
                "required": True,
            },
            "poll_interval": {
                "type": "number",
                "label": "Poll Interval (seconds)",
                "default": 2.0,
                "min": 0.5,
            },
        }

    @classmethod
    def output_schema(cls):
        return {
            "path": {"type": "text", "label": "File Path"},
            "filename": {"type": "text", "label": "File Name"},
            "event_type": {"type": "text", "label": "Event Type"},
        }

    @classmethod
    def icon(cls):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def supports_streaming(self) -> bool:
        return True

    def stream(self, request: dict):
        """Yield an event each time a new file appears in the watched folder."""
        seen = set()
        if os.path.isdir(self.watch_path):
            seen = set(os.listdir(self.watch_path))

        BBLogger.log(f"Monitoring {self.watch_path} (poll every {self.poll_interval}s)")

        while True:
            try:
                if os.path.isdir(self.watch_path):
                    current = set(os.listdir(self.watch_path))
                    new_files = current - seen
                    for filename in sorted(new_files):
                        full_path = os.path.join(self.watch_path, filename)
                        BBLogger.log(f"New file detected: {full_path}")
                        yield {
                            "path": full_path,
                            "filename": filename,
                            "event_type": "created",
                        }
                    seen = current
            except Exception as e:
                BBLogger.log(f"Monitor error: {e}")

            time.sleep(self.poll_interval)

    def run(self, request: dict) -> dict:
        """Batch fallback: list current folder contents."""
        files = []
        if os.path.isdir(self.watch_path):
            files = os.listdir(self.watch_path)
        return {"path": self.watch_path, "filename": "", "event_type": "scan", "files": files}
```

### 12.3 On-Demand Chat Datasource

```python
import os
from typing import Any
from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveExampleChatDataSource(SubjectiveDataSource):
    """On-demand chat datasource that responds to user messages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self._connection.get("api_key", "")
        self.model = self._connection.get("model", "default-chat")
        self.system_prompt = self._connection.get("system_prompt", "")
        self._conversation = []

    @classmethod
    def connection_schema(cls):
        return {
            "api_key": {
                "type": "password",
                "label": "API Key",
                "required": True,
            },
            "model": {
                "type": "select",
                "label": "Model",
                "options": ["default-chat", "large-chat"],
                "default": "default-chat",
            },
            "system_prompt": {
                "type": "textarea",
                "label": "System Prompt",
                "placeholder": "You are a helpful assistant.",
                "rows": 4,
            },
        }

    @classmethod
    def icon(cls):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def supports_chat(self) -> bool:
        return True

    def handle_message(self, message: str, files: list | None = None) -> Any:
        """Process a chat message and return the response."""
        if not self.api_key:
            return {"error": "Missing API key"}

        BBLogger.log(f"Chat message received: {str(message)[:100]}...")

        self._conversation.append({"role": "user", "content": message})

        import requests
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._conversation)

        resp = requests.post(
            "https://api.example.com/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages},
            timeout=60,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")

        self._conversation.append({"role": "assistant", "content": response_text})
        return response_text

    def run(self, request: dict) -> dict:
        """Initial execution — return ready status. Chat happens via handle_message()."""
        return {"status": "ready", "model": self.model}
```

---

## 13) How the Framework Executes Your Datasource

Understanding the execution flow helps avoid common mistakes:

1. **User creates a connection** in the connection editor → form fields saved to `per.sis` as `connection_data`.
2. **User starts the connection** → UI calls `start_data_source_on_server()`.
3. **DataSourceConnector** extracts `connection_data` from persisted record, sends to DataSourceManager.
4. **DataSourceManager** launches `datasource_launcher.py` as a subprocess with `--connection-data` and `--request-data` as separate CLI arguments.
5. **Launcher** detects `api_version == "v2"`, instantiates: `ds_class(connection=connection_data, config=runtime_config)`.
6. **Launcher** calls `ds_instance.run(request_data)` (batch) or iterates `ds_instance.stream(request_data)` (streaming).
7. **If `supports_chat()`** → launcher also starts a Redis listener thread on `ondemand_chat_request_{pid}`. Incoming messages invoke `handle_message()`. Responses publish to `ondemand_chat_response_{pid}`.
8. **For on-demand datasources** — the process stays alive after `run()` returns, waiting for chat messages.

**In a pipeline:**
- The pipeline runner instantiates your datasource with `connection_data` resolved from the connection record.
- `run(request)` receives a dict built from upstream node outputs wired through input ports.
- Your return dict is available to downstream nodes through output ports.
- If your datasource is streaming, the runner starts it in a daemon thread and each `yield` triggers downstream execution.

---

## 14) LLM Prompt Template

Copy this into the LLM and fill in the placeholders:

```text
You are implementing a Subjective v2 datasource plugin.

== Plugin Location ==
Plugin folder: C:\Users\pablo\.Subjective\com_subjective_userdata\com_subjective_plugins\subjective_<technology>_datasource\
Main file: Subjective<Technology>DataSource.py
Class name: Subjective<Technology>DataSource

== V2 Contract (all required) ==
1. Inherit from SubjectiveDataSource (from subjective_abstract_data_source_package).
2. Constructor: `def __init__(self, **kwargs)` → call `super().__init__(**kwargs)` first.
3. Read connection values from `self._connection` dict (populated by the framework from persisted connection data).
4. Implement `connection_schema(cls)` as @classmethod — declares connection editor fields.
5. Implement `run(self, request: dict) -> dict` — main execution logic.
6. Implement `request_schema(cls)` as @classmethod — declares per-run input ports (optional).
7. Implement `output_schema(cls)` as @classmethod — declares output ports matching run() return keys (optional).
8. Implement `icon(cls)` as @classmethod that reads icon.svg from the plugin folder.

== Rules ==
- Do NOT add v1 methods: fetch(), get_connection_data(), get_icon(), _process_message().
- Do NOT put framework fields in connection_schema(): connection_name, server, ip, ds_class, ds_type, start_at_startup, is_local_folder.
- Return a JSON-serializable dict from run() whose keys match output_schema().
- For streaming: implement supports_streaming() → True and stream(request) as a generator.
- For chat: implement supports_chat() → True and handle_message(message, files=None).

== Schema Field Types ==
text, password, url, email, number, int, select (with "options" list), textarea, checkbox/bool, file_path, folder_path.
Common attributes: label, required, default, placeholder, description, options (for select), min/max/step (for number/int).

== Available Framework Properties (after super().__init__) ==
- self._connection: dict — persisted connection fields (API keys, URLs, etc.)
- self._config: dict — framework runtime config
- self.input_dir: str — pipeline input directory
- self.output_dir: str — context output directory
- self.scratch_dir: str — temporary workspace
- self.connection_name: str — user-assigned connection name
- self.get_connection_temp_dir(): str — per-connection temp storage

== What This Datasource Should Do ==
<describe the technology, API, behavior, input/output>

== Deliverable ==
- A single production-ready Python file: Subjective<Technology>DataSource.py
- A pyproject.toml with the correct dependencies
- Keep the code clear and self-contained.
```

---

## 15) Validation Checklist

Before considering a v2 datasource complete, verify:

- [ ] `py_compile` passes for the datasource Python file
- [ ] Class inherits from `SubjectiveDataSource`, not from deprecated wrappers
- [ ] Constructor is `def __init__(self, **kwargs)` with `super().__init__(**kwargs)` as first line
- [ ] `connection_schema()` contains only datasource-specific fields (no forbidden framework fields)
- [ ] `connection_schema()` field types are valid (text, password, select, number, etc.)
- [ ] Connection values read from `self._connection` (not `self.params`)
- [ ] `run(request)` returns a dict whose keys match `output_schema()` when declared
- [ ] No v1 methods present (`fetch`, `get_connection_data`, `get_icon`, `_process_message`)
- [ ] `pyproject.toml` lists all required dependencies including the abstract package
- [ ] `icon.svg` exists in the plugin folder
- [ ] If streaming: `supports_streaming()` returns `True` and `stream(request)` yields dicts
- [ ] If chat: `supports_chat()` returns `True` and `handle_message()` returns a response
- [ ] Error cases return error dicts rather than raising unhandled exceptions

---

## 16) Related Architecture References

- `com_subjective_architecture_docs/03_datasource_framework_and_plugins.md` — Plugin framework, discovery, dependency management
- `com_subjective_architecture_docs/26_datasource_api_v2_refactor.md` — Full v2 specification and design rationale
- `com_subjective_architecture_docs/04_pipeline_system.md` — Pipeline execution model (batch + streaming + event-driven)
- `com_subjective_architecture_docs/datasource_migration_v1_v2.md` — Step-by-step migration guide from v1 to v2

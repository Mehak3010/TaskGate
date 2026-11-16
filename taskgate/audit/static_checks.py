"""Static audit: AST and filesystem rules. No containers, no model calls.

Every rule is implemented as its own function so the unit tests can hit the
positive and the negative case for each one in isolation.
"""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..models import Finding, Severity, Task
from .rules import rule as get_rule

MAX_FILE_BYTES = 400_000

TEXT_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".ps1", ".txt", ".md", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".tsv", ".js", ".ts",
    ".sql", ".dockerfile", ".conf", ".env", ".lock", ".r", ".go", ".rb",
}
TEXT_FILENAMES = {"Dockerfile", "dockerfile", "Makefile", "requirements.txt", "task.yaml", "task.yml"}

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")

HIDDEN_NAMES = {"tests", "test"}
SOLUTION_NAMES = {"solution", "solutions", "solve"}
ANSWER_FILE_RE = re.compile(r"^(answer|answers|expected|golden|solution|solve|reference)\b", re.I)
PROBE_ANSWER_RE = re.compile(r"(answer|expected_output|golden|solution)\.(json|txt|yaml|yml|csv)$", re.I)

# Executables / calls that imply runtime network access. A bare URL is NOT here:
# mentioning a link is not a network call, so only the actual calls are listed.
NETWORK_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcurl\b", "curl"),
    (r"\bwget\b", "wget"),
    (r"\bgit\s+clone\b", "git clone"),
    (r"requests\.(get|post|put|head|request)\s*\(", "requests.<method>()"),
    (r"urllib\.request|\burlopen\s*\(", "urllib"),
    (r"\bsocket\.(create_connection|socket)\s*\(", "raw socket"),
    (r"\bnc\s+-", "netcat"),
    (r"\bssh\s+[\w.-]+@", "ssh"),
    (r"\bpip3?\s+install\b", "pip install at run time"),
    (r"\buv\s+(?:pip\s+install|add|sync)\b", "uv install at run time"),
    (r"\bnpm\s+(?:install|i|add|ci)\b", "npm install at run time"),
    (r"\bapt(-get)?\s+install\b|\bapk\s+add\b", "package manager at run time"),
)

NONDET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\btime\.sleep\s*\(", "time.sleep() is sleep-based synchronisation"),
    (r"\btime\.time\s*\(", "time.time()"),
    (r"\btime\.monotonic\s*\(", "time.monotonic()"),
    (r"\bdatetime\.now\s*\(|\bdatetime\.utcnow\s*\(", "datetime.now()/utcnow()"),
    (r"\bdate\.today\s*\(", "date.today()"),
    (r"\bos\.urandom\s*\(|\buuid\.uuid4\s*\(", "os.urandom()/uuid4()"),
    (r"\brandom\.(random|randint|choice|choices|shuffle|sample)\s*\(", "random without a seed"),
    (r"\blist\s*\(\s*set\s*\(", "list(set(...)) has no defined order"),
    (r"\bnext\s*\(\s*iter\s*\(\s*set\s*\(", "next(iter(set(...))) has no defined order"),
)

WEAK_ASSERT_CALLS = {
    "exists", "isfile", "isdir", "is_file", "is_dir", "os.path.exists",
    "os.path.isfile", "os.path.isdir", "pathlib.Path.exists",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFile:
    rel: str
    path: Path
    text: str


def _is_texty(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def _read(rel: str, path: Path) -> SourceFile | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "\x00" in text[:2000]:
        return None
    return SourceFile(rel=rel, path=path, text=text)


def iter_sources(task: Task) -> Iterator[SourceFile]:
    for rel in task.rel_files:
        path = task.file(rel)
        if _is_texty(path):
            src = _read(rel, path)
            if src is not None:
                yield src


def sources_under(task: Task, *prefixes: str) -> Iterator[SourceFile]:
    wanted = tuple(p.rstrip("/") + "/" for p in prefixes)
    for src in iter_sources(task):
        rel = src.rel
        if any(rel.startswith(p) for p in wanted):
            yield src


def dockerfile_source(task: Task) -> SourceFile | None:
    path = task.dockerfile_path
    if path is None:
        return None
    return _read(path.name, path)


def compose_source(task: Task) -> SourceFile | None:
    for name in COMPOSE_FILENAMES:
        candidate = task.path / name
        if candidate.is_file():
            return _read(name, candidate)
    return None


def test_sources(task: Task) -> list[SourceFile]:
    """The hidden verifier: everything under tests/ plus root-level test files."""
    out: list[SourceFile] = list(sources_under(task, "tests"))
    for src in iter_sources(task):
        if "/" not in src.rel and (src.rel.startswith("test_") or src.rel.startswith("check")):
            out.append(src)
    return out


def solution_sources(task: Task) -> list[SourceFile]:
    return list(sources_under(task, "solution"))


def make_finding(
    rule_id: str,
    file: str,
    line: int | None,
    message: str,
    severity: Severity | None = None,
) -> Finding:
    """Build a finding. ``severity`` overrides the registry default for the few
    cases where the same rule carries different weight depending on where it fires
    (e.g. S006 in tests/ is a BLOCKER, in solution/ only a WARN)."""
    r = get_rule(rule_id)
    return Finding(
        rule_id=rule_id,
        severity=severity or r.severity,
        file=file,
        line=line,
        message=message,
        fix_hint=r.fix_hint,
    )


def _safe_split(text: str) -> list[str]:
    """``shlex.split`` that survives a shell line it cannot tokenise.

    A Dockerfile ``RUN`` line ending in a backslash (a shell line-continuation)
    makes ``shlex.split`` raise ``ValueError: No escaped character``. Falling back
    to a whitespace split keeps the audit running instead of aborting the rule.
    """
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join Dockerfile line-continuations, returning (first_line_no, joined).

    A multi-line ``RUN a \\\n && b`` becomes one string so a rule can reason about
    the whole shell command rather than a fragment that ends in a backslash.
    """
    out: list[tuple[int, str]] = []
    buf = ""
    start = 1
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not buf:
            start = i
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        out.append((start, buf))
        buf = ""
    if buf:
        out.append((start, buf))
    return out


def _line_of(text: str, needle: str) -> int | None:
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    return None


def _line_matches(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            out.append((i, line.strip()))
    return out


# -- Dockerfile / compose parsing ------------------------------------------
COPY_RE = re.compile(r"^\s*(COPY|ADD)\b(?P<args>.*)$", re.I)
RUN_RE = re.compile(r"^\s*RUN\b(?P<args>.*)$", re.I)
FROM_RE = re.compile(r"^\s*FROM\s+(?P<image>\S+)(?:\s+AS\s+\S+)?\s*$", re.I)
USER_RE = re.compile(r"^\s*USER\s+(?P<user>\S+)", re.I)


@dataclass(frozen=True)
class CopyInstruction:
    line: int
    verb: str
    sources: list[str]
    dest: str
    flags: list[str]
    raw: str


def parse_copy(text: str) -> list[CopyInstruction]:
    out: list[CopyInstruction] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = COPY_RE.match(line)
        if not m:
            continue
        try:
            tokens = shlex.split(m.group("args"), comments=False)
        except ValueError:
            tokens = m.group("args").split()
        flags = [t for t in tokens if t.startswith("--")]
        rest = [t for t in tokens if not t.startswith("--")]
        if len(rest) < 2:
            continue
        out.append(
            CopyInstruction(
                line=i,
                verb=m.group(1).upper(),
                sources=rest[:-1],
                dest=rest[-1],
                flags=flags,
                raw=stripped,
            )
        )
    return out


def _components(token: str) -> list[str]:
    cleaned = token.strip().strip('"').strip("'").rstrip("/")
    parts = [p for p in re.split(r"[\\/]+", cleaned) if p not in ("", ".")]
    return parts


def _is_whole_context(sources: list[str]) -> bool:
    return any(s.strip() in (".", "./", "*") for s in sources)


def _touched_dir(sources: list[str], wanted: set[str]) -> str | None:
    for s in sources:
        comps = _components(s)
        if comps and comps[0].lower() in wanted:
            return s
    return None


def build_context_dir(task: Task) -> Path:
    """The Docker build context root.

    Terminal-Bench/Harbor builds the agent image from ``environment/`` (the loader
    records it as the spec's ``build_context``); the demo format builds from the
    task root. Rules that reason about what a ``COPY`` can reach must look here,
    not at the task root, or they flag a ``COPY . /app`` for tests/ that are not
    part of the context at all.
    """
    rel = getattr(task.spec, "build_context", None)
    if isinstance(rel, str) and rel:
        candidate = task.path / rel
        if candidate.is_dir():
            return candidate
    return task.path


def _context_has_dir(task: Task, name: str) -> bool:
    """Whether ``name`` (e.g. ``tests``) sits inside the build context."""
    return (build_context_dir(task) / name).is_dir()


def _in_build_context(task: Task, rel: str) -> bool:
    """Whether a task-relative file path falls inside the build context."""
    context = build_context_dir(task)
    if context == task.path:
        return True
    try:
        (task.path / rel).relative_to(context)
    except ValueError:
        return False
    return True


def _dockerignore(task: Task) -> str:
    path = build_context_dir(task) / ".dockerignore"
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _ignored_by_dockerignore(task: Task, name: str) -> bool:
    text = _dockerignore(task)
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.strip("/") == name or entry.strip("/") == f"{name}/":
            return True
        if entry.strip("/") == f"**/{name}":
            return True
    return False


# --------------------------------------------------------------------------
# S001 / S002 / S010 -- what ends up inside the agent's image
# --------------------------------------------------------------------------
def check_s001(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    df = dockerfile_source(task)
    if df is not None:
        for inst in parse_copy(df.text):
            hit = _touched_dir(inst.sources, HIDDEN_NAMES)
            if hit:
                findings.append(
                    make_finding(
                        "S001",
                        df.rel,
                        inst.line,
                        f"`{inst.verb} {hit}` copies the hidden verifier into the "
                        f"agent-visible image (destination `{inst.dest}`).",
                    )
                )
            elif _is_whole_context(inst.sources) and _context_has_dir(task, "tests"):
                if not _ignored_by_dockerignore(task, "tests"):
                    findings.append(
                        make_finding(
                            "S001",
                            df.rel,
                            inst.line,
                            f"`{inst.verb} . {inst.dest}` copies the whole build context, "
                            "which includes tests/; .dockerignore does not exclude it.",
                        )
                    )
        for line, raw in _line_matches(df.text, RUN_RE):
            parts = raw.split(None, 1)
            tokens = _safe_split(parts[1]) if len(parts) > 1 else []
            if not tokens:
                continue
            joined = " ".join(tokens)
            if re.search(r"\b(?:install|cp|mv|ln)\b[^|;]*\btests?\b", joined):
                findings.append(
                    make_finding(
                        "S001",
                        df.rel,
                        line,
                        "RUN step materialises a tests/ path inside the agent image.",
                    )
                )

    compose = compose_source(task)
    if compose is not None:
        for line, raw in _line_matches(compose.text, re.compile(r"tests?/", re.I)):
            findings.append(
                make_finding(
                    "S001",
                    compose.rel,
                    line,
                    "compose file references tests/ inside the agent container.",
                )
            )
    return findings


def check_s002(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    df = dockerfile_source(task)
    df_text = df.text if df else ""
    if df is not None:
        for inst in parse_copy(df.text):
            hit = _touched_dir(inst.sources, SOLUTION_NAMES)
            named = next((s for s in inst.sources if ANSWER_FILE_RE.match(Path(s).name)), None)
            dest = inst.dest.rstrip("/")
            if (hit or named) and dest and dest in task.instruction_md:
                # The instruction sends the agent to this path on purpose
                # (e.g. visible dev/reference cases), so it is not a leak.
                continue
            if hit:
                findings.append(
                    make_finding(
                        "S002",
                        df.rel,
                        inst.line,
                        f"`{inst.verb} {hit}` bakes the reference solution into the image.",
                    )
                )
            elif named:
                findings.append(
                    make_finding(
                        "S002",
                        df.rel,
                        inst.line,
                        f"`{inst.verb} {named}` bakes an answer-like file into the image.",
                    )
                )
            elif _is_whole_context(inst.sources) and _context_has_dir(task, "solution"):
                if not _ignored_by_dockerignore(task, "solution"):
                    findings.append(
                        make_finding(
                            "S002",
                            df.rel,
                            inst.line,
                            f"`{inst.verb} . {inst.dest}` copies the whole build context, "
                            "which includes solution/; .dockerignore does not exclude it.",
                        )
                    )
            elif _is_whole_context(inst.sources):
                for rel in task.rel_files:
                    if "/" in rel or not _in_build_context(task, rel):
                        continue
                    if ANSWER_FILE_RE.match(rel) and (task.path / rel).is_file():
                        findings.append(
                            make_finding(
                                "S002",
                                df.rel,
                                inst.line,
                                f"answer-like file `{rel}` is in the build context copied by `{inst.raw}`.",
                            )
                        )

    # Answer files sitting in the build context, even outside a Dockerfile COPY.
    for rel in task.rel_files:
        if not _in_build_context(task, rel):
            continue
        parts = rel.split("/")
        if parts[0] in ("tests", "solution"):
            continue
        if PROBE_ANSWER_RE.match(Path(rel).name):
            if not _ignored_by_dockerignore(task, Path(rel).name):
                findings.append(
                    make_finding(
                        "S002",
                        rel,
                        None,
                        f"answer-like file `{rel}` is in the build context and is not "
                        "excluded by .dockerignore.",
                    )
                )
    return findings


def check_s010(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    df = dockerfile_source(task)
    if df is None:
        return findings
    for inst in parse_copy(df.text):
        chmod = next((f for f in inst.flags if f.startswith("--chmod=")), None)
        if not chmod:
            continue
        touches_tests = _touched_dir(inst.sources, HIDDEN_NAMES) or (
            _is_whole_context(inst.sources) and _context_has_dir(task, "tests")
        )
        if touches_tests and _mode_is_writable(chmod.split("=", 1)[1]):
            findings.append(
                make_finding(
                    "S010",
                    df.rel,
                    inst.line,
                    f"tests are copied with `{chmod}`, i.e. writable by the agent user.",
                )
            )
    chmod_re = re.compile(r"\bchmod\b[^;&|]*\btests?\b", re.I)
    for line, raw in _line_matches(df.text, RUN_RE):
        if not chmod_re.search(raw):
            continue
        mode = re.search(r"\bchmod\s+(?:-R\s+)?([0-7]{3,4}|[ugoa]*\+[rwxX]+)", raw, re.I)
        if mode and _mode_is_writable(mode.group(1)):
            findings.append(
                make_finding(
                    "S010",
                    df.rel,
                    line,
                    f"tests are made writable with `{mode.group(0).strip()}`.",
                )
            )
    return findings


def _mode_is_writable(mode: str) -> bool:
    mode = mode.strip()
    if "+w" in mode.replace(" ", ""):
        return True
    digits = re.fullmatch(r"[0-7]{3,4}", mode)
    if not digits:
        return False
    value = mode[-3:]
    return any(d in "2367" for d in value)


# --------------------------------------------------------------------------
# S003 / S004 -- verifier quality (Python AST, shell fallback)
# --------------------------------------------------------------------------
def _parse_python(src: SourceFile) -> ast.Module | None:
    if not src.rel.endswith(".py") and not src.rel.endswith(".pyw"):
        return None
    try:
        return ast.parse(src.text, filename=src.rel)
    except SyntaxError:
        return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _usable_literal(value: object) -> bool:
    if isinstance(value, str):
        return len(value.strip()) >= 4 and not value.strip().lower() in {"none", "true", "false"}
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return abs(value) >= 10
    return False


CODE_SUFFIXES = (".py", ".sh", ".bash", ".js", ".mjs", ".ts", ".rb", ".go")
LOCAL_ADDR_RE = re.compile(r"127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|AF_UNIX|unix://", re.I)
INSTALL_LABELS = ("pip install", "npm install", "package manager")


def _is_code_file(name: str) -> bool:
    return name.lower().endswith(CODE_SUFFIXES)


def check_s003(task: Task) -> list[Finding]:
    instruction = task.instruction_md
    if not instruction.strip():
        return []
    folded = instruction.lower()
    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()

    for src in test_sources(task):
        if not _is_code_file(src.rel):
            continue
        tree = _parse_python(src)
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
                    continue
                operands: list[ast.AST] = [node.test.left, *node.test.comparators]
                for op, comp in zip(node.test.ops, node.test.comparators):
                    if not isinstance(op, (ast.Eq, ast.In)):
                        continue
                    candidates = [comp] + ([node.test.left] if isinstance(op, ast.In) else [])
                    for cand in candidates:
                        if not isinstance(cand, ast.Constant):
                            continue
                        if not _usable_literal(cand.value):
                            continue
                        text = str(cand.value)
                        if text in instruction or text.lower() in folded:
                            key = (src.rel, node.lineno)
                            if key in seen:
                                continue
                            seen.add(key)
                            findings.append(
                                make_finding(
                                    "S003",
                                    src.rel,
                                    node.lineno,
                                    f"asserts on literal {text!r}, which also appears in "
                                    "instruction.md (answer leakage, or a specified output "
                                    "contract: human check).",
                                    severity=Severity.INFO,
                                )
                            )
        else:
            for line, raw in _line_matches(src.text, re.compile(r'==\s*["\'][^"\']{5,}["\']')):
                literal = re.search(r'["\']([^"\']{4,})["\']', raw)
                if not literal:
                    continue
                text = literal.group(1)
                if (text in instruction or text.lower() in folded) and (src.rel, line) not in seen:
                    seen.add((src.rel, line))
                    findings.append(
                        make_finding(
                            "S003",
                            src.rel,
                            line,
                            f"compares against literal {text!r}, which also appears in "
                            "instruction.md (answer leakage, or a specified output "
                            "contract: human check).",
                            severity=Severity.INFO,
                        )
                    )
    return findings


def _is_weak_assert(node: ast.Assert) -> bool:
    test = node.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return True
    if isinstance(test, ast.Call):
        return _call_name(test.func) in WEAK_ASSERT_CALLS or _call_name(test.func).endswith(".exists")
    if isinstance(test, (ast.Name, ast.Attribute, ast.Subscript)):
        return True
    if isinstance(test, ast.Constant):
        return True
    if isinstance(test, ast.Compare):
        for op, comp in zip(test.ops, test.comparators):
            if isinstance(op, ast.IsNot):
                return True
            if isinstance(op, ast.NotEq) and isinstance(comp, ast.Constant) and comp.value in (0, "", None):
                return True
            if isinstance(op, ast.Gt) and isinstance(comp, ast.Constant) and comp.value == 0:
                if isinstance(test.left, ast.Call) and _call_name(test.left.func) in ("len", "count", "size"):
                    return True
        left = test.left
        if isinstance(left, ast.Call) and _call_name(left.func) in ("len", "count", "size"):
            for op, comp in zip(test.ops, test.comparators):
                if isinstance(comp, ast.Constant) and comp.value == 0 and isinstance(op, (ast.Gt, ast.GtE, ast.NotEq)):
                    return True
    return False


TEST_FUNC_RE = re.compile(r"^(test|check|verify|criterion)", re.I)


def _asserts_in_test_functions(tree: ast.Module) -> list[ast.Assert]:
    """Assertions that sit inside test/check functions (or decorated criteria).

    Harness and reporting code (``assert rows, "no results"``) is sanity-checking
    the verifier itself, not verifying the agent's work, so it is out of scope.
    """
    out: list[ast.Assert] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if TEST_FUNC_RE.match(fn.name) or fn.decorator_list:
            out.extend(n for n in ast.walk(fn) if isinstance(n, ast.Assert))
    return out


def check_s004(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    for src in test_sources(task):
        tree = _parse_python(src)
        if tree is not None:
            asserts = _asserts_in_test_functions(tree)
            if asserts and all(_is_weak_assert(a) for a in asserts):
                findings.append(
                    make_finding(
                        "S004",
                        src.rel,
                        min(a.lineno for a in asserts),
                        f"all {len(asserts)} assertion(s) are existence/non-emptiness checks; "
                        "nothing verifies behaviour.",
                    )
                )
        elif src.rel.endswith((".sh", ".bash")) or src.rel.endswith(".sh"):
            checks = [
                raw
                for _, raw in _line_matches(src.text, re.compile(r"\btest\s+-[fse]\b|\[\s+-[fse]\s|\[\[\s+-[fse]\s"))
            ]
            if checks and not re.search(r"diff|grep|cmp|==|python|pytest|assert", src.text):
                findings.append(
                    make_finding(
                        "S004",
                        src.rel,
                        _line_of(src.text, checks[0]) or 1,
                        f"{len(checks)} file-existence check(s) and no behavioural assertion.",
                    )
                )
    return findings


# --------------------------------------------------------------------------
# S005 -- unpinned dependencies
# --------------------------------------------------------------------------
PIP_IGNORE = {".", "-e", "-r", "-c", "--editable", "install"}

# A package manager invocation, and the shell separators that end its argument list.
APT_INSTALL_RE = re.compile(r"\bapt(?:-get)?\s+install\b|\bapk\s+add\b", re.I)
PIP_INSTALL_RE = re.compile(r"\b(?:pip3?|uv)\s+(?:pip\s+)?install\b", re.I)
NPM_INSTALL_RE = re.compile(r"\bnpm\s+(?:install|i|add)\b", re.I)
_ARG_SEPARATOR_RE = re.compile(r"&&|;|\|\|")
#: Tokens that are never a package name: shell operators and globs/expansion.
_NONPACKAGE_TOKENS = {"\\", "&&", "||", ";"}
_NONPACKAGE_CHARS = set("*/\\<>|&$;")


def _unpinned_pip(tokens: list[str]) -> list[str]:
    bad: list[str] = []
    for token in tokens:
        if token.startswith("-") or token in PIP_IGNORE:
            continue
        if token.endswith(".txt") or token.endswith(".whl") or token.endswith(".tar.gz"):
            continue
        if token.startswith(("http://", "https://", "git+")) or "/" in token:
            continue
        if "==" in token or "===" in token:
            continue
        bad.append(token.split(";")[0])
    return bad


def _apt_install_packages(logical: str) -> list[str]:
    """Unpinned packages named after ``apt-get install`` / ``apk add``.

    Only the segment up to the next ``&&`` / ``;`` / ``||`` is considered, so the
    trailing ``rm -rf /var/lib/apt/lists/*`` (and any later chained command) is not
    mistaken for a package. Flags (``-y``), shell operators and any token holding a
    glob/redirection/expansion character (``* / \\ < > | & $ ;``) are skipped.
    """
    bad: list[str] = []
    for match in APT_INSTALL_RE.finditer(logical):
        segment = _ARG_SEPARATOR_RE.split(logical[match.end():])[0]
        for token in segment.split():
            if token.startswith("-") or token in _NONPACKAGE_TOKENS:
                continue
            if any(ch in _NONPACKAGE_CHARS for ch in token):
                continue
            if "=" in token:  # pinned as package=version
                continue
            bad.append(token)
    return bad


def check_s005(task: Task) -> list[Finding]:
    findings: list[Finding] = []

    df = dockerfile_source(task)
    if df is not None:
        for line, logical in _logical_lines(df.text):
            stripped = logical.strip()
            if not stripped or stripped.startswith("#"):
                continue
            low = stripped.lower()
            if PIP_INSTALL_RE.search(low):
                tokens = _safe_split(stripped)
                idx = next((i for i, t in enumerate(tokens) if t.startswith("install")), 0)
                bad = _unpinned_pip(tokens[idx + 1:])
                if bad:
                    findings.append(
                        make_finding(
                            "S005",
                            df.rel,
                            line,
                            f"unpinned pip requirement(s): {', '.join(bad)}",
                        )
                    )
            if APT_INSTALL_RE.search(stripped):
                bad = _apt_install_packages(stripped)
                if bad:
                    # Advisory: an unpinned system package is a reproducibility
                    # smell, not a correctness blocker, so it is an INFO.
                    findings.append(
                        make_finding(
                            "S005",
                            df.rel,
                            line,
                            f"unpinned system package(s): {', '.join(bad)}",
                            severity=Severity.INFO,
                        )
                    )
            if NPM_INSTALL_RE.search(low):
                tokens = _safe_split(stripped)
                idx = next((i for i, t in enumerate(tokens) if t in ("install", "i", "add")), 0)
                bad = [t for t in tokens[idx + 1:] if not t.startswith("-") and "@" not in t]
                if bad:
                    findings.append(
                        make_finding("S005", df.rel, line, f"unpinned npm package(s): {', '.join(bad)}")
                    )
        for line, raw in _line_matches(df.text, FROM_RE):
            m = FROM_RE.match(raw)
            if not m:
                continue
            image = m.group("image")
            if image.lower().endswith(":latest") or ":" not in image.split("/")[-1]:
                findings.append(
                    make_finding(
                        "S005",
                        df.rel,
                        line,
                        f"base image `{image}` is not pinned to an explicit tag.",
                    )
                )

    for src in iter_sources(task):
        if Path(src.rel).name != "requirements.txt":
            continue
        for line, raw in enumerate(src.text.splitlines(), start=1):
            entry = raw.split("#", 1)[0].strip()
            if not entry or entry.startswith(("-", "http", "git+")):
                continue
            if "==" not in entry and "===" not in entry:
                findings.append(
                    make_finding(
                        "S005",
                        src.rel,
                        line,
                        f"requirement `{entry}` is not pinned with ==.",
                    )
                )
    return findings


# --------------------------------------------------------------------------
# S006 -- runtime network access
# --------------------------------------------------------------------------
def _is_dependency_manifest(name: str) -> bool:
    """Lock files pin transitive dependencies; they are not runtime code.

    Scanning them for network calls produces thousands of false positives (a
    ``uv.lock``/``package-lock.json`` is full of URLs by construction).
    """
    lowered = name.lower()
    return (
        lowered.endswith(".lock")
        or lowered == "package-lock.json"
        or lowered.endswith("-lock.json")
    )


EXTERNAL_URL_RE = re.compile(r"https?://(?!127\\.|localhost|0\\.0\\.0\\.0|\\[::1\\])[A-Za-z0-9]", re.I)


def _s006_severity(raw: str, label: str, base: Severity | None) -> Severity | None:
    """BLOCKER only when the line visibly reaches an external host.

    Installs of agent-supplied dependencies at grading time, and calls whose
    target is a variable (usually the agent's own local service), are real
    risks but cannot be proven external statically, so they are WARN.
    """
    if any(k in label for k in INSTALL_LABELS):
        return Severity.WARN
    if EXTERNAL_URL_RE.search(raw):
        return base
    return Severity.WARN


def check_s006(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    # Dockerfiles are build-time: curl/apt-get/uv in them fetch at image build,
    # not at verification time. Only the runtime verifier and solution files are
    # scanned (test.sh, checks.py, solve.sh, ...), where a network call is real.
    # Dependency manifests are skipped: they are data, not code.
    verifier = [
        src for src in test_sources(task) if not _is_dependency_manifest(Path(src.rel).name)
    ]
    solution = [
        src for src in solution_sources(task) if not _is_dependency_manifest(Path(src.rel).name)
    ]
    # A network call in the hidden verifier is a BLOCKER (the rule default); the
    # solution may legitimately reach the network on the author's machine, so its
    # findings are downgraded to WARN.
    targets: list[tuple[SourceFile, Severity | None]] = [(s, None) for s in verifier]
    targets += [(s, Severity.WARN) for s in solution]
    for src, severity in targets:
        if Path(src.rel).name.lower() == "dockerfile" or not _is_code_file(src.rel):
            continue
        seen_lines: set[int] = set()
        for i, raw in enumerate(src.text.splitlines(), start=1):
            stripped = raw.strip()
            if stripped.startswith(("#", "import ", "from ")) or LOCAL_ADDR_RE.search(raw):
                continue
            for pattern, label in NETWORK_PATTERNS:
                if re.search(pattern, raw):
                    if i in seen_lines:
                        continue
                    seen_lines.add(i)
                    findings.append(
                        make_finding(
                            "S006",
                            src.rel,
                            i,
                            f"runtime network access in the verifier/solution: {label}.",
                            severity=_s006_severity(raw, label, severity),
                        )
                    )
                    break
    return findings


# --------------------------------------------------------------------------
# S007 -- non-determinism in tests
# --------------------------------------------------------------------------
def check_s007(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    for src in test_sources(task):
        seen_lines: set[int] = set()
        random_lines: list[int] = []
        has_seed = bool(re.search(r"\brandom\.seed\s*\(", src.text))
        for i, raw in enumerate(src.text.splitlines(), start=1):
            if raw.strip().startswith("#"):
                continue
            for pattern, label in NONDET_PATTERNS:
                if re.search(pattern, raw):
                    if i in seen_lines:
                        continue
                    seen_lines.add(i)
                    if label.startswith("random"):
                        random_lines.append(i)
                        if has_seed:
                            continue
                        findings.append(
                            make_finding(
                                "S007",
                                src.rel,
                                i,
                                "uses random.* without random.seed(): the verifier is not reproducible.",
                            )
                        )
                        continue
                    findings.append(
                        make_finding("S007", src.rel, i, f"nondeterminism source: {label}.")
                    )
                    break
    return findings


# --------------------------------------------------------------------------
# S008 -- declared output format is strictly validated
# --------------------------------------------------------------------------
FORMAT_SIGNATURES: dict[str, tuple[str, ...]] = {
    "json": (r"\bjson\.loads?\s*\(", r"\bjson\.JSONDecodeError\b", r"\borjson\.loads?\s*\("),
    "yaml": (r"\byaml\.safe_load\s*\(", r"\byaml\.load\s*\("),
    "csv": (r"\bcsv\.(DictReader|reader)\s*\(", r"\bpd\.read_csv\s*\(", r"\bpandas\.read_csv\s*\("),
}


def check_s008(task: Task) -> list[Finding]:
    fmt = task.spec.expected_output_format
    if fmt in (None, "text"):
        return []
    signatures = FORMAT_SIGNATURES.get(fmt, ())
    tests = test_sources(task)
    if not tests:
        return [
            make_finding(
                "S008",
                "task.yaml",
                None,
                f"task declares output format '{fmt}' but there is no hidden verifier to check it.",
            )
        ]
    combined = "\n".join(src.text for src in tests)
    if not any(re.search(sig, combined) for sig in signatures):
        first = tests[0]
        return [
            make_finding(
                "S008",
                first.rel,
                None,
                f"task declares output format '{fmt}' but no test parses it as {fmt}.",
            )
        ]
    if not re.search(r"\bassert\b|isinstance\s*\(|pytest\.raises", combined):
        first = tests[0]
        return [
            make_finding(
                "S008",
                first.rel,
                None,
                f"tests parse {fmt} but never assert on the parsed structure.",
            )
        ]
    return []


# --------------------------------------------------------------------------
# S009 -- instruction ambiguity (advisory)
# --------------------------------------------------------------------------
OUTPUT_PATH_RE = re.compile(
    r"(?:[\w.@-]*/)?[\w.-]+\.(?:json|jsonl|csv|txt|yaml|yml|tsv|log|md|xml|out|bin)\b", re.I
)
FORMAT_WORD_RE = re.compile(
    r"\b(json|jsonl|csv|yaml|yml|tsv|xml|plain text|text format|format|schema|stdout|JSON object)\b",
    re.I,
)
PRONOUN_RE = re.compile(r"\b(it|this|that|the file|the output|the result|the artefact|the artifact)\b", re.I)


def check_s009(task: Task) -> list[Finding]:
    instruction = task.instruction_md
    if not instruction.strip():
        return [
            make_finding("S009", "instruction.md", None, "instruction is empty; there is no spec to solve.")
        ]
    findings: list[Finding] = []
    path_match = OUTPUT_PATH_RE.search(instruction)
    format_match = FORMAT_WORD_RE.search(instruction)
    if not path_match:
        findings.append(
            make_finding(
                "S009",
                "instruction.md",
                _line_of(instruction, "output") or 1,
                "instruction never names the file (or path) the agent must produce.",
            )
        )
    if not format_match:
        findings.append(
            make_finding(
                "S009",
                "instruction.md",
                1,
                "instruction never states the expected output format.",
            )
        )
    if not path_match and PRONOUN_RE.search(instruction):
        m = PRONOUN_RE.search(instruction)
        findings.append(
            make_finding(
                "S009",
                "instruction.md",
                _line_of(instruction, m.group(0)) if m else 1,
                f"instruction uses an unresolved reference ({m.group(0)!r}) and never names the artefact.",
            )
        )
    return findings


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
ALL_CHECKS = (
    check_s001,
    check_s002,
    check_s003,
    check_s004,
    check_s005,
    check_s006,
    check_s007,
    check_s008,
    check_s009,
    check_s010,
)


def run_static_audit(task: Task, checks=ALL_CHECKS) -> list[Finding]:
    findings: list[Finding] = []
    for check in checks:
        try:
            findings.extend(check(task))
        except Exception as exc:  # a broken rule must not abort the audit
            findings.append(
                Finding(
                    rule_id="AUDIT-ERROR",
                    severity=Severity.WARN,
                    file=str(task.path),
                    message=f"{check.__name__} crashed: {exc.__class__.__name__}: {exc}",
                    fix_hint="Report this as a TaskGate bug.",
                )
            )
    deduped: dict[tuple[str, str, int | None, str], Finding] = {}
    for f in findings:
        deduped.setdefault((f.rule_id, f.file, f.line, f.message), f)
    return sorted(
        deduped.values(),
        key=lambda f: ({"BLOCKER": 0, "WARN": 1, "INFO": 2}[f.severity.value], f.file, f.line or 0, f.rule_id),
    )

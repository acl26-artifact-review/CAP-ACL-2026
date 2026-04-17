#!/usr/bin/env python3
"""
Export prompt templates that are embedded as module-level constants in `code/`.

This is intended for reviewer convenience: prompts are easier to browse as
individual text files. The code remains the source of truth.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


PROMPT_KEYS = ("PROMPT", "GUIDANCE", "INSTRUCTION", "INSTRUCTIONS", "RUBRIC")


def _is_prompt_name(name: str) -> bool:
    return name.isupper() and any(k in name for k in PROMPT_KEYS)


def _safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def _joinedstr_to_template(node: ast.JoinedStr) -> str:
    parts: List[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            # Keep a readable placeholder without evaluating code.
            try:
                expr = ast.unparse(value.value)  # py>=3.9
            except Exception:
                expr = "expr"
            parts.append("{" + expr + "}")
        else:
            parts.append("{...}")
    return "".join(parts)


EvalResult = Union[str, Dict[str, str]]


def _eval_str_expr(node: ast.AST) -> Optional[EvalResult]:
    # String literal
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    # r'''...''' still becomes a normal Constant(str) in the AST; handled above.

    # "a" + "b"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_str_expr(node.left)
        right = _eval_str_expr(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return None

    # f"...{x}..."
    if isinstance(node, ast.JoinedStr):
        return _joinedstr_to_template(node)

    # {"soap": "...", "sectioned": "..."} etc.
    if isinstance(node, ast.Dict):
        out: Dict[str, str] = {}
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None
            vv = _eval_str_expr(v)
            if not isinstance(vv, str):
                return None
            out[k.value] = vv
        return out

    # textwrap.dedent("..."), "...".strip(), etc.
    if isinstance(node, ast.Call):
        # Handle attr calls on a string constant: """...""".strip()
        if isinstance(node.func, ast.Attribute):
            base = _eval_str_expr(node.func.value)
            if not isinstance(base, str):
                return None
            if node.args or node.keywords:
                return None
            method = node.func.attr
            if method == "strip":
                return base.strip()
            if method == "lstrip":
                return base.lstrip()
            if method == "rstrip":
                return base.rstrip()
            return None

        # Handle calls like textwrap.dedent("...") or dedent("...")
        if isinstance(node.func, ast.Name) and node.func.id in {"dedent"}:
            if len(node.args) != 1 or node.keywords:
                return None
            inner = _eval_str_expr(node.args[0])
            if isinstance(inner, str):
                return textwrap.dedent(inner)
            return None

        if isinstance(node.func, ast.Attribute) and node.func.attr == "dedent":
            if len(node.args) != 1 or node.keywords:
                return None
            inner = _eval_str_expr(node.args[0])
            if isinstance(inner, str):
                return textwrap.dedent(inner)
            return None

    return None


@dataclass(frozen=True)
class PromptItem:
    src_file: str
    src_line: int
    name: str
    out_files: List[str]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_prompts(code_dir: Path, out_dir: Path) -> List[PromptItem]:
    items: List[PromptItem] = []
    for py in sorted(code_dir.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        mod = ast.parse(src)
        for node in mod.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue

            targets: List[str] = []
            value: Optional[ast.AST] = None
            if isinstance(node, ast.Assign):
                value = node.value
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t.id)
            else:
                value = node.value
                if isinstance(node.target, ast.Name):
                    targets.append(node.target.id)

            if value is None:
                continue

            for name in targets:
                if not _is_prompt_name(name):
                    continue
                evaluated = _eval_str_expr(value)
                if evaluated is None:
                    continue

                module_dir = out_dir / py.stem
                out_files: List[str] = []
                if isinstance(evaluated, str):
                    out_path = module_dir / f"{name}.txt"
                    _write_text(out_path, evaluated)
                    out_files.append(str(out_path.relative_to(out_dir)))
                else:
                    # Dict[str,str]: export one file per key + a json for convenience.
                    json_path = module_dir / f"{name}.json"
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    json_path.write_text(json.dumps(evaluated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    out_files.append(str(json_path.relative_to(out_dir)))
                    for k, v in evaluated.items():
                        key_slug = _safe_slug(k)
                        out_path = module_dir / f"{name}.{key_slug}.txt"
                        _write_text(out_path, v)
                        out_files.append(str(out_path.relative_to(out_dir)))

                items.append(
                    PromptItem(
                        src_file=f"code/{py.name}",
                        src_line=getattr(node, "lineno", 1),
                        name=name,
                        out_files=sorted(out_files),
                    )
                )
    return items


def write_index(out_dir: Path, items: List[PromptItem]) -> None:
    lines: List[str] = []
    lines.append("# Prompt Index")
    lines.append("")
    lines.append("These files are exported snapshots of prompt templates embedded in `code/`.")
    lines.append("The code remains the source of truth; run `tools/export_prompts.py` to regenerate.")
    lines.append("")
    for item in sorted(items, key=lambda x: (x.src_file, x.src_line, x.name)):
        lines.append(f"- `{item.src_file}:{item.src_line}` `{item.name}`")
        for out_file in item.out_files:
            lines.append(f"  - `prompts/{out_file}`")
    (out_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", type=Path, default=Path("code"))
    ap.add_argument("--out-dir", type=Path, default=Path("prompts"))
    args = ap.parse_args()

    code_dir = args.code_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items = export_prompts(code_dir=code_dir, out_dir=out_dir)
    write_index(out_dir=out_dir, items=items)
    print(f"exported_items={len(items)} out_dir={out_dir}")


if __name__ == "__main__":
    main()


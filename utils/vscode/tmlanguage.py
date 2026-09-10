#!/usr/bin/env python3

"""A minimal TextMate grammar tokenizer.

TextMate grammars are normally run by Oniguruma, which we cannot depend on
hermetically. Carbon's grammar happens to use no Oniguruma-only regex syntax,
so `re` runs its patterns unchanged and this tokenizer reproduces VS Code's
output exactly. `check_regex_dialect` guards that property; if it fails, this
tokenizer can no longer be trusted and the grammar needs to move back into the
shared subset.

Only the grammar features Carbon uses are implemented: `match`, `begin`/`end`
with captures and `contentName`, `include` into the repository, and `\\N`
backreferences from `begin` into `end`.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import json
import re
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional

# Regex constructs Oniguruma supports and `re` either rejects or interprets
# differently. A grammar using any of these would invalidate this tokenizer.
_FOREIGN_SYNTAX = {
    "possessive quantifier": r"(?:[*+?}])\+",
    "atomic group": r"\(\?>",
    r"\G anchor": r"\\G",
    r"\h horizontal space": r"\\[hH]",
    "conditional": r"\(\?\(",
    "named backreference": r"\\k<",
    "variable-length lookbehind": r"\(\?<[=!][^)]*[*+]",
}


class Token(NamedTuple):
    """A scope assignment covering `[start, end)` of line `line`."""

    line: int
    start: int
    end: int
    scopes: list[str]


class Rule(NamedTuple):
    """A `begin`/`end` region that is currently open."""

    scopes: list[str]
    content_scopes: list[str]
    end_re: Optional[re.Pattern[str]]
    end_captures: dict[str, Any]
    patterns: list[dict[str, Any]]


class Grammar:
    """A parsed TextMate grammar, ready to tokenize with."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.scope_name: str = raw.get("scopeName", "")
        self._repository: dict[str, Any] = raw.get("repository", {})
        self._compiled: dict[str, re.Pattern[str]] = {}

    @staticmethod
    def load(path: Path) -> "Grammar":
        with path.open(encoding="utf-8") as f:
            return Grammar(json.load(f))

    def compile(self, pattern: str) -> re.Pattern[str]:
        compiled = self._compiled.get(pattern)
        if compiled is None:
            compiled = self._compiled[pattern] = re.compile(pattern)
        return compiled

    def resolve(
        self,
        patterns: Optional[list[dict[str, Any]]],
        active: frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]]:
        """Flattens `include` directives into a list of concrete rules.

        A `name` on the repository entry itself is deliberately dropped: VS
        Code inlines an include-only rule's patterns without pushing its scope,
        so honoring it here would disagree with the real tokenizer. `active`
        breaks a cycle between mutually including entries.

        An include of a different grammar, such as `source.cpp`, contributes
        nothing: we tokenize Carbon, and embedded regions are left to whatever
        scope their `contentName` assigns.
        """
        resolved: list[dict[str, Any]] = []
        for pattern in patterns or []:
            target = pattern.get("include")
            if target is None:
                resolved.append(pattern)
            elif target.startswith("#"):
                name = target[1:]
                if name in active:
                    continue
                entry = self._repository.get(name, {})
                resolved.extend(
                    self.resolve(entry.get("patterns"), active | {name})
                )
            elif target == "$self":
                resolved.extend(self.resolve(self.raw.get("patterns"), active))
        return resolved

    def initial_stack(self) -> list[Rule]:
        base = [self.scope_name]
        return [
            Rule(base, base, None, {}, self.resolve(self.raw.get("patterns")))
        ]

    def all_patterns(self) -> Iterator[tuple[str, str]]:
        """Yields every `(key, regex)` in the grammar, for validation."""

        def walk(node: Any) -> Iterator[tuple[str, str]]:
            if isinstance(node, dict):
                for key in ("match", "begin", "end"):
                    value = node.get(key)
                    if isinstance(value, str):
                        yield key, value
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)

        yield from walk(self.raw)


def check_regex_dialect(grammar: Grammar) -> list[str]:
    """Returns a message per pattern `re` cannot stand in for Oniguruma on."""
    problems: list[str] = []
    for key, pattern in grammar.all_patterns():
        # A `\N` backreference is substituted before compiling, so a bare one
        # is not a valid pattern on its own.
        probe = re.sub(r"\\\d", "x", pattern)
        try:
            re.compile(probe)
        except re.error as e:
            problems.append(f"`{key}` pattern does not compile: {pattern}: {e}")
            continue
        for name, foreign in _FOREIGN_SYNTAX.items():
            if re.search(foreign, pattern):
                problems.append(
                    f"`{key}` pattern uses {name}, which `re` does not share "
                    f"with Oniguruma: {pattern}"
                )
    return problems


def _substitute_backrefs(pattern: str, match: re.Match[str]) -> str:
    """Replaces `\\N` in an `end` with the text `begin` captured, as VS Code
    does before compiling the `end` pattern."""

    def replace(ref: re.Match[str]) -> str:
        return re.escape(match.group(int(ref.group(1))) or "")

    return re.sub(r"\\(\d)", replace, pattern)


def _emit(
    tokens: list[Token], line: int, start: int, end: int, scopes: list[str]
) -> None:
    if end > start:
        tokens.append(Token(line, start, end, list(scopes)))


def _emit_captures(
    tokens: list[Token],
    line: int,
    match: re.Match[str],
    captures: dict[str, Any],
    scopes: list[str],
) -> None:
    """Scopes each captured group, and the gaps between them."""
    if not captures:
        _emit(tokens, line, match.start(), match.end(), scopes)
        return
    pos = match.start()
    for group in range((match.re.groups or 0) + 1):
        spec = captures.get(str(group))
        if spec is None or match.start(group) < pos:
            continue
        if match.start(group) == match.end(group):
            continue
        _emit(tokens, line, pos, match.start(group), scopes)
        _emit(
            tokens,
            line,
            match.start(group),
            match.end(group),
            scopes + [spec["name"]],
        )
        pos = match.end(group)
    _emit(tokens, line, pos, match.end(), scopes)


def tokenize_line(
    grammar: Grammar, line: int, text: str, stack: list[Rule]
) -> tuple[list[Token], list[Rule]]:
    """Tokenizes one line. `text` must carry its trailing newline."""
    tokens: list[Token] = []
    pos = 0
    steps = 0
    while True:
        rule = stack[-1]
        # The open region's `end` is tried first, so it wins ties on offset.
        best: Optional[tuple[int, str, dict[str, Any], re.Match[str]]] = None
        if rule.end_re is not None:
            match = rule.end_re.search(text, pos)
            if match:
                best = (match.start(), "end", {}, match)
        for candidate in rule.patterns:
            pattern = candidate.get("match") or candidate.get("begin")
            if pattern is None:
                continue
            match = grammar.compile(pattern).search(text, pos)
            if match is None or (best is not None and match.start() >= best[0]):
                continue
            kind = "match" if "match" in candidate else "begin"
            best = (match.start(), kind, candidate, match)
        if best is None:
            _emit(tokens, line, pos, len(text), rule.content_scopes)
            return tokens, stack
        start, kind, candidate, match = best
        _emit(tokens, line, pos, start, rule.content_scopes)
        if kind == "end":
            _emit_captures(tokens, line, match, rule.end_captures, rule.scopes)
            stack = stack[:-1]
        else:
            scopes = rule.content_scopes + (
                [candidate["name"]] if "name" in candidate else []
            )
            if kind == "match":
                _emit_captures(
                    tokens, line, match, candidate.get("captures", {}), scopes
                )
            else:
                _emit_captures(
                    tokens,
                    line,
                    match,
                    candidate.get("beginCaptures", {}),
                    scopes,
                )
                content = scopes + (
                    [candidate["contentName"]]
                    if "contentName" in candidate
                    else []
                )
                stack = stack + [
                    Rule(
                        scopes,
                        content,
                        grammar.compile(
                            _substitute_backrefs(candidate["end"], match)
                        ),
                        candidate.get("endCaptures", {}),
                        grammar.resolve(candidate.get("patterns")),
                    )
                ]
        # A zero-width match must still make progress. Entering or leaving a
        # region is progress by itself, and the character stays available to
        # the enclosing rule, which is what a lookahead `end` such as
        # `(?=[\\[\\(])` relies on. Only a `match` that consumed nothing has to
        # step over a character to avoid spinning.
        if match.end() > start:
            pos = match.end()
        elif kind == "match":
            pos = start + 1
        else:
            pos = start
            steps += 1
            # Guards against a region that opens and closes without consuming.
            if steps > len(text) + 64:
                _emit(tokens, line, pos, len(text), stack[-1].content_scopes)
                return tokens, stack
        if pos > len(text):
            return tokens, stack


def tokenize(grammar: Grammar, text: str) -> Iterator[Token]:
    """Tokenizes a whole file."""
    stack = grammar.initial_stack()
    for number, line in enumerate(text.split("\n")):
        tokens, stack = tokenize_line(grammar, number, line + "\n", stack)
        yield from tokens


def scope_stack_ends_open(grammar: Grammar, text: str) -> bool:
    """Whether a `begin`/`end` region was still open at end of file.

    A region that never closes recolors everything after it, so this is the
    single most damaging thing a grammar can do.
    """
    stack = grammar.initial_stack()
    for number, line in enumerate(text.split("\n")):
        _, stack = tokenize_line(grammar, number, line + "\n", stack)
    return len(stack) > 1

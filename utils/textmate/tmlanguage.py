#!/usr/bin/env python3

"""A minimal TextMate grammar tokenizer.

TextMate grammars are normally run by Oniguruma, which we cannot depend on
hermetically. Carbon's grammar happens to use no Oniguruma-only regex syntax,
so `re` compiles its regexes unchanged. That is necessary but not sufficient,
since the two engines can still read shared syntax differently, so what
establishes that this reproduces VS Code's output is `conformance_test.py`,
which tokenizes the repository with both and compares. `check_regex_dialect`
catches the half of that a hermetic test can reach: if it fails, the grammar
has left the shared subset and this tokenizer can no longer be trusted.

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


# Every key a TextMate grammar can put a regex under. `match`, `begin`, and
# `end` are the three this tokenizer runs; an editor also runs the folding
# markers, so they are checked for dialect even though nothing here reads them.
_REGEX_KEYS = (
    "match",
    "begin",
    "end",
    "foldingStartMarker",
    "foldingStopMarker",
)


class Token(NamedTuple):
    """A scope assignment covering `[start, end)` of line `line`.

    `scopes` runs outermost first, starting with the grammar's own scope name,
    so the last entry is the most specific. A theme matches from there
    outward, taking the first entry it has a color for.
    """

    line: int
    start: int
    end: int
    scopes: list[str]


class Rule(NamedTuple):
    """An entry of the region stack: an open `begin`/`end`, or the grammar.

    The grammar's own entry sits at the bottom with an `end_re` of `None`,
    since nothing closes it.
    """

    scopes: list[str]
    content_scopes: list[str]
    end_re: Optional[re.Pattern[str]]
    end_captures: dict[str, Any]
    patterns: list[dict[str, Any]]


class _RuleMatch(NamedTuple):
    """A rule that matched, and where it matched.

    `kind` names the grammar key the matching regex came from: `"end"` for
    the open region's own `end`, in which case `pattern` is empty, or
    `"match"` or `"begin"` for one of the rules in the region's `patterns`.
    `match` is always a real match; it is what `kind` describes.
    """

    kind: str
    pattern: dict[str, Any]
    match: re.Match[str]


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

    def compile(self, regex: str) -> re.Pattern[str]:
        compiled = self._compiled.get(regex)
        if compiled is None:
            compiled = self._compiled[regex] = re.compile(regex)
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

    def all_regexes(self) -> Iterator[tuple[str, str]]:
        """Yields every `(key, regex)` the grammar holds, for validation.

        A grammar keeps regexes only under the keys in `_REGEX_KEYS`, so
        finding them all means walking every dictionary in it and picking
        those keys out. `key` says which one it was, which is all a message
        needs to point at the right place.
        """

        def walk(node: Any) -> Iterator[tuple[str, str]]:
            if isinstance(node, dict):
                for key in _REGEX_KEYS:
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
    """Returns a message per regex `re` cannot stand in for Oniguruma on."""
    problems: list[str] = []
    for key, regex in grammar.all_regexes():
        # A `\N` backreference is substituted before compiling, so do the
        # same here with an arbitrary value before validating.
        probe = re.sub(r"\\\d", "x", regex)
        try:
            re.compile(probe)
        except re.error as e:
            problems.append(f"`{key}` regex does not compile: {regex}: {e}")
            continue
        for name, foreign in _FOREIGN_SYNTAX.items():
            if re.search(foreign, regex):
                problems.append(
                    f"`{key}` regex uses {name}, which `re` does not share "
                    f"with Oniguruma: {regex}"
                )
    return problems


def _substitute_backrefs(regex: str, begin_match: re.Match[str]) -> str:
    """Replaces `\\N` in an `end` with the text `begin` captured, as VS Code
    does before compiling the `end` regex."""

    def replace(ref: re.Match[str]) -> str:
        return re.escape(begin_match.group(int(ref.group(1))) or "")

    return re.sub(r"\\(\d)", replace, regex)


def _append_token(
    tokens: list[Token], linenum: int, start: int, end: int, scopes: list[str]
) -> None:
    """Appends a token, unless it would be empty.

    Callers append the text between two things without first checking that
    there is any, and about a quarter of the time there is none, so the check
    lives here rather than at every call.
    """
    if end > start:
        tokens.append(Token(linenum, start, end, scopes))


def _append_capture_tokens(
    tokens: list[Token],
    linenum: int,
    match: re.Match[str],
    captures: dict[str, Any],
    scopes: list[str],
) -> None:
    """Appends the tokens for one match, splitting it at its capture groups.

    `captures` maps a group number, written as a string, to the scope the
    group's text takes on top of `scopes`; group `"0"` is the whole match. The
    tokens tile the match with no gaps: text not covered by a listed group is
    still appended, under `scopes` alone.
    """
    if not captures:
        _append_token(tokens, linenum, match.start(), match.end(), scopes)
        return
    pos = match.start()
    for group in range((match.re.groups or 0) + 1):
        spec = captures.get(str(group))
        # Skip a group the grammar does not name, one that did not
        # participate in the match (`start` is then -1), and one that lies
        # behind text already appended, since tokens come out in order.
        if spec is None or match.start(group) < pos:
            continue
        if match.start(group) == match.end(group):
            continue
        _append_token(tokens, linenum, pos, match.start(group), scopes)
        _append_token(
            tokens,
            linenum,
            match.start(group),
            match.end(group),
            scopes + [spec["name"]],
        )
        pos = match.end(group)
    _append_token(tokens, linenum, pos, match.end(), scopes)


def _push_scope(
    scopes: list[str], pattern: dict[str, Any], key: str
) -> list[str]:
    """Adds the scope `pattern[key]` names, if it names one."""
    name = pattern.get(key)
    return scopes + [name] if name is not None else scopes


def _find_earliest(
    grammar: Grammar, rule: Rule, text: str, pos: int
) -> Optional[_RuleMatch]:
    """Finds which of `rule`'s regexes matches soonest at or after `pos`.

    The open region's own `end` is tried first and ties are broken towards
    whatever was tried earlier, which gives TextMate's two ordering rules: a
    region that can close here closes here, and otherwise the first rule
    listed in the grammar wins.
    """
    best: Optional[_RuleMatch] = None
    if rule.end_re is not None:
        match = rule.end_re.search(text, pos)
        if match is not None:
            best = _RuleMatch("end", {}, match)
    for pattern in rule.patterns:
        # The key a regex came from is what the rule does with it, so take
        # both from the same lookup.
        if (regex := pattern.get("match")) is not None:
            kind = "match"
        elif (regex := pattern.get("begin")) is not None:
            kind = "begin"
        else:
            continue
        match = grammar.compile(regex).search(text, pos)
        if match is None:
            continue
        if best is not None and match.start() >= best.match.start():
            continue
        best = _RuleMatch(kind, pattern, match)
    return best


def tokenize_line(
    grammar: Grammar, linenum: int, text: str, stack: list[Rule]
) -> tuple[list[Token], list[Rule]]:
    """Tokenizes one line, given the regions open when it starts.

    `text` must carry its trailing newline: VS Code tokenizes a line together
    with its terminator, so that newline gets a token of its own, and carrying
    it is what keeps the two token streams identical. `stack` is the chain of
    `begin`/`end` regions open at the start of the line, outermost first and
    never empty: its first entry is the grammar itself. The stack returned is
    the one open at the start of the next line, which is how a region spans
    lines.
    """
    tokens: list[Token] = []
    pos = 0
    # Counts region transitions that consumed nothing, which make no progress.
    stalls = 0
    while pos <= len(text):
        rule = stack[-1]
        found = _find_earliest(grammar, rule, text, pos)
        if found is None:
            break
        # Text before the match belongs to the region containing it.
        _append_token(
            tokens, linenum, pos, found.match.start(), rule.content_scopes
        )
        if found.kind == "end":
            _append_capture_tokens(
                tokens, linenum, found.match, rule.end_captures, rule.scopes
            )
            stack = stack[:-1]
        else:
            scopes = _push_scope(rule.content_scopes, found.pattern, "name")
            captures = "captures" if found.kind == "match" else "beginCaptures"
            _append_capture_tokens(
                tokens,
                linenum,
                found.match,
                found.pattern.get(captures, {}),
                scopes,
            )
            if found.kind == "begin":
                stack = stack + [
                    Rule(
                        scopes,
                        _push_scope(scopes, found.pattern, "contentName"),
                        grammar.compile(
                            _substitute_backrefs(
                                found.pattern["end"], found.match
                            )
                        ),
                        found.pattern.get("endCaptures", {}),
                        grammar.resolve(found.pattern.get("patterns")),
                    )
                ]

        if found.match.end() > found.match.start():
            pos = found.match.end()
        elif found.kind == "match":
            # A `match` that consumed nothing would match there forever, so
            # step over a character.
            pos = found.match.start() + 1
        else:
            # Entering or leaving a region is progress in itself, and the
            # character stays available to the region now on top: a lookahead
            # `end` such as `(?=[\[\(])` depends on that.
            pos = found.match.start()
            stalls += 1
            # The bound is arbitrary; it only has to exceed the transitions a
            # real line could ask for. Reaching it means a rule opens and
            # closes forever, which is a grammar bug: say so rather than
            # silently truncate the line.
            if stalls > len(text) + 64:
                raise RuntimeError(
                    f"line {linenum} stopped making progress at offset {pos}: "
                    f"rule {found.pattern!r} neither consumes nor terminates"
                )
    # Whatever is left over, including the newline, belongs to the region the
    # line ends inside of.
    _append_token(tokens, linenum, pos, len(text), stack[-1].content_scopes)
    return tokens, stack


def split_lines(text: str) -> list[str]:
    """Splits a file into lines the way an editor numbers them.

    Terminators are not kept on the lines, and a file ending in a newline
    yields an empty final line, which is the blank line an editor shows there.

    Not `str.splitlines`: that also breaks on `\\f`, `\\v`, and a handful of
    Unicode separators, which a TextMate grammar sees as ordinary characters
    within a line, and it drops that final empty line.
    """
    return text.split("\n")


def tokenize(grammar: Grammar, text: str) -> Iterator[Token]:
    """Tokenizes a whole file."""
    stack = grammar.initial_stack()
    for linenum, line in enumerate(split_lines(text)):
        tokens, stack = tokenize_line(grammar, linenum, line + "\n", stack)
        yield from tokens

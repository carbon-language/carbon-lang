# Format

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Goals and non-goals](#goals-and-non-goals)
-   [Relationship to clang-format](#relationship-to-clang-format)
-   [Architecture](#architecture)
    -   [Pipeline](#pipeline)
    -   [Libraries and entry points](#libraries-and-entry-points)
-   [Token annotations](#token-annotations)
-   [Spacing and line structure](#spacing-and-line-structure)
-   [Line-breaking solver](#line-breaking-solver)
    -   [Search states and break legality](#search-states-and-break-legality)
    -   [Soft column limit](#soft-column-limit)
    -   [Continuation indentation and operand alignment](#continuation-indentation-and-operand-alignment)
    -   [Split penalties](#split-penalties)
    -   [Member-access chains](#member-access-chains)
-   [Comments](#comments)
    -   [Full-line comments and directives](#full-line-comments-and-directives)
    -   [Mid-statement comments](#mid-statement-comments)
    -   [Trailing comments and alignment](#trailing-comments-and-alignment)
-   [Embedded C++ snippets](#embedded-c-snippets)
-   [Minimal edits and range formatting](#minimal-edits-and-range-formatting)
    -   [WhitespaceManager and replacements](#whitespacemanager-and-replacements)
    -   [Range formatting](#range-formatting)
-   [Error recovery](#error-recovery)
-   [Canonical style](#canonical-style)
-   [Testing and invariants](#testing-and-invariants)
-   [Worked examples](#worked-examples)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Overview

`carbon format` formats Carbon source code, normalizing whitespace, indentation,
line breaking, and comment layout.

The formatter combines `clang-format`'s layout and edit model (a penalty-driven
shortest-path line-break solver paired with minimal whitespace replacements)
with syntactic structure from Carbon's
[`TokenizedBuffer`](/toolchain/lex/tokenized_buffer.h) and
[parse tree](parse.md).

The implementation lives in [`toolchain/format/`](/toolchain/format/).

## Goals and non-goals

Goals:

-   Penalty-driven line wrapping under a soft column limit.
-   Block indentation, continuation indentation, and operator-operand alignment.
-   `//` comment re-indentation, over-long line wrapping, and trailing-comment
    alignment.
-   Best-effort formatting of invalid or incomplete code (such as during
    format-on-save), producing output without fabricating tokens or reformatting
    erroneous parse subtrees.
-   Minimal, non-overlapping byte-range edits supporting whole-file formatting,
    line-range formatting, and language-server `TextEdit`s.
-   A single canonical style based on `clang-format`'s LLVM style.
-   Formatting embedded source code inside block string literals with a known
    grammar, starting with C++ snippets (`'''cpp`, `#'''cpp`, and `inline Cpp`)
    formatted with `clang-format`.

Non-goals:

-   Behavior-changing AST rewrites or token insertion, such as brace insertion.
    Formatting preserves Carbon tokens and program semantics: only whitespace,
    comments, [embedded C++ snippets](#embedded-c-snippets), and
    newline-sensitive `#*"""` string escapes may change.
-   Splitting or reflowing general string literals. Carbon
    [disallows adjacent string literal concatenation](/docs/design/lexical_conventions/string_literals.md)
    and forbids raw newlines or backslash-newline line continuations in `"..."`
    literals, while reflowing untagged `'''` block literals would alter their
    decoded value. Only block string literals holding
    [embedded C++ code](#embedded-c-snippets) are reformatted (single-line
    `#*"""` literals that would re-lex as multi-line introducers after a newline
    escape their third quote; see
    [Spacing and line structure](#spacing-and-line-structure)).
-   `clang-format`'s broad configuration surface (hundreds of options, multiple
    preset styles, and config-file discovery). `Style` holds only the parameters
    the formatter reads; see [Canonical style](#canonical-style).

## Relationship to clang-format

Carbon reuses `clang-format`'s layout model and terminology while reading
syntactic structure directly from the lexer and parse tree:

| clang-format subsystem                             | Carbon                                    | Notes                                                                                                                  |
| -------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `FormatTokenLexer` (re-lex + token merging)        | omitted                                   | Carbon lexes once; per-token annotations are indexed by `Lex::TokenIndex`.                                             |
| `UnwrappedLineParser` (token scan into lines)      | token walk in `Formatter::Run`            | Line boundaries come from `;` and parse-tree-annotated comma and brace roles.                                          |
| `TokenAnnotator::AnnotatingParser` (role guessing) | `RoleForNodeKind`                         | Token roles come from `Parse::NodeKind`; no `*`/`&` or `<`/`>` heuristics.                                             |
| Fake-parenthesis precedence reconstruction         | operand-alignment scopes                  | Computed directly from operator subtree spans in `Parse::Tree`.                                                        |
| `MatchingParen` discovery                          | `TokenizedBuffer::GetMatchedClosingToken` | Recorded during lexing.                                                                                                |
| Preprocessor / macro passes                        | omitted                                   | Carbon has no preprocessor or macros.                                                                                  |
| `ContinuationIndenter` + `LineState`/`ParenState`  | `SolveLineBreaks` + `State`               | Dijkstra shortest-path break solver in `line_wrapper.cpp`.                                                             |
| `WhitespaceManager` (+ alignment)                  | `WhitespaceManager`                       | Per-token whitespace `Change`s, `TokenSpan` anchors, and trailing-comment alignment.                                   |
| `BreakableToken` (comment/string breaking)         | `CommentText`                             | `//` line-comment re-indent and wrap. Carbon has no `/* */` comments; see [Goals and non-goals](#goals-and-non-goals). |
| `AffectedRangeManager`                             | `Formatter::AffectedByteRanges`           | Unwrapped-line, brace-pair, and trailing-comment-run closure for range formatting.                                     |
| `FormatStyle`                                      | `Style`                                   | Minimal parameter struct defaulting to Carbon's canonical LLVM-based style.                                            |

## Architecture

### Pipeline

The formatter consumes the toolchain's existing lex and parse output without
re-lexing or re-parsing:

```mermaid
flowchart LR
    src([source text]) --> Lex --> TB[TokenizedBuffer]
    TB --> Parse --> Tree
    TB --> F[Formatter]
    Tree --> F
    F --> Out([formatted text<br/>or minimal edits])
```

Formatting performs one postorder walk of `Parse::Tree` followed by one forward
walk of `TokenizedBuffer`:

1.  **Annotate.** `Formatter`'s constructor walks `Parse::Tree` in postorder,
    merging non-recovery child token ranges on a stack in linear time to
    populate [`TokenInfoStore`](#token-annotations) with token roles (including
    brace and block-comma categories), operator break penalties,
    operand-alignment scopes, member-chain metadata, `InlineImportBody` flags,
    and per-subtree error region IDs (`verbatim_region_id`).
2.  **Build lines.** `Formatter::Run` walks `TokenizedBuffer` (skipping
    lexer-inserted recovery tokens), emitting between-line comments and
    buffering tokens and [mid-statement comments](#mid-statement-comments) into
    unwrapped lines (statements or declarations prior to wrapping). `;`,
    `BlockComma`, non-empty block, lambda, or multi-line struct `{`, and
    block-closing `}` end a line, while inline `StructOpen`/`StructClose` braces
    and `LambdaOpen` or `MultilineStructOpen` closing `}` tokens stay on the
    enclosing line (see
    [Spacing and line structure](#spacing-and-line-structure)).
3.  **Solve.** In `FlushLine`, lines without mid-statement comments that fit
    within `column_limit`, and lines containing error-region tokens
    (`verbatim_region_id >= 0`), skip search; overflowing lines and lines with
    mid-statement comments run [`SolveLineBreaks`](#line-breaking-solver).
4.  **Record.** Each token's leading whitespace (or verbatim source gap, plus
    any rewritten C++ literal or escaped `#*"""` string) and each full-line or
    trailing comment is recorded in `WhitespaceManager` as a `Change`.
5.  **Generate.** `WhitespaceManager::Generate` runs
    [trailing-comment alignment](#trailing-comments-and-alignment), materializes
    the formatted text, and records `TokenSpan` anchors for minimal-edit
    extraction.

### Libraries and entry points

`//toolchain/format` ([`toolchain/format/`](/toolchain/format/)) exposes
[`format.h`](/toolchain/format/format.h) (`Format`, `FormatReplacements`,
`ApplyReplacements`, `Replacement`, and `LineRange`) and `style.h` (`Style`),
backed by six internal modules:

-   `token_info`: `TokenRole`, `TokenInfo`, `TokenInfoStore`, and the
    `SpacesBefore`, `EscapeTripleQuoteString`, `CanBreakBefore`, `SplitPenalty`,
    and `RenderedWidth` queries.
-   `comment`: `CommentText` for `//` block re-indentation and line wrapping.
-   `cpp_snippet`: `CppSnippet` for embedded C++ string literal formatting with
    `clang-format`.
-   `line_wrapper`: `SolveLineBreaks` shortest-path line-break solver and
    `LineBreakConstraints`.
-   `whitespace_manager`: `WhitespaceManager` and `TokenSpan` (recording
    per-item `Change`s).
-   `formatter`: `Formatter`, driving the annotation pass, unwrapped-line
    construction, whitespace recording, and replacement extraction.

Two toolchain components call into `format`:

-   **Driver**
    ([`format_subcommand.cpp`](/toolchain/driver/format_subcommand.cpp)):
    formats files atomically in place (preserving symlinks and skipping
    unchanged files), to `--output=FILE`, or to stdout (`--output=-`),
    optionally restricted to a single input file with `--lines=START:END`.
    Duplicate `-` stdin arguments and `--lines` or `--output=FILE` with multiple
    input files are diagnosed and rejected. On lex or parse errors, it still
    writes best-effort output and exits non-zero.
-   **Language server**
    ([`handle_formatting.cpp`](/toolchain/language_server/handle_formatting.cpp)):
    handles `textDocument/formatting` and `textDocument/rangeFormatting`
    (`LspRangeToLineRange`), converting `FormatReplacements` byte offsets to the
    negotiated position encoding (UTF-16 code units or UTF-8 bytes) and
    returning no edits for `file_test` files.

## Token annotations

Per-token formatting metadata is stored in `TokenInfoStore`, a
`FixedSizeValueStore` indexed by `Lex::TokenIndex`. Data already on
`TokenizedBuffer` is read directly: token kinds from `GetKind` and source
spellings from `GetTokenText` (preserving numeric radixes, string escapes, and
`r#` raw identifiers). Each `TokenInfo` stores:

-   `role`: syntactic `TokenRole` derived from `Parse::Tree`.
-   `is_call_link`: true on a member-access `.` or `->` whose accessed member is
    directly called (`CallExprStart`) or indexed (`IndexExprStart`).
-   `column_width`: byte length of the token's first physical line (excluding
    any trailing `\r`, using the escaped spelling when `EscapeTripleQuoteString`
    applies).
-   `open_scopes` / `close_scopes`: counts of operand-alignment scopes opening
    and closing at this token.
-   `break_penalty_after`: break-after penalty for an infix, `in`, or constraint
    operator token, or `-1` otherwise.
-   `break_penalty_before`: break-before penalty for a member-access `.` or `->`
    (`150`, lowered to `35` on the outermost link of chains with at least two
    links), or `-1` otherwise.
-   `member_chain_id`: receiver-root token index of the enclosing left-nested
    member-access chain, or `-1`.
-   `is_cpp_string`: true when the token is an `InlineImportBody` string
    literal.
-   `verbatim_region_id`: non-negative ID of the outermost error subtree
    containing this token, or `-1` for error-free tokens, file boundaries, and
    recovery tokens.

`RoleForNodeKind` and `Formatter`'s postorder walk assign `TokenRole` values:

-   `Unknown`: no special role; token-kind rules govern.
-   `PostfixBracket`: `(` or `[` opening a call (`F(x)`), explicit or implicit
    parameter list (`F[T: type](x: T)`), subscript (`a[i]`), or type/form
    constructor (`array(i32, 4)`, `typeof(x)`, `form(...)`).
-   `PrefixOperator`: prefix unary operator (`*p`, `-x`, `&v`, `not b`,
    `const T`) or leading designator `.` (`.Red`, `{.x = 1}`).
-   `PostfixOperator`: postfix unary operator (`p*`).
-   `MemberAccess`: member-access or qualified-name `.` and `->` (`a.b`, `p->x`,
    `fn A.F()`, `fn Box(T: type).Make()`, `namespace A.B;`), covering
    `MemberAccessExpr`, `PointerMemberAccessExpr`,
    `IdentifierNameQualifierWithoutParams`, and
    `IdentifierNameQualifierWithParams`, and distinguishing pointer member
    access from `ReturnType` `->`.
-   `BlockOpen`: `{` opening a declaration or statement block
    (`ChoiceDefinitionStart`, `ClassDefinitionStart`, `CodeBlockStart`,
    `FunctionDefinitionStart`, `ImplDefinitionStart`,
    `InterfaceDefinitionStart`, `MatchFirstDefinitionStart`,
    `MatchHandlerStart`, `MatchStatementStart`,
    `NamedConstraintDefinitionStart`).
-   `LambdaOpen`: `{` opening a lambda body (`LambdaDefinitionStart`).
-   `StructOpen` / `StructClose`: matched non-recovery `{` and `}` of an inline
    `StructLiteral`, `StructTypeLiteral`, or `StructPattern` (closed, without a
    direct trailing comma, and not an empty `{ // comment }` body).
-   `MultilineStructOpen`: `{` of a `StructLiteral`, `StructTypeLiteral`, or
    `StructPattern` that expands across lines (`RoleForNodeKind` assigns this
    default on `Struct*Start`, and `Formatter` promotes closed inline structs to
    `StructOpen` / `StructClose` unless a direct trailing comma or interior
    comment on an empty `{}` keeps it multi-line).
-   `BlockComma`: `,` ending an unwrapped line inside an expanded brace list,
    covering `ChoiceAlternativeListComma` in `choice` definitions and direct
    field commas (`StructLiteralComma`, `StructTypeLiteralComma`,
    `PatternListComma`) of a trailing-comma `MultilineStructOpen` struct.

`OperatorInfoForNodeKind` maps `InfixOperator*` and `ShortCircuitOperator*`
nodes to their operator tokens (`CARBON_PARSE_NODE_KIND_INFIX_OPERATOR`) and
queries `Parse::PrecedenceGroup` (alongside matching tiers for `ForIn` and
constraint/requirement operator nodes) to compute
`{break_penalty, aligns_operands}`:

-   Assignment and compound-assignment operators (`PrecedenceGroup` matching
    `=`) set `aligns_operands = false` and use `penalty_break_assignment` (`2`).
-   Non-assignment expression operators (plus `==` and expression-level `impls`
    constraints) set `aligns_operands = true`, while `ForIn`, `RequirementAnd`,
    `Impl*As`, `Require*Impls`, and `RequirementEqual` set
    `aligns_operands = false`.
-   `InfixPrecedencePenalty` counts how many tiers in `{=, and, ==, +}` bind
    strictly looser (`PrecedenceGroup::GetPriority` returning `LeftFirst` on a
    distinct group), assigning `5` to `or`, `and`, `as`, `in`, and
    `RequirementAnd`; `6` to relational operators (`==`, `!=`, `<`, `<=`, `>`,
    `>=`, `<=>`); `7` to additive (`+`, `-`), bitwise (`&`, `|`, `^`), shift
    (`<<`, `>>`), modulo (`%`), `impls`, and `Impl*As`; and `8` to
    multiplicative (`*`, `/`). Looser operators break first at equal bracket
    depth (so `a & b == c` breaks after `==` at `6` before `&` at `7`), while
    incomparable operators in Carbon require explicit parentheses that add
    `PenaltyPerBracketDepth` (`20`) per nesting level.
-   Inside `WhereExpr`, `where` opens an alignment scope on its first
    requirement so `and`-separated requirements align, and `RequirementEqual`
    (`.Assoc = Type`) uses penalty `9` so clauses wrap before `where` (`3`) or
    after `and` (`5`) before splitting an individual rewrite.

## Spacing and line structure

`SpacesBefore(tokens, token_infos, left, right)` computes horizontal spaces
between adjacent tokens on the same physical line, defaulting to 1 space when no
tighter rule matches:

| Construct                                                                        | Spacing                                                                                                                |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Binary operators, `=`, compound assignment, `as`, `and`, `or`                    | 1 space each side                                                                                                      |
| Symbolic prefix (`*p`, `-x`, `&v`) and postfix (`p*`) operators                  | 0 spaces on operand side unless gluing tokens would re-lex (`- -x`); word prefix operators (`not x`) keep 1 space      |
| Leading designator `.`                                                           | 0 spaces after, 1 space before (`case .Red`, `{.x = 1}`)                                                               |
| Member access `.` and `->`                                                       | 0 spaces each side (`a.b`, `p->x`, `t.0.1`), except 1 space before `.` after a bare integer literal (`2 .rt`)          |
| `PostfixBracket` `(` and `[`                                                     | 0 spaces before (`F(x)`, `F[T: type](x: T)`, `array(i32, 4)`), except 1 space after `fn` or `forall`                   |
| Colons `:` and `:?`                                                              | 0 spaces before, 1 space after (`x: i32`, `x:? Form`, `.x: i32`)                                                       |
| Return arrows `) -> Type` and `) ->? Form`                                       | 1 space each side                                                                                                      |
| `,`                                                                              | 0 spaces before, 1 space after (inline); `BlockComma` ends unwrapped line                                              |
| `;`                                                                              | 0 spaces before; ends unwrapped line                                                                                   |
| Inside `(...)`, `[...]`, inline `{...}` (`StructOpen`/`StructClose`), empty `{}` | 0 spaces (`{.x = 0, .y = 0}`, `{}`)                                                                                    |
| `BlockOpen` closing `}`                                                          | continues line before `else`, `,`, `;`, `)`, or `]` (`} else {`, `});`); ends unwrapped line otherwise                 |
| `LambdaOpen` or `MultilineStructOpen` closing `}`, and `StructClose` `}`         | stays on the enclosing unwrapped line (`fn F(p: {}*)`, `var p: {.a: i32}* = &s;`, `{.a = 1}.a`, `fn F() -> {.x: i32}`) |

`WouldMergeTokens` preserves `1` space whenever removing whitespace between
`left` and `right` would change lexed tokens:

-   **Integer literals before `.`.** `1` space is kept between `IntLiteral` and
    `.` unless `IntLiteral` immediately follows a `MemberAccess` `.` or `->`
    with no intervening comment, preventing `2 .rt` from lexing as a real
    literal while keeping chained tuple indexing (`t.0.1`, `p->0.1`) tight.
-   **Adjacent operator symbols.** `1` space is kept between adjacent symbol
    tokens whose concatenation forms a combined token, such as `- -x` (two
    `Minus` tokens, never `--x`) or symbolic operators before `=`, `==`, `=>`,
    `--`, `->`, or `->?`.

`EscapeTripleQuoteString` guards against a related newline-sensitivity hazard in
single-line `#`-delimited string literals: on an unterminated final source line,
`#"""// "#` lexes as a single-line raw string, while appending a newline after
the line causes `Lex::StringLiteral::Introducer::Lex` to treat `#"""` followed
by `//` as an unclosed multi-line `"""` introducer. When a single-line `#*"""`
literal contains a `//` comment marker hiding its closing `"#*` delimiter (with
no `'`, `#`, or `"` before `//`), the formatter escapes the third opening quote
as `\<hashes>"` (`#""\#"// "#`), preserving `TokenKind::StringLiteral` and the
decoded string value.

Brace and comma roles govern multi-line expansion:

-   **Blocks, lambdas, and `choice` alternatives.** Non-empty `BlockOpen` and
    `LambdaOpen` `{...}` bodies (plus unclosed `{` blocks and empty
    `{ // comment }` bodies) expand across lines at `+indent_width`, while empty
    `{}` with neither tokens nor comments stays compact. In `choice`
    definitions, `ChoiceAlternativeListComma` carries `BlockComma`, flushing the
    unwrapped line after each `,` (and flushing any final alternative before
    `}`) so every alternative sits on its own `+indent_width` line while
    parameter commas inside `Alt(x: i32, y: i32)` remain inline.
-   **Struct literals, struct type literals, and struct patterns.** A closed
    `StructLiteral`, `StructTypeLiteral`, or `StructPattern` without a direct
    trailing comma (and not an empty `{ // comment }` body) is annotated as a
    matched `StructOpen`/`StructClose` pair: `{` and `}` stay inline on the
    current unwrapped line with `0` interior spaces (`{.x = 0, .y = 0}`),
    wrapping through [`SolveLineBreaks`](#line-breaking-solver) only when the
    enclosing line overflows `column_limit`. A direct trailing comma
    (`{.x = 0, .y = 0,}`), an empty body containing comments (`{ // comment }`),
    or an unclosed `{` marks `{` as `MultilineStructOpen` (and marks direct
    field commas as `BlockComma`), expanding one field per line at
    `+indent_width`.
-   **Expression-level closing `}`.** `StructClose` and the closing `}` of
    `LambdaOpen` or `MultilineStructOpen` stay on the enclosing unwrapped line,
    keeping trailing `;`, `,`, `)`, `]`, postfix `*`, `.`, `->`, `(`, `[`, `=>`,
    and header `{` tokens attached to the composite type, value, pattern, or
    lambda.

Blank lines between unwrapped lines and before full-line comment blocks are
clamped to `max_empty_lines_to_keep`, and removed at the start of the file,
immediately after `{` (including when `{` carries a trailing `//` comment), and
immediately before `}`.

## Line-breaking solver

An unwrapped line without [mid-statement comments](#mid-statement-comments) that
fits within `column_limit` (or any line containing a token with
`verbatim_region_id >= 0`) skips search. Overflowing lines and lines with
mid-statement comments run `SolveLineBreaks`, a uniform-cost (Dijkstra)
shortest-path search over layout prefixes modeled on `clang-format`'s
`ContinuationIndenter`.

### Search states and break legality

Each `State` in the solver's node arena represents a laid-out prefix of the line
as a 24-byte trivial struct (six `int32_t` fields, with no per-state vectors):

-   `index`: next token position in the line.
-   `column`: current end column on the physical line.
-   `line_indent`: starting indent of the current physical line.
-   `scope_id`: interned continuation-indent stack ID in
    [`ScopeTable`](#continuation-indentation-and-operand-alignment).
-   `parent`: predecessor `State` index in the arena (or `-1` at the root).
-   `newline_indent`: indentation column if the transition into `State` broke
    the line before `line[index - 1]`, or `-1` for a same-line step.

Before search begins, `SolveLineBreaks` precomputes `bracket_depths[i]` (the
count of unclosed `(`, `[`, and inline `StructOpen` `{` brackets before
`line[i]`), [`ChainInfo`](#member-access-chains), and two backward-liveness
bitvectors from `i = token_count - 1` down to `1`:

-   `scope_open_ahead[i]`: true when some token at or after `i` (before the next
    `must_break_before` boundary) pushes a column-dependent scope entry (an
    inline open bracket, `open_scopes > 0`, or a member-chain root).
-   `line_indent_live[i]`: true when `line_indent` can still be read by an
    immediate-break-after-open-bracket demotion at or after `i` before the next
    `must_break_before` boundary. A `must_break_before` at `i` clears both flags
    for positions after `i` because the forced break starts a new line, while
    still setting `line_indent_live[i]` if `line[i - 1]` itself is an inline
    open bracket with `close_scopes == 0`.

At each step, Dijkstra pops the lowest-penalty `State`, stops immediately if
`index == token_count`, and otherwise prunes settled states using `min_column`
(`Map<SettledKey, int32_t>`). With `col` set to `0` when
`must_break_before[index]` holds and `column` otherwise, `SettledKey` records:

-   `index`: current token position.
-   `key_column`: `col` when `scope_open_ahead[index]` is true, or `-1`
    otherwise.
-   `effective_line_indent`: `line_indent` when `line_indent_live[index]` is
    true, or `0` otherwise.
-   `scope_id`: active `ScopeTable` ID.

`min_column` stores the smallest `col` settled for each `SettledKey` and drops
any later state with `col >= min_column[key]`. When `scope_open_ahead[index]` is
true, `key_column == col` performs exact column deduplication because a wider
column could anchor a later scope or chain receiver differently; when false,
`col` affects only current-line overflow, and monotonic excess-column penalties
plus non-decreasing Dijkstra settlement order make any state with a larger or
equal column Pareto-dominated. If state generation exceeds `MaxStatesGenerated`
(`100,000`), `SolveLineBreaks` falls back to `SolveLineBreaksGreedy`, a linear
first-fit pass sharing `ScopeTable`, `ApplyTokenScopes`, and `ChainInfo`.

When mid-statement comments appear on the line, `FlushLine` passes
`LineBreakConstraints` to `SolveLineBreaks` (and `SolveLineBreaksGreedy`):

-   `must_break_before`: bitset over token indices marking positions preceded by
    a mid-statement trailing or full-line comment. At a `must_break_before`
    position, the same-line transition is suppressed and the break transition is
    always generated with `0` split penalty, overriding `CanBreakBefore` and
    member-chain break blocking so comments at normally unbreakable positions
    still break cleanly (and setting `ChainMeta::has_forced_break` on any
    enclosing builder chain).
-   `trailing_comment_widths`: per-token extra column width (`1 +` trimmed
    comment byte length) added after placing token `k - 1` when a trailing
    comment precedes token `k`. Open-bracket and member-chain anchors still
    record the token's own column, while `State::column`, soft-limit overflow,
    and builder-chain `total_width` include the trailing comment so earlier
    tokens wrap when `code // comment` would exceed `column_limit`.

Unless forced by `must_break_before`, `CanBreakBefore` forbids breaks in
positions that violate syntax or style:

-   Before `,`, `;`, `:`, `:?`, `)`, `]`, `}`, or `=>`.
-   Before a `BlockOpen`, `LambdaOpen`, or `MultilineStructOpen` `{`, or before
    `else` after `}` (`StructOpen` `{` allows breaking before `{` so an inline
    struct initializer can move as a unit onto a continuation line).
-   Before a `PostfixBracket` `(` or `[`, unless preceded by `fn` or `forall`.
-   Before `=` or any operator with a break-after penalty
    (`break_penalty_after >= 0`).
-   After `.`, `->`, or `->?` (member chains, designators, and return
    types/forms break before the operator).
-   After a symbolic `PrefixOperator` or before a `PostfixOperator`, preserving
    Carbon's whitespace-sensitive unary-operator fixity rules.

All other token boundaries allow breaks and are ranked by
[`SplitPenalty`](#split-penalties).

### Soft column limit

`column_limit` is a soft cost: each column past `column_limit` adds
`penalty_excess_character`, computed incrementally above
`max(previous_column, column_limit)` on the same line (and above
`min(break_indent, column_limit)` after a break, so indentation past
`column_limit` is still charged). When an over-long token or unbreakable span
makes overflow unavoidable, the solver still finds the minimum-penalty layout.

### Continuation indentation and operand alignment

Block indentation advances by `indent_width` per `{}` level; statement
continuations default to `indent + continuation_indent_width`. Active
continuation-indent stacks are interned in a solver-local persistent DAG,
`ScopeTable`, whose `Node {parent, column, count}` entries coalesce consecutive
matching columns above `parent` (rooted at ID `0` with `parent = -1`,
`column = indent + continuation_indent_width`, and `count = 1`; `Pop` never
removes that final root count).

Because `.`, `->`, `()`, and `[]` bind tighter than infix operators and `where`,
and any bracketed expression inside a chain's receiver or call arguments closes
before the chain's final `.` or `->`, operand scopes, member-chain receiver
anchors, and inline brackets nest in strict LIFO order. As each token at `pos`
is placed from `start_column` to `token_end`, `ApplyTokenScopes` updates
`scope_id` in symmetric order:

1.  **Operand scopes (`open_scopes > 0`).** Pushes
    `max(start_column, TopColumn(scope_id))` (`open_scopes` times), aligning
    wrapped operands of a binary expression or `where` clause with its first
    operand (`AlignOperands = Align`) without outdenting left of the enclosing
    anchor.
2.  **Member-chain roots (`ChainInfo::root_slot[pos] >= 0`).** Pushes
    `start_column + continuation_indent_width` at the first token of a chain's
    receiver.
3.  **Inline brackets.** Inline open brackets `(`, `[`, and `StructOpen` `{`
    push `token_end`, aligning wrapped elements (including wrapped
    `.field = value` entries of an inline struct) just past the opener
    (`AlignAfterOpenBracket = Align`); matching inline closing brackets `)`,
    `]`, and `StructClose` `}` pop one entry. When a line break is taken
    **immediately after** `(`, `[`, or `StructOpen` `{` (with
    `close_scopes == 0` on that opener), that bracket's anchor is replaced with
    `line_indent + continuation_indent_width` so long first arguments, single
    arguments, and over-long struct fields wrap to continuation indent.
4.  **Member-chain closes (`ChainInfo::closes_chain[pos]`).** Pops the chain's
    receiver anchor as its final `.` or `->` token (`last_member_pos`) is
    placed, after any break before that final link has already read
    `TopColumn(scope_id)`.
5.  **Operand-scope closes (`close_scopes > 0`).** Pops `close_scopes` entries
    after the last operand token.

### Split penalties

Configurable penalties on `Style`:

| Field                                       | Default   | clang-format                           |
| ------------------------------------------- | --------- | -------------------------------------- |
| `penalty_excess_character`                  | 1,000,000 | `PenaltyExcessCharacter`               |
| `penalty_break_assignment`                  | 2         | `PenaltyBreakAssignment`               |
| `penalty_break_before_first_call_parameter` | 19        | `PenaltyBreakBeforeFirstCallParameter` |

Each optional break transition at `index` adds
`SplitPenalty(tokens, token_infos, left, right, style)` plus
`PenaltyPerBracketDepth * bracket_depths[index]` (`20 * bracket_depths[index]`,
so outer breaks beat breaks nested inside `(`, `[`, or inline `{...}`):

| Break position                                                         | Base penalty                                | clang-format analog                   |
| ---------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------- |
| After `,`                                                              | 1                                           | comma (1)                             |
| After assignment or initializer `=`                                    | `penalty_break_assignment` (2)              | `PenaltyBreakAssignment`              |
| Default (unmatched boundary, including before `where`)                 | 3                                           | default (3)                           |
| After non-assignment infix, `in`, or requirement operator              | 5 to 9 (`PrecedenceGroup` / clause rank)    | operator precedence                   |
| After `(`, `[`, or `StructOpen` `{`                                    | `penalty_break_before_first_call_parameter` | `PenaltyBreakBeforeFirstCallParameter` |
| Before member-access `.` or `->` (final link of `>= 2`-link chain)     | 35                                          | `isMemberAccess` last link            |
| After word prefix (`not`, `const`, `ref`) or `...`, or before `unsafe` | 60                                          | unary-operator penalty                |
| Before return-type `->` or return-form `->?`                           | 60                                          | `PenaltyReturnTypeOnItsOwnLine`       |
| After binding `:` or `:?`                                              | 100                                         | type-annotation split                 |
| Before member-access `.` or `->` (1-link or non-final) or other `.`    | 150                                         | `isMemberAccess` (150)                |
| After any other keyword (`fn`, `var`, `return`, `private`, ...)        | 200                                         | `TT_StartOfName` specifier separation |

### Member-access chains

Left-nested `.` and `->` chains share `member_chain_id` (the token index of the
chain's receiver root). `ComputeChainInfo` precomputes `ChainInfo` and per-chain
`ChainMeta` (`chain_id`, `root_pos`, `last_member_pos`, `total_width`,
`seen_call`, `is_builder`, `has_forced_break`) once per unwrapped line:

-   **Break before `.` / `->`.** `CanBreakBefore` forbids breaking after `.` or
    `->`, keeping `.member()` together.
-   **Cheaper final link on multi-link chains.** `Formatter` lowers the
    outermost member access from `150` to `MemberAccessLastLinkBreakPenalty`
    (`35`) on chains with at least two links (`>= 2`), so overflowing
    non-builder chains break at their tail first while single-dot qualified
    names such as `Pkg.LongType` keep `150` (allowing
    `fn F(...) -> Pkg.LongType` to break before `->` at `60`).
-   **Builder chains.** Call and subscript links are identified from
    `Parse::Tree` (`TokenInfo::is_call_link`, set when `CallExprStart` or
    `IndexExprStart` wraps a member access). Once a chain has seen an initial
    call or subscript (on the receiver or an earlier link), subsequent
    `is_call_link` links are marked fluent (`ChainInfo::is_fluent`) and promote
    the chain to `ChainMeta::is_builder`:
    -   `ComputeChainInfo` extends the chain's span `end` from `last_member_pos`
        across the final member (including a parenthesized member expression),
        any attached postfix `(...)` or `[...]` brackets, and any immediately
        following unbreakable tokens (`!CanBreakBefore`, such as `;`, `,`, `)`,
        postfix `*`, or `=`), sets `has_forced_break` if `must_break_before`
        falls anywhere in `(root_pos .. end]`, and computes the single-line
        `total_width` (`RenderedWidth` across `line[root_pos .. end]` plus any
        `trailing_comment_widths[end]`).
    -   At each link of a builder chain, subtracting `continuation_indent_width`
        from `TopColumn(scope_id)` recovers `receiver_start_column` directly
        from `ScopeTable` without per-chain solver state. When
        `!has_forced_break` and
        `receiver_start_column + total_width <= column_limit`, every fluent link
        stays packed on the current line; otherwise (`chain_must_break`),
        non-fluent links stay attached while every fluent link is forced to
        break.
    -   Non-builder chains (`!is_builder`, such as `(a + b).f1.f2.f3`) break
        minimally using ordinary split penalties.
-   **Receiver-anchored indentation.** Breaks before `.` or `->` read
    `TopColumn(scope_id)` (`receiver_start_column + continuation_indent_width`).
    Because inner `(`, `[`, and nested-chain scopes push above the receiver
    anchor and pop before subsequent outer links, fluent links after a
    multi-line call argument still break cleanly to
    `receiver_start_column + continuation_indent_width`.

## Comments

### Full-line comments and directives

The lexer coalesces consecutive `//` lines with matching indentation into a
single comment block returned by
[`TokenizedBuffer::GetCommentText`](/toolchain/lex/tokenized_buffer.h).
`CommentText` (`comment.h`) re-indents each line to `indent` (stripping `\r`
before measuring width) and wraps lines exceeding `column_limit` at whitespace,
repeating `//` plus the line's leading post-`//` spaces or tabs on every wrapped
line and preserving interior spacing. Consecutive comment lines are never merged
into paragraphs, preserving lists, tables, and code examples.

Two cases skip word-wrapping after re-indentation:

-   A single word longer than the available width remains whole on an over-long
    line.
-   A line where `//` is followed immediately by a non-whitespace byte, such as
    a `//===` divider, a malformed comment, or a `//@` tooling directive
    (`//@include-in-dumps`, `//@dump-sem-ir-begin`), is never wrapped, keeping
    directives intact in toolchain tests.

### Mid-statement comments

Comments between unwrapped lines (or where `ShouldFlushBeforeComments` flushes
`current_line_` first: immediately before `FileEnd`, a recovery `}` or non-empty
recovery `{`, a non-`StructClose` closing `}`, or after a line containing an
error-region token) are emitted directly at block `indent_`.

Comments that appear while `current_line_` remains open stay inside the
enclosing unwrapped line as `MidLineComment` entries indexed by the following
token's position `before_index`:

-   A comment sharing the preceding token's source line (`IsTrailingComment`
    with only horizontal whitespace since that token) is marked `is_trailing`,
    adding `1 +` its trimmed width to
    `LineBreakConstraints::trailing_comment_widths[before_index - 1]` and
    attaching to token `before_index - 1` with
    `WhitespaceManager::AddTrailingComment`.
-   Full-line comments record surrounding blank-line counts (clamped to
    `max_empty_lines_to_keep`) without altering the unwrapped line's own leading
    newline state, and re-indent with `CommentText` to
    `newline_indents[before_index]`, matching the solved continuation,
    open-bracket, or operand-alignment column of the token they precede.
-   Every `MidLineComment` before a token (`before_index <
    current_line_.size()`) sets `must_break_before[before_index]` so the
    following token starts on a new indented physical line; any comment at
    `before_index == current_line_.size()` (before a skipped recovery token) is
    emitted after the line's final token.

### Trailing comments and alignment

[`TokenizedBuffer::IsTrailingComment`](/toolchain/lex/tokenized_buffer.h)
identifies comments that share a source line with preceding code. Both
end-of-statement trailing comments (emitted after flushing an unwrapped line)
and mid-statement trailing comments (emitted after token `before_index - 1`
within an unwrapped line) are recorded with
`WhitespaceManager::AddTrailingComment` (`1` leading space, no trailing
newline). Trailing comments are not wrapped; see [Future work](#future-work).

When `align_trailing_comments` is true, `WhitespaceManager` aligns consecutive
trailing comments after all line breaks are chosen. It partitions `Change`s into
physical lines (measuring rewritten token text when present and stripping `\r`)
and pads each maximal run of lines sharing the same `indent_level` so their `//`
markers match the rightmost natural column in the run. A blank line, an
`indent_level` change, a comment-only line, a code line without a trailing
comment, or a line containing a multi-line token (such as a `'''` block string
literal) ends the run. A single trailing comment keeps its `1`-space separation.

## Embedded C++ snippets

When `format_cpp_snippets` is true, `CppSnippet` (`cpp_snippet.h`) reformats
both normal `'''...'''` and `#`-delimited raw `#'''...'''#` multi-line block
string literals that hold C++ code:

-   literals whose introducer file type indicator matches `cpp`, `cc`, `cxx`,
    `c++`, `h`, `hpp`, `hxx`, or `hh` (case-insensitive, ignoring any trailing
    `//` comment on the introducer line); and
-   `InlineImportBody` literals of `inline Cpp` / `import Cpp inline`
    declarations (`TokenInfo::is_cpp_string`), regardless of indicator.

Single-line `"..."` literals are ignored. `CppSnippet` strips any `\r` line
endings and the closing delimiter line's leading whitespace from each non-blank
body line, formats the body with `clang::format::reformat`, and re-indents the
body and closing delimiter to the token's actual line indent (`line_indent`,
accounting for wrapped continuation lines), preserving the opening introducer
line verbatim.

`clang-format` runs with LLVM base style configured from Carbon's `Style`:
`ColumnLimit = column_limit - line_indent` (skipping formatting when fewer than
`20` columns remain), `IndentWidth = indent_width`,
`ContinuationIndentWidth = continuation_indent_width`, `UseTab = UT_Never`,
`PointerAlignment = PAS_Left` (`T* p`), `DerivePointerAlignment = false`,
`SortIncludes.Enabled = false`, `SortUsingDeclarations = SUD_Never`, and
`FixNamespaceComments = false`.

The literal is left unchanged if:

-   the body or formatted output contains the literal's active Carbon escape
    prefix (`\` in `'''...'''`, `\#` in `#'''...'''#`, and so on), preserving
    Carbon escape sequences while allowing plain C++ `\` escapes and macro line
    splices inside `#'''` raw block strings;
-   the body is empty or mis-indented, or contains control characters (including
    literal tabs, which Carbon multi-line strings disallow);
-   the input or formatted body contains the closing delimiter (`'''` followed
    by the matching number of `#` characters); or
-   `clang::format::reformat` reports `!status.FormatComplete` or is not a
    fixed point on a second `reformat` pass.

When formatting leaves the literal unchanged, `CppSnippet` returns
`std::nullopt` and the token stays an ordinary anchor; otherwise
`WhitespaceManager` emits the rewritten literal in place of the source token.

## Minimal edits and range formatting

### WhitespaceManager and replacements

`WhitespaceManager` collects one compact `Change` (`ChangeKind`) per
non-recovery token or comment, referencing source slices with `llvm::StringRef`
for trailing comments and verbatim error gaps and storing owned text in a side
deque only for re-indented comment blocks and rewritten string literals
([embedded C++ snippets](#embedded-c-snippets) and `EscapeTripleQuoteString`
literals). Line breaks are stored on the `Change` that follows them, and
`Generate` appends a single trailing `\n` for non-empty files.

Lexer-inserted bracket recovery tokens (`TokenizedBuffer::IsRecoveryToken`) are
skipped in `Formatter::Run` (updating `indent_` and flushing as needed on
synthesized `{` and `}`), so fabricated `)`, `]`, and `}` tokens are never
emitted into formatted output or replacements.

While emitting output, `Generate` records a `TokenSpan`
(`{source_begin, output_begin, length}`) for every unchanged source token
(including `VerbatimGapToken` items without rewritten text), omitting rewritten
string literals so their rewritten bytes fold into the surrounding gap edit.

`Formatter::ComputeReplacements` (`FormatReplacements` in
[`format.h`](/toolchain/format/format.h)) walks the gaps between consecutive
`TokenSpan`s (plus the leading and trailing file gaps), trims matching leading
and trailing bytes (without splitting multi-byte UTF-8 sequences) from each
differing `(source_gap, output_gap)` pair, and emits a minimal `Replacement`
(`{offset, length, text}`). Trimming common prefixes keeps pure line-start
indentation edits anchored after the preceding line's `\n`, attributing them to
the line being indented. Replacements are sorted by byte offset and
non-overlapping, and `ApplyReplacements(source, replacements)` reproduces
`Format`'s output on all inputs.

### Range formatting

`FormatReplacements` accepts an optional 1-based inclusive `LineRange`.
`Formatter::AffectedByteRanges` expands the requested lines to a fixed point in
linear time across three couplings, then filters the whole-file `Replacement`s
with a single two-pointer sweep to gaps overlapping (or inserting at the start
of) the affected byte ranges:

-   **Whole unwrapped lines.** Any unwrapped line overlapping the range
    (including across multi-line tokens and mid-statement comments) is included
    in full so statements are never partially wrapped.
-   **Matched braces.** If either `{` or `}` of a non-recovery brace pair lies
    on an affected line, its partner's line is included too.
-   **Trailing-comment alignment runs.** Any aligned trailing-comment group
    overlapping an affected line is included in full so `--lines` never aligns
    one line's comment to a neighbor's formatted column while leaving that
    neighbor unformatted.

Because block indentation comes from brace depth, edits earlier in the file
cannot shift unaffected lines, and applying range replacements followed by whole
formatting produces the same output as whole formatting.

## Error recovery

`Parse::Tree` is structurally valid even when lex or parse errors occur.
`Format` and `FormatReplacements` always produce output and return `false` when
`tokens.has_errors() || tree.has_errors()`.

Two mechanisms prevent error recovery from corrupting user code:

-   **Recovery tokens are never materialized.** Inline lexer recovery brackets
    (`IsRecoveryToken`) are skipped during emission, and synthesized recovery
    and file-boundary tokens are excluded from parse-tree subtree spans.
-   **Per-subtree verbatim error regions.** The postorder annotation pass
    assigns a distinct non-negative `verbatim_region_id` to every outermost
    error subtree (a node with `tree.node_has_error(node)` and no erroneous
    ancestor), excluding `FileStart`, `FileEnd`, and recovery tokens:
    -   The first token of each error subtree is indented normally by the
        surrounding code.
    -   Any unwrapped line containing a token with `verbatim_region_id >= 0`
        skips `SolveLineBreaks` (breaking at `indent +
        continuation_indent_width` only after any mid-statement comments outside
        verbatim gaps).
    -   `VerbatimGapBefore(token)` copies the raw source `StringRef` slice
        (whitespace and comments alike) verbatim only when `token - 1` and
        `token` are non-recovery tokens sharing the same
        `verbatim_region_id >= 0` (except an empty gap between `""` and a
        `"..."` token, where `SpacesBefore` inserts a space so a trailing
        newline cannot fuse them into `"""`, while `EscapeTripleQuoteString`
        still escapes ambiguous `#*"""` tokens). Distinct IDs per error subtree
        keep comments between adjacent erroneous statements formatted normally,
        and excluding `FileEnd` preserves trailing comments at the end of the
        file.

## Canonical style

A default-constructed `Style` defines Carbon's canonical style. `snake_case`
rows below are `Style` fields; remaining rows are fixed behavior:

| Setting                         | Canonical value                                                                                                                                      | Source                                                        |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `column_limit`                  | 80                                                                                                                                                   | LLVM                                                          |
| `indent_width`                  | 2                                                                                                                                                    | LLVM                                                          |
| `continuation_indent_width`     | 4                                                                                                                                                    | LLVM                                                          |
| `max_empty_lines_to_keep`       | 1                                                                                                                                                    | LLVM                                                          |
| `align_trailing_comments`       | `true` (1 space minimum)                                                                                                                             | LLVM (`AlignTrailingComments`)                                |
| `format_cpp_snippets`           | `true`                                                                                                                                               | Carbon C++ interop                                            |
| Penalty fields                  | see [Split penalties](#split-penalties)                                                                                                              | LLVM                                                          |
| Brace attachment                | attach `{` to header                                                                                                                                 | LLVM                                                          |
| Binary operator breaks          | break after operator                                                                                                                                 | LLVM                                                          |
| Bracket and operand alignment   | align (`AlignAfterOpenBracket`, `AlignOperands`)                                                                                                     | LLVM                                                          |
| Argument and parameter packing  | bin-pack                                                                                                                                             | LLVM                                                          |
| Multi-line string opener        | keep on statement line                                                                                                                               | LLVM (`AlwaysBreakBeforeMultilineStrings: false`)             |
| Non-empty `{}` bodies           | block/declaration `{}` and `choice` alternatives expand across lines; struct `{...}` stays inline unless written with a trailing comma or overflowing | Carbon divergence (`AllowShortFunctionsOnASingleLine` family) |
| `match` arm indentation         | indent one `indent_width` inside `{}`                                                                                                                | Carbon divergence (`IndentCaseLabels`)                        |
| Comment reflow                  | wrap over-long `//` lines; never merge lines                                                                                                         | Carbon divergence (`ReflowComments`)                          |
| Blank lines at block/file edges | drop at file start, after `{`, and before `}`                                                                                                        | Carbon divergence (`KeepEmptyLines`)                          |

`match` arms are block statements in Carbon's grammar and indent like any other
statement in `{}` (C++ `case` labels are unindented in LLVM style). A non-empty
file starts at its first token or comment and ends with one `\n`.

## Testing and invariants

On error-free input without control characters other than `\t`, `\r`, and `\n`,
the formatter maintains four invariants:

1.  **Minimal-edit equivalence.** Applying `FormatReplacements(tree)` with
    `ApplyReplacements` produces the exact bytes of `Format(tree)` (this
    invariant holds on erroneous inputs as well).
2.  **Idempotency.** Formatting already-formatted output produces identical text
    and zero `Replacement`s (including embedded C++ snippets, enforced by
    `CppSnippet`'s fixed-point check).
3.  **Token preservation.** Re-lexing formatted output yields the same
    `TokenKind` sequence and preserves non-string token spellings
    (`GetTokenText`).
4.  **Validity preservation.** Error-free input re-lexes and re-parses without
    errors after formatting, catching whitespace-sensitive fixity regressions
    such as splitting unary operators or merging adjacent operator tokens.

Test suites:

-   **Golden file tests** (`toolchain/format/testdata/` under `file_test` with
    `AUTOUPDATE`), verifying formatted output across spacing, wrapping,
    comments, member chains, C++ snippets, error recovery, and `--lines` range
    formatting.
-   **Unit tests** (`formatter_test`), verifying the formatting invariants,
    greedy-fallback and operator-chain edge cases, minimal-edit equivalence,
    `--lines` range couplings, `Style` overrides, error recovery, and
    `CppSnippet` safety guards.
-   **Fuzzing** (`format_fuzzer`), checking crash-freedom and minimal-edit
    equivalence on all inputs (including lex-error and parse-error inputs),
    lex-cleanliness and comment preservation on lex-clean inputs (including
    parse-error inputs), and idempotency, token kind/spelling preservation,
    validity preservation, and `--lines` range-formatting convergence on
    error-free inputs (`format_cpp_snippets` is disabled in the fuzzer because
    `clang-format`'s debug assertions can fire on arbitrary non-C++ bytes).
-   **Benchmarks** (`toolchain/benchmarking/format_benchmark.cpp`), measuring
    both synthetic declaration/implementation workloads (`Format` and
    `FormatReplacements`) and real repository files alongside `clang-format`.

## Worked examples

Examples below use the canonical style (`column_limit = 80`, `indent_width = 2`,
`continuation_indent_width = 4`).

**Overflowing parameter list** (83 columns on one line):

```carbon
fn RegisterHandler(event_name: String, priority: i32,
                   callback: Callback) -> bool {
  return true;
}
```

Bin-packing keeps `event_name` and `priority` on the first line and aligns
`callback` to the column after `(`.

**Overflowing initializer** (84 columns on one line):

```carbon
fn F() {
  var is_valid: bool =
      has_permission and is_authenticated and not is_token_expired;
}
```

Breaking after `=` (`penalty_break_assignment = 2`) lets the entire right-hand
side fit at `+4` continuation indent, beating a break after `and` (penalty 5).

**Mixed-precedence binary expression** (89 columns on one line):

```carbon
fn F() -> i32 {
  return aaaaaaaaaaaaaaaa * bbbbbbbbbbbbbbbb +
         cccccccccccccccc * dddddddddddddddd + eee;
}
```

Breaking after the first `+` (penalty 7, looser than `*` at 8) keeps both `*`
subexpressions intact and aligns `cccccccccccccccc` under `aaaaaaaaaaaaaaaa`.

**Builder call chain with receiver-anchored indent** (98 columns on one line):

```carbon
fn F() {
  var x: auto = aaaaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbb()
                    .cccccccccccccccc()
                    .dddddddddddddddd();
}
```

Breaking after `=` (88 columns) would still overflow, so `x`'s initializer stays
on the `=` line and breaks all-or-nothing before each fluent `.`, aligned to
`receiver_start_column + 4`.

**Struct literals (inline versus trailing comma):**

```carbon
fn F() {
  var origin: {.x: i32, .y: i32} = {.x = 0, .y = 0};
  var expanded: Point = {
    .x = 0,
    .y = 0,
  };
}
```

Without a trailing comma, `StructOpen`/`StructClose` keeps short struct types
and literals inline with `0` interior spaces; adding a trailing comma selects
`MultilineStructOpen` and `BlockComma`, expanding one field per line at `+2`
indent.

**Mid-statement comments in a call:**

```carbon
fn F() {
  Call(aaa,  // first
       bbbb, // second
       ccc);
}
```

Mid-statement trailing comments force line breaks inside the unwrapped line
while preserving open-bracket alignment after `(` and aligning consecutive `//`
markers.

**`match` block expansion.**
`match (color) { case .Red => { return 1; } default => { return 0; } }` formats
to:

```carbon
match (color) {
  case .Red => {
    return 1;
  }
  default => {
    return 0;
  }
}
```

## Future work

-   **Configuration surface.** Command-line flags or config files populating
    `Style`, including non-default `Style` support in `file_test`.
-   **Display and multi-line token widths.** Unicode display-width measurement
    for `TokenInfo::column_width`, plus tracking a multi-line token's final-line
    column (`LastLineColumnWidth` in `clang-format`) for tokens following a
    multi-line string literal.
-   **Call-argument unpacking.** Unpacking all arguments of a call onto separate
    lines when any single argument spans multiple lines.
-   **Trailing-comment wrapping.** Wrapping trailing comments that exceed
    `column_limit`.
-   **Semantics-preserving sorting.** Sorting imports and normalizing
    declaration-modifier order.
-   **Consecutive token alignment.** Aligning consecutive assignments or
    declarations if adopted by the canonical style.
-   **Additional embedded languages.** Dispatching embedded block-string
    formatting by file type indicator beyond C++.

## Alternatives considered

-   **Heuristic token-only front end (`clang-format`).** Reconstructing syntax
    and operator precedence heuristically from tokens duplicates work already
    done by `Parse::Tree` with lower accuracy and substantially more code.
-   **Greedy group-local pretty-printing (Wadler / Prettier
    `group`/`nest`/`line` combinators).** While syntactic structure still comes
    from `Parse::Tree` (such as token roles, operator spans, and block-brace
    boundaries), evaluating each group's break decision greedily in isolation
    cannot weigh competing split penalties across nested groups or optimize
    soft-limit overflow across a whole statement the way shortest-path search
    does.
-   **Whole-file rewrite output only.** Simpler than tracking `TokenSpan`
    anchors, but produces noisy diffs and cannot support `--lines` range
    formatting or incremental LSP `TextEdit`s.

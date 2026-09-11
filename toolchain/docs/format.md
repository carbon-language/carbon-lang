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
    -   [Libraries](#libraries)
-   [Token roles and per-token information](#token-roles-and-per-token-information)
-   [Spacing rules](#spacing-rules)
-   [The line-breaking solver](#the-line-breaking-solver)
    -   [Search states](#search-states)
    -   [The soft column limit](#the-soft-column-limit)
    -   [Continuation indentation and operand alignment](#continuation-indentation-and-operand-alignment)
-   [Penalty model](#penalty-model)
-   [Member-access call chains](#member-access-call-chains)
-   [Comment reflow](#comment-reflow)
-   [Trailing comments](#trailing-comments)
-   [Embedded C++ snippets](#embedded-c-snippets)
-   [The `WhitespaceManager` and minimal edits](#the-whitespacemanager-and-minimal-edits)
-   [Range and incremental formatting](#range-and-incremental-formatting)
-   [Error recovery](#error-recovery)
-   [Canonical style](#canonical-style)
-   [Idempotency and stability](#idempotency-and-stability)
-   [Editor integration](#editor-integration)
-   [Testing](#testing)
-   [Worked examples](#worked-examples)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Overview

`carbon format` is the Carbon toolchain's code formatter, analogous in scope to
`clang-format` for C++: it normalizes whitespace, indentation, line breaking,
wrapping, and comment layout to produce consistent, readable Carbon source.

The design borrows `clang-format`'s _layout model_: the optimizing,
penalty-driven line-break solver and the minimal-whitespace edit model. Its
front end is significantly simpler: rather than a heuristic pseudo-parse, Carbon
can read the needed structures directly off the toolchain's
[parse tree](parse.md).

The implementation is in [`toolchain/format/`](/toolchain/format/).

## Goals and non-goals

Goals:

-   Full line wrapping and breaking under a configurable column limit, with the
    penalty-driven "least-bad" wrapping `clang-format` is known for.
-   Correct indentation and continuation indentation.
-   Comment re-indentation and reflow.
-   Best-effort formatting of invalid or incomplete code (as arises with
    format-on-save), never refusing to produce output.
-   Minimal, stable edits suitable for editor integration, including formatting
    a selected line range and lowering to language-server `TextEdit`s.
-   A single canonical style whose defaults follow `clang-format`'s LLVM style.
-   Formatting the contents of string literals containing a known language,
    especially C++ string literals used for interop with `clang-format` itself.
-   Normalizing order where reordering is expected to preserve semantics, such
    as sorting imports and declaration modifier keywords. These are in scope but
    not yet implemented; see [Future work](#future-work).

Non-goals:

-   Behavior-changing rewrites, such as `clang-format`'s brace insertion.
    Formatting must preserve the program: only whitespace, comments,
    [embedded-snippet](#embedded-c-snippets) re-encoding, and the
    semantics-preserving reordering above may change.
-   `clang-format`'s full configuration surface (hundreds of options, multiple
    named base styles, a discovered config file). The `Style` object holds only
    the knobs the implementation reads; see [Canonical style](#canonical-style).
-   A preprocessor or macro model. Carbon has neither, which removes an entire
    class of `clang-format` complexity (per-`#if`-branch "runs" and macro
    expand/reconstruct passes).
-   Block comments. Carbon has only `//` line comments and no `/* */` form is
    planned, so there is no `BreakableBlockComment` analog.

## Relationship to clang-format

Carbon keeps `clang-format`'s vocabulary where the implementation genuinely
mirrors it, so the design stays legible to `clang-format` authors. But it adopts
the _concepts_ with much simpler data structures, because the parse tree removes
the work those structures existed to do. The table maps `clang-format`
subsystems to how Carbon realizes them:

| clang-format subsystem                             | Carbon                                    | Notes                                                                                                      |
| -------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `FormatTokenLexer` (re-lex + token merging)        | dropped                                   | Carbon lexes once; per-token data is stored by `Lex::TokenIndex`. No max-munch merging.                    |
| `UnwrappedLineParser` (token-scan into lines)      | a token walk in `Formatter`               | Statement/block/bracket boundaries come from token kinds and the tree; no `calculateBraceTypes` lookahead. |
| `TokenAnnotator::AnnotatingParser` (role guessing) | `RoleForNodeKind` (a `switch`)            | Roles are a function of `Parse::NodeKind`; no `determineStarAmpUsage` / `parseAngle` heuristics.           |
| Fake-parenthesis precedence reconstruction         | operand-alignment scopes from the tree    | Operator precedence is the real expression nesting, so scopes are placed exactly rather than inferred.     |
| `MatchingParen` discovery                          | `TokenizedBuffer::GetMatchedClosingToken` | Already recorded by the lexer, so the solver's scope stack is exact.                                       |
| Preprocessor / macro passes                        | dropped                                   | Carbon has neither.                                                                                        |
| `ContinuationIndenter` + `LineState`/`ParenState`  | `SolveLineBreaks` + `State`               | The optimizing shortest-path break solver, in `line_wrapper.cpp`.                                          |
| `WhitespaceManager` (+ alignment)                  | `WhitespaceManager`                       | Per-token whitespace `Change`s, minimal `Replacement`s, and trailing-comment alignment.                    |
| `BreakableToken` (comment/string breaking)         | `CommentText`                             | `//` line-comment re-indent and wrap. No block-comment or string-literal breaking.                         |
| `AffectedRangeManager`                             | `Formatter::AffectedByteRanges`           | Brace-pair expansion for range formatting.                                                                 |
| `FormatStyle`                                      | `Style`                                   | A small curated subset; defaults from the LLVM style.                                                      |

## Architecture

### Pipeline

The formatter consumes the toolchain's existing lex and parse output; it never
re-lexes or re-parses:

```mermaid
flowchart LR
    src([source text]) --> Lex --> TB[TokenizedBuffer]
    TB --> Parse --> Tree
    TB --> F[Formatter]
    Tree --> F
    F --> Out([formatted text<br/>or minimal edits])
```

Formatting is one walk over the parse tree, then one over the token stream:

1.  **Annotate.** Walk the parse tree once, in the `Formatter` constructor,
    recording per-token roles, operand-alignment scopes, break penalties, and
    member-chain identity in the
    [per-token information store](#token-roles-and-per-token-information);
    subtree token ranges, merged on a stack during the walk, supply the operand
    spans and chain roots.
2.  **Build lines.** Walk the token stream, buffering each _unwrapped line_ (the
    `UnwrappedLine` analog: one statement or similar unit, possibly spanning
    several source lines). A `;` ends a line; so do `{` and `}`, except an empty
    `{}` and a `}` continued by a closer, separator, `=`, or `else` (see
    [Spacing rules](#spacing-rules)).
3.  **Solve.** An over-long line goes to `SolveLineBreaks`, the
    [penalty solver](#the-line-breaking-solver).
4.  **Record.** Each token's leading whitespace becomes a `WhitespaceManager`
    `Change`; comment blocks become raw `Change`s.
5.  **Generate.** Run [trailing-comment alignment](#trailing-comments), then
    materialize the text, recording where each token landed.

The result is consumed either as the whole formatted text (`Format`) or as
minimal edits, optionally restricted to a `LineRange` (`FormatReplacements`).
The driver writes the text out; the language server lowers the edits to
`TextEdit`s.

### Libraries

The `toolchain/format/` library family is named after the role each plays, not
one-to-one after `clang-format` classes:

-   `style`: the `Style` parameter object (header only).
-   `token_info`: `TokenRole` and the per-token `TokenInfo`, with the spacing /
    break / penalty functions over them.
-   `comment`: `CommentText`, the `//` comment re-indent and wrap.
-   `cpp_snippet`: `CppSnippet`, the reformatting of an embedded C++ string
    literal with `clang-format`.
-   `line_wrapper`: `SolveLineBreaks`, the optimizing solver.
-   `whitespace_manager`: `WhitespaceManager`, `Change`, and trailing-comment
    alignment.
-   [`format`](/toolchain/format/format.h): the `Formatter`, the public `Format`
    / `FormatReplacements` / `ApplyReplacements` entry points, and the
    `Replacement` / `LineRange` edit model.

The driver subcommand
([`format_subcommand`](/toolchain/driver/format_subcommand.cpp)) and a
language-server handler (`handle_formatting.cpp`) sit on top; see
[Editor integration](#editor-integration).

## Token roles and per-token information

Per-token formatting data lives in a `TokenInfoStore`: a `FixedSizeValueStore`
indexed by `Lex::TokenIndex`. The `Formatter` constructor fills the store in one
postorder walk of the parse tree. Information the tokenized buffer already
provides is read from it rather than copied: kinds come from
`TokenizedBuffer::GetKind`, and token text from `TokenizedBuffer::GetTokenText`,
which preserves literal radixes, string escapes, and raw-identifier `r#`
spellings. Each `TokenInfo` holds what the spacing and wrapping decisions need:

-   `role`: the `TokenRole` (see below).
-   `column_width`: the byte length of the token's text (of just its first
    physical line, for a multi-line token), an approximation of its display
    width (see [Future work](#future-work)).
-   `open_scopes` / `close_scopes`: the count of operand-alignment scopes (the
    analog of `clang-format`'s fake parentheses) that begin and end at this
    token.
-   `break_penalty_after`: if non-negative, the token is an infix operator after
    which a break is the canonical split point. An initializer's `=` is not an
    infix-operator node and has no penalty here; a `SplitPenalty` fallback
    prices it instead.
-   `break_penalty_before`: if non-negative, the token is a member-access
    `.`/`->` before which a break is the canonical split point.
-   `member_chain_id`: for a member-access `.`/`->`, the chain it belongs to
    (its receiver-root token index).

`TokenRole` is the Carbon analog of `clang-format`'s ~250-value `TokenType`
enum, but it currently needs a few states:

-   `Unknown`: no distinguishing role.
-   `PostfixBracket`: a `(` or `[` opening a call, parameter list, or subscript
    after a callee or name, as in `F(x)`.
-   `PrefixOperator`: a prefix operator or leading designator `.`, as in `*p` or
    `case .Red`.
-   `PostfixOperator`: a postfix operator, as in the pointer type `p*`.
-   `MemberAccess`: a member-access `.` or `->`, as in `a.b`.

`RoleForNodeKind` derives the role from the owning `Parse::NodeKind`; the enum
grows as more behavior comes to depend on finer roles (see
[Future work](#future-work)). The spacing each role produces is in
[Spacing rules](#spacing-rules).

The remaining per-token data is precedence and break information.
`OperatorInfoForNodeKind` maps each `InfixOperator*` / `ShortCircuitOperator*`
node to a `{break_penalty, aligns_operands}` pair: the break penalty is the
operator's precedence rank, lowest for the loosest operator, and every operator
aligns its operands except the assignment family, whose broken right-hand side
takes the continuation indent instead. `MemberAccessBreakPenalty` marks
member-access `.`/`->` tokens. This is where the dual `->` is resolved without
heuristics: a `PointerMemberAccessExpr` `->` is a member access, while a
`ReturnType` `->` is not, told apart by the owning node even though they share
the `MinusGreater` token kind.

## Spacing rules

`SpacesBefore(left, right)` is the Carbon analog of `clang-format`'s
`spaceRequiredBefore`, expressed over two adjacent tokens' kinds and roles. The
values are grounded in idiomatic Carbon from `docs/design/` and follow the LLVM
style wherever a C++ construct maps:

| Construct                                                       | Spacing                                                                                                                            |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Binary operators, `=`, compound assignment, `as`                | one space each side                                                                                                                |
| `and` / `or` / `not` (keyword operators)                        | one space each side                                                                                                                |
| `,`                                                             | no space before, one after                                                                                                         |
| `;` (statement terminator)                                      | no space before, line break after                                                                                                  |
| Member access `.`, pointer member `->`                          | tight (`a.b`, `x->y`)                                                                                                              |
| Symbolic prefix operator, postfix operator                      | tight (`*p`, `-x`, `p*`); word operators (`not x`) spaced                                                                          |
| Leading designator `.`                                          | tight to its name, ordinary spacing before (`case .Red`, `{.x = 1}`)                                                               |
| Binding colon `name: Type`                                      | no space before, one after; a phase keyword (`generic`/`template`/`runtime`, as in `template T: type`) takes ordinary word spacing |
| Return arrow `) -> Type`                                        | one space each side                                                                                                                |
| Callable name to `(`, deduced-param `[`, parameterized-type `(` | no space (`F(`, `F[T: type]`, `Wrapper(T: type)`)                                                                                  |
| Inside `(...)`, `[...]`                                         | no padding                                                                                                                         |
| Empty `{}`                                                      | compact, no padding                                                                                                                |
| After a closing `}`                                             | `;`, `,`, `)`, `]`, `=`, or `else` continues the line (`};`, `})`, `} else {`); anything else starts a new one                     |

Vertical whitespace is not in the table: blank lines between unwrapped lines
(and before comment blocks) are kept up to `max_empty_lines_to_keep`, and
dropped at the start of the file, just after a `{`, and just before a `}`.

## The line-breaking solver

A unwrapped line that fits within the column limit is emitted unchanged. This is
both the common case and a fast path that keeps short output byte-stable. A line
that overflows goes to `SolveLineBreaks`, a uniform-cost (Dijkstra)
shortest-path search over layout states, mirroring `clang-format`'s
`ContinuationIndenter`.

### Search states

Each `State` is a prefix of the line laid out, holding: the index of the next
token to place, the current column, a stack of continuation-indent anchors (one
entry per open bracket and per open operand-alignment scope, plus a never-popped
bottom entry for the statement level), and the open member-access chains (see
[Member-access call chains](#member-access-call-chains)). From a state, the
solver enqueues two successors for the next token, keeping it on the line or
breaking before it (where a break is legal), each charged an incremental
penalty. States are deduplicated by a key over the layout-relevant fields, so
equivalent prefixes are explored once; a `MaxStatesGenerated` cap leaves
pathounwrapped lines unwrapped rather than searching on.

Break legality is `clang-format`'s `canBreakBefore` in Carbon terms
(`CanBreakBefore` in `token_info`): a break is legal before any token except a
separator or closing bracket, a `PostfixBracket` `(`/`[` (which stays with its
callee), an `=`, and an infix operator (the style breaks _after_ operators), and
is never legal after a member-access `.`/`->` (a chain breaks before its
members). The penalties then choose among the legal breaks.

### The soft column limit

The column limit is not a hard constraint. Each column past it costs
`penalty_excess_character`, charged as a delta over the furthest column the line
has already paid for, so each excess column is charged exactly once along a
search path. This single mechanism unifies "fit within N columns" with "but
never break absurdly to get there", and avoids the dead ends of a naive hard
limit, which yields worse output and can fail to format at all.

### Continuation indentation and operand alignment

A wrapped token indents to its bracket level's anchor. There are two anchor
kinds, both stored as the column their continuations indent to and maintained by
`ApplyTokenScopes`:

-   A real bracket aligns its contents just after the opener
    (`AlignAfterOpenBracket = Align`).
-   An operand-alignment scope (the fake-parenthesis analog) opens at the first
    token of a binary operator's operand span and aligns the operands under it,
    but never further left than the enclosing anchor, so an operator at the very
    start of a line indents by the continuation width rather than aligning at
    column 0 (`AlignOperands = Align`). Because precedence comes from the tree,
    these scopes are placed exactly: a broken `a + b * c` indents as its tree
    dictates.

Block indentation uses `indent_width`; continuation uses
`continuation_indent_width`.

## Penalty model

Two penalty sources steer the solver, both anchored to the LLVM style. Because
idiomatic Carbon follows C++ formatting conventions, we baseline the penalties
based on `clang-format`'s values. `SplitPenalty` prices each legal break: some
rows read the `Style` fields, the rest are constants in `token_info`.

Style penalties (`Style` fields):

| Field                                       | Value     | clang-format                           |
| ------------------------------------------- | --------- | -------------------------------------- |
| `penalty_excess_character`                  | 1,000,000 | `PenaltyExcessCharacter`               |
| `penalty_break_assignment`                  | 2         | `PenaltyBreakAssignment`               |
| `penalty_break_before_first_call_parameter` | 19        | `PenaltyBreakBeforeFirstCallParameter` |

Per-construct split penalties (`SplitPenalty`), carrying over the integer scale
of `clang-format`'s `splitPenalty`:

| Split point                                              | Penalty                                     | clang-format analog                    |
| -------------------------------------------------------- | ------------------------------------------- | -------------------------------------- |
| before a member access `.`/`->` mid-chain (last link 35) | 150                                         | `isMemberAccess` 150/35                |
| before a return type `-> Type` on its own line           | 60                                          | `PenaltyReturnTypeOnItsOwnLine`        |
| after `(` / `[`, before the first argument               | `penalty_break_before_first_call_parameter` | `PenaltyBreakBeforeFirstCallParameter` |
| after `=`, onto the right-hand side                      | `penalty_break_assignment`                  | `PenaltyBreakAssignment`               |
| after a binary operator, before its right operand        | operator precedence level                   | precedence fallback                    |
| after `,`, between list elements                         | 1                                           | `comma` = 1                            |
| default (unmatched)                                      | 3                                           | default = 3                            |

Further construct-specific penalties are added as the constructs that need them
are formatted; see [Future work](#future-work).

## Member-access call chains

A long member-access chain wraps the way `clang-format` wraps it:

-   A break after a `.`/`->` is never allowed, so a chain wraps _before_ each
    member, keeping `.member()` attached.
-   The break-before penalty is parse-tree-driven, following the dual-`->`
    disambiguation in
    [the per-token information](#token-roles-and-per-token-information).
-   The _last link_ in each chain breaks at the cheaper last-link penalty (see
    [the penalty model](#penalty-model)), so a chain that must wrap prefers to
    break at its end; the `Formatter` reduces that link's penalty once the
    chain's outermost member is known.
-   **Fluent / builder shape.** A chain containing a member access that follows
    a call or subscript (`)`/`]`) is a _builder chain_: it formats
    all-or-nothing, breaking before _every_ such member, or staying packed if it
    fits. The first segment and any plain field accesses stay attached. A chain
    with no such boundary is a plain field chain, which uses the ordinary
    minimum-break with the 35 last link.
-   **Receiver-anchored indent.** Member continuations align under the chain's
    receiver plus the continuation indent (`receiver_col + 4`), not the
    statement continuation indent.

Chain identity is the receiver-root token shared by the left-nested chain
(`TokenInfo::member_chain_id`). The all-or-nothing coupling is enforced in the
solver by a per-chain decision (`ChainState`) carried in the search state, which
also records the anchor captured when the receiver is placed.

## Comment reflow

Carbon has only `//` line comments, and the lexer coalesces consecutive comment
lines with an identical leading prefix (indentation included) into one block
([`GetCommentText`](/toolchain/lex/tokenized_buffer.h) returns the whole span).
`CommentText` (`comment.h`) re-indents every line of the block to the current
code indent (the lexer keeps each comment line's original indentation, which
need not match the surrounding code) and wraps any line past the column limit at
whitespace onto further lines; a line that already fits is kept verbatim. The
comment prefix, `//` plus the whitespace after it, is repeated on every produced
line, so an indented bullet stays a bullet, and the retained text keeps its
interior spacing verbatim.

Two guards preserve intent. A single word too long to fit is left on its own
over-long line rather than broken. A `//` not followed by whitespace (a `//===`
divider, or a lexically invalid comment kept best-effort) is re-indented but
never word-wrapped, which would corrupt it.

A `//@...` tooling directive line (`//@include-in-dumps`,
`//@dump-sem-ir-begin`/`-end`) is recorded by the lexer as a comment alongside
its tooling side effect, so the formatter preserves it like any other comment.
The second guard covers it: a directive is re-indented to the code indent but
never wrapped, which would break its recognition. This is what makes the
toolchain's own file tests formattable without losing information.

A full-line comment inside a statement ends the buffered line at the comment:
the statement's parts before and after it lay out as separate lines at the block
indent.

This is `clang-format`'s `BreakableLineCommentSection` behavior specialized to
Carbon, with one deliberate narrowing: consecutive comment lines are not merged
into reflowed paragraphs (only over-long lines wrap), so deliberate comment
layout such as lists, tables, and examples survives. This lands between LLVM's
`ReflowComments` values of `IndentOnly` and `Always`.

## Trailing comments

A _trailing comment_ shares its line with the code it follows. The lexer
classifies each comment
([`IsTrailingComment`](/toolchain/lex/tokenized_buffer.h)), so the formatter
does not have to guess. A trailing comment is kept on the line of the code it
annotates: the `Formatter` flushes that code line and then records the comment
through `WhitespaceManager::AddTrailingComment`, a raw `Change` with no leading
newline that appends to the line after a single space.

A run of consecutive trailing comments at the same indent is aligned into one
column (`clang-format`'s `alignTrailingComments`), so a block of annotated
declarations or statements lines up. Alignment is a `WhitespaceManager` pass
over the recorded `Change`s (`AlignChanges`), running after all other layout is
decided so the comments align against the final code columns: it partitions the
changes into physical lines, finds each line's trailing comment, and pads a
maximal run of consecutive same-indent lines so their `//`s share a column (the
run's rightmost natural one). A blank line, an indent change, or a line without
a trailing comment breaks the run, except a wrapped statement's continuation
lines, which are skipped without breaking it (mirroring `clang-format`'s
deeper-nesting skip). A lone trailing comment keeps its single space. The
`align_trailing_comments` style knob mirrors the `clang-format` option and is on
by default, following the LLVM style. A trailing comment is never wrapped, so
its line can exceed the column limit (see [Future work](#future-work)).

## Embedded C++ snippets

A multi-line string literal holds a C++ snippet in two cases: its file type
indicator names C++ (`'''cpp`, or any of `cc`/`cxx`/`c++`/`h`/`hpp`/`hxx`/`hh`,
case-insensitive), or it is the body of an `inline Cpp` / `import Cpp inline`
declaration (the parse tree's `InlineImportBody` node). The declaration body is
C++ regardless of its indicator, so an untagged `'''` there is still
reformatted. A trailing comment on the introducer line ends the indicator the
way trailing whitespace does, mirroring the lexer, so `'''cpp // why` is still a
C++ snippet. `CppSnippet` (`cpp_snippet.h`) reformats that snippet's body with
`clang::format::reformat` and re-encodes the literal: the body is de-indented by
the closing `'''`'s indentation, formatted in isolation, then re-indented along
with the closing `'''` to the statement's indent, while the opening
`'''<indicator>` line (comment included) is kept verbatim. A snippet that could
not be re-encoded as a valid, stable literal is left untouched. An unchanged
snippet is not treated as a rewrite, so the literal keeps anchoring
[minimal edits](#the-whitespacemanager-and-minimal-edits). The feature is gated
on the `format_cpp_snippets` style knob, on by default.

`clang-format` runs under Carbon's style, not its own configuration: the base
is the LLVM style, the `Style`'s column limit, indent width, and continuation
indent override `clang-format`'s, and tabs are disabled. Pointer alignment is
pinned left (`T* p`) rather than derived from the snippet, mirroring how
Carbon itself spells pointer types and matching Carbon's own C++ code style.
The body is reformatted to a column limit of `column_limit - indent`, because
it is re-indented by `indent` columns when placed back in the literal, so the
result still respects Carbon's line length.

C++ is the first embedded language, not the only one intended; see
[Future work](#future-work).

This is the one place the formatter rewrites a token's text rather than only the
whitespace around it. The `WhitespaceManager` emits the reformatted literal in
place of the token's source spelling; the
[minimal-edit model](#the-whitespacemanager-and-minimal-edits) treats the
rewritten literal as part of the surrounding gap rather than as an anchor. A
multi-line literal's `column_width` is its first line's width, so the wrapping
solver treats the opening line like any other and never tries to break the
snippet body. Later tokens on that unwrapped line are placed from that
approximate column; in practice little follows a multi-line literal but a
closing `)` or `;`.

## The `WhitespaceManager` and minimal edits

All layout whitespace flows through one channel, the `WhitespaceManager`. The
`Formatter` records one `Change` per token, holding the newlines and spaces
before it plus the brace- and bracket-nesting the alignment pass needs, and a
single `Generate` step runs alignment and emits the text, recording where each
token landed (a `TokenSpan` list, the map the minimal-edit walk below reads).
Carbon's comments are not tokens, so a formatted comment block is recorded as
one raw, verbatim `Change` between tokens; a
[trailing comment](#trailing-comments) is also a raw `Change`. Line breaks are
attributed to the following content, so no content carries its own trailing
newline; `Generate` appends the one final newline that ends a non-empty file.

On top of this sits the minimal-edit model in
[`format.h`](/toolchain/format/format.h). Because formatting only ever changes
whitespace and comments, never the tokens themselves, tokens act as fixed
anchors, and an edit is the differing _gap_ between two consecutive token
anchors. There are two exceptions. A string literal's text may be rewritten when
it holds an [embedded C++ snippet](#embedded-c-snippets); such a rewritten
literal is not an anchor, and its edit folds into the surrounding gap. A
lexer-inserted recovery token's text exists in no source byte range, so it too
is not an anchor and its text flows into the neighboring gap's edit.
`FormatReplacements` returns these as a `Replacement` list
(`{offset, length, text}`, the `tooling::Replacement` analog) ordered by offset
and non-overlapping; `ApplyReplacements(source, replacements)` reproduces what
`Format` would write. The granularity is gap-level (a changed comment block is
one edit): finer than a whole-document replace, coarser than `clang-format`'s
per-token records.

## Range and incremental formatting

Range formatting filters whole-file formatting rather than re-running it.
`FormatReplacements` takes an optional 1-based inclusive `LineRange`,
`Formatter::AffectedByteRanges` lowers those lines to the byte ranges they
affect, and only the edits whose gap intersects those ranges are kept. The kept
edits are exactly the whole-file solution's edits for those lines, which is
sound because unwrapped lines lay out independently, with two structural
couplings the byte-range expansion closes over to a fixed point:

-   A unwrapped line lays out as a unit, so a partially requested one is wholly
    affected rather than re-wrapped in part.
-   A brace whose matching brace is affected becomes affected too, so
    reformatting a `{` line fixes its dangling `}`.

One coupling is deliberately left open:
[trailing-comment alignment](#trailing-comments) is computed on the whole
formatted file, so a range-limited format can pad an in-range comment to the
column its run settles on under whole-file formatting, even when the
out-of-range neighbors have not moved yet; a later whole-file format converges
on the same columns. The other `AffectedRangeManager` cases, such as a line that
"moved" relative to an affected predecessor, do not arise: indentation is
derived from bracket structure, not carried forward from the preceding line. A
partly selected comment block is already whole because it sits within a single
gap. Range formatting drives the driver's `--lines=START:END` flag and the
language-server `rangeFormatting` handler; see
[Editor integration](#editor-integration).

## Error recovery

The parse tree is structurally valid even with errors. The formatter runs
best-effort: well-formed subtrees format normally, and the return value reports
whether the input was error-free (so the driver can set its exit code), but
output is produced either way rather than refusing to format the file. This is
the behavior [the parse design](parse.md) anticipates for tooling that operates
on invalid code while preserving author intent.

Within an erroneous region the parse, and so the formatting decisions built on
it, is unreliable, so the formatter does not reformat it: each maximal error
subtree (an erroneous parse node with no erroneous ancestor) is emitted with its
original source text; that subtree is the smallest span confining the errors.
Its first token is placed normally, at the surrounding code's indent, and every
gap after that within the region, whitespace and comments alike, is copied from
the source. A line holding such a region is never re-wrapped, and under the
[minimal-edit model](#the-whitespacemanager-and-minimal-edits) the unchanged
gaps produce no edits, so format-on-save leaves half-typed broken code untouched
while still cleaning up around it. One gap is the exception: a gap bounded by a
lexer-inserted recovery token, whose synthesized offset is not a real source
position, cannot be sliced out of the source and is formatted normally; the
region's other gaps stay verbatim, and its line is still never re-wrapped.

## Canonical style

Carbon ships one canonical style: a default-constructed `Style`. Its values
follow the LLVM style except where noted. The snake_case rows are `Style`
fields; the remaining rows are fixed behavior, named by the `clang-format`
option each corresponds to.

| Setting                            | Canonical value                                  | Source                                              |
| ---------------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| `column_limit`                     | 80                                               | LLVM                                                |
| `indent_width`                     | 2                                                | LLVM                                                |
| `continuation_indent_width`        | 4                                                | LLVM                                                |
| `max_empty_lines_to_keep`          | 1                                                | LLVM                                                |
| `align_trailing_comments`          | true (one space before)                          | LLVM (`AlignTrailingComments`)                      |
| `format_cpp_snippets`              | true                                             | no analog; Carbon C++ interop                       |
| Penalty fields                     | per [penalty model](#penalty-model)              | LLVM                                                |
| Brace wrapping                     | attach                                           | LLVM                                                |
| Break before binary operators      | none (break after the operator)                  | LLVM                                                |
| Bracket and operand alignment      | align (`AlignAfterOpenBracket`, `AlignOperands`) | LLVM                                                |
| Bin-pack arguments / parameters    | yes                                              | LLVM                                                |
| `match` arm indentation            | one level, like any block statement              | Carbon divergence (`IndentCaseLabels`)              |
| Blank line just after a `{`        | dropped                                          | Carbon divergence (`KeepEmptyLines.AtStartOfBlock`) |
| Multi-line string opener           | stays on its statement's line                    | LLVM (`AlwaysBreakBeforeMultilineStrings`)          |
| Short non-empty bodies on one line | never                                            | Carbon divergence                                   |
| Comment reflow                     | wrap over-long `//` lines, never merge           | Carbon divergence                                   |
| File edges                         | no leading blank lines, one final newline        | Carbon divergence                                   |

The deliberate divergences are short-body expansion,
[comment reflow](#comment-reflow) (between LLVM's `ReflowComments` values),
`match` arm indentation, and the blank lines dropped at block and file edges.
Carbon always expands a non-empty brace-delimited body (function,
control-flow, or `match` arm) onto its own lines, for gofmt-like vertical
uniformity; only an empty `{}` stays compact. See the `match`
[worked example](#worked-examples). A `match` arm indents like any other
statement in a braced block, because in Carbon's grammar that is what it is:
there is no analog of C++'s `case` labels, which the LLVM style leaves
unindented. Blank lines at a block's edges are dropped symmetrically: where
the LLVM style drops a blank line just before a `}` but keeps one just after a
`{`, Carbon drops both, so a body neither begins nor ends with padding. A
non-empty file starts at its first token or comment and ends with exactly one
newline; `clang-format` keeps blank lines at the start of a file.

The `Style` object exists to give the canonical style a single source of truth
and a place for a future configuration surface to write; there is no
command-line or config-file surface yet (see [Future work](#future-work)).

## Idempotency and stability

Formatting is idempotent: formatting already-formatted output is a no-op. The
minimal-edit model, per-line layout independence, and a single canonical style
make this hold, and it is enforced as a test invariant across the corpus
(re-running the formatter produces no `Replacement`s). A second invariant checks
that formatting never changes the token sequence, so valid input stays
semantically identical; the only token whose _text_ can change is a string
literal holding an [embedded C++ snippet](#embedded-c-snippets), and there the
rewrite is itself required to be a fixed point.

## Editor integration

Formatting lowers to language-server edits by way of the minimal-edit model. The
`handle_formatting.cpp` handlers implement `textDocument/formatting` (whole
document) and `textDocument/rangeFormatting` (a selection, converted to a
`LineRange` by `LspRangeToLineRange`), each returning the `Replacement`s as LSP
`TextEdit`s. The driver writes whole-file output in place, to `--output=FILE`,
or to stdout (`--output=-`), and formats a line range with `--lines=START:END`.

## Testing

Testing uses the toolchain's `file_test` harness with `testdata/` cases and the
`AUTOUPDATE` / `CHECK:STDOUT` machinery, covering spacing, wrapping, breaking,
indentation, comments and reflow, member chains, error recovery, and range
formatting. The optimizing solver, the spacing/penalty functions, comment
reflow, and trailing-comment alignment also have direct unit tests. The
[idempotency and token-preservation invariants](#idempotency-and-stability) run
across the whole corpus.

A libFuzzer target (`format_fuzzer`) extends those invariants to arbitrary
inputs: any lex-clean input is checked for crash-freedom, and error-free,
ordinary input (no control characters beyond tab and newline) is also checked
for idempotency, token preservation, and the minimal-edit path reproducing the
whole-text output. Embedded C++ formatting is disabled under the fuzzer, since
`clang-format`'s own debug assertions fire on arbitrary non-C++ text; it is
covered by `cpp_snippet_test` instead. A benchmark
(`toolchain/benchmarking/format_benchmark.cpp`) compares end-to-end formatting
cost against `clang-format` on equivalent generated sources.

## Worked examples

These show idiomatic Carbon at the canonical `column_limit = 80`,
`indent_width = 2`, `continuation_indent_width = 4`.

**Parameter list that overflows** (the one-line form is 83 columns):

```carbon
fn RegisterHandler(event_name: String, priority: i32,
                   callback: Callback) -> bool {
  // ...
}
```

Bin-packing packs as many parameters as fit and wraps the rest;
`AlignAfterOpenBracket = Align` keeps the first parameter on the opening line
and aligns the continuation under it (the column just after `(`).

**Assignment whose right-hand side overflows** (the one-line form is 82
columns):

```carbon
var is_valid: bool =
    has_permission and is_authenticated and not is_token_expired;
```

The solver weighs the cheap assignment break (`penalty_break_assignment`, and
the right-hand side then fits on one `+4` continuation line) against breaking
after `and` (precedence-rank penalty 5). The assignment break wins.

**Mixed-precedence expression, showing operand alignment** (the one-line form is
89 columns):

```carbon
fn F() -> i32 {
  return aaaaaaaaaaaaaaaa * bbbbbbbbbbbbbbbb +
         cccccccccccccccc * dddddddddddddddd + eee;
}
```

The break lands after the loosest operator (`+`, precedence-rank penalty 13,
over `*` at 14), and the continuation aligns under the first operand: the
operand-alignment scope opened at `aaaaaaaaaaaaaaaa` anchors the wrapped
operands exactly where the expression tree dictates.

**Builder chain too long for one line even at statement indent:**

```carbon
some_object_receiver_x.method_one_call()
    .method_two_call()
    .method_three_call_xy()
    .mfour();
```

The receiver and first call stay attached and every later member breaks,
all-or-nothing, per [Member-access call chains](#member-access-call-chains).

**`match`, showing total block expansion.**
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

These pieces are designed for but not yet built:

-   **A configuration surface.** A command-line flag and/or config file exposing
    the `Style` knobs (and letting `file_test` exercise non-default styles).
-   **Display-width-aware token widths.** Use true display width where
    [`column_width`](#token-roles-and-per-token-information) is byte length
    today.
-   **Richer `TokenRole`s and per-construct penalties.** The enum and the split
    penalties grow as more constructs (declarations, binding colons, `match`
    arrows, `where` clauses) come to need distinct handling, following
    `clang-format`'s tables.
-   **Semantics-preserving sorting.** Import sorting and declaration-modifier
    ordering, the reordering scoped in
    [Goals and non-goals](#goals-and-non-goals).
-   **Consecutive assignment and declaration alignment.** `clang-format`'s
    off-by-default `AlignConsecutiveAssignments` /
    `AlignConsecutiveDeclarations` passes (for Carbon, the type after a binding
    `:` would align, since the name precedes the type). The trailing-comment
    `AlignChanges` engine is written to generalize to these; add them only if
    the canonical style turns them on.
-   **Braced value literals.** A struct literal's braces are still laid out as a
    block, one line per brace (the parse tree's `StructLiteralStart` can
    distinguish them); keeping a short `{.x = 0, .y = 0}` inline is future work.
-   **`LineJoiner`.** Merging adjacent unwrapped lines onto one physical line.
    Beyond the empty-`{}` case handled today, this only becomes relevant if the
    canonical style ever allows short bodies on one line, which it deliberately
    does not (see [Canonical style](#canonical-style)).
-   **Bin-packing refinement.** When one call argument spans multiple lines (as
    a wrapped chain argument does), `clang-format` puts the remaining arguments
    each on their own line; Carbon currently keeps them packed.
-   **Paren-anchor demotion.** A break after an open bracket currently lands at
    the bracket's alignment anchor, the same column as not breaking, so
    `penalty_break_before_first_call_parameter` never selects a break.
    `clang-format` instead places the elements at continuation indent when that
    break is taken, which can rescue a bracket that opens near the column
    limit.
-   **Trailing-comment wrapping.** Wrap the over-long trailing comments that
    [Trailing comments](#trailing-comments) today leaves past the column limit.
-   **Pluggable embedded-language formatting.** The
    [embedded C++ support](#embedded-c-snippets) is expected to generalize to a
    formatter plugged in per known file type indicator, with C++ only the first
    (and most important) instance.
-   **Solver allocation.** The settled-state set allocates a key vector per
    state; a hashed `DenseSet` key would avoid it if the solver ever needs to be
    faster.

## Alternatives considered

-   **Token-only, like `clang-format`.** Rejected for the reason the
    [Overview](#overview) gives: reconstructing structure heuristically when a
    real parse tree exists would be more code and less accurate. The token
    stream is still used for text and trivia, but not as the source of
    structure.

-   **An algebraic pretty-printer (Wadler / Prettier `group`/`nest`/`line`
    combinators).** A coherent alternative and a clean fit for a tree, but its
    greedy, group-local breaking model is weaker at bin-packing, column
    alignment, and penalty-driven wrapping, which are the `clang-format`-grade
    behaviors in scope. Adopting `clang-format`'s optimizing solver keeps both
    the behavior and the vocabulary its authors share.

-   **Whole-file rewrite output.** Simpler than minimal edits, but produces
    noisy diffs and supports neither range formatting nor clean editor
    integration.

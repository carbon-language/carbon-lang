---
name: Language server
description:
    Instructions for working on Carbon's LSP language server, including its
    architecture, its file_test-based tests, and the VS Code extension.
---

# Language server

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Introduction

This skill covers [`toolchain/language_server/`](/toolchain/language_server/),
which implements `carbon language-server`, and
[`utils/vscode/`](/utils/vscode/), the VS Code extension that launches it.

## Architecture

The server is built on clangd's LSP transport (`clang::clangd`), not on a
Carbon-specific one. That means clangd's `Protocol.h` types (`Position`,
`Range`, `Location`, `Hover`, `MarkupContent`) are the interface currency.

-   `server.cpp` / `incoming_messages.cpp`: message dispatch. A handler must be
    registered in `incoming_messages.cpp` before it can be called.
-   `handle_*.cpp`: one file per request family, each declaring its entry point
    in `handle.h`.
-   `handle_initialize.cpp`: the advertised capabilities. **Adding a capability
    changes `Content-Length` in every test that calls `initialize`**, so expect
    a large autoupdate diff.
-   `context.h` / `context.cpp`: `Context::File` per open document, plus the
    compile driver. `Context::File::unit()` has a `CARBON_CHECK` on the compile
    driver, so any handler that reaches for the parse tree must first rule out
    documents that were never compiled.
-   `position.h`, `sem_ir_index.h`: mapping source positions to SemIR
    instructions for real Carbon files.
-   `sem_ir_text.h`, `handle_sem_ir_text.h`: navigation within the *formatted
    SemIR* in a test file's `// CHECK:STDOUT:` lines. This is a heuristic text
    index, deliberately independent of the real SemIR data structures. See
    [the SemIR text reader](#the-semir-text-reader).

### Document kinds

The server handles two kinds of document, distinguished by the `languageId`
from `textDocument/didOpen`, with a content sniff as a fallback:

-   `carbon`: a real Carbon file. Compiled; diagnostics published.
-   `carbon-testdata`: a test file. **Not compiled**, because we lack logic to
    split it into one file per `// ---` split marker.

> [!IMPORTANT]
> A handler that assumes every file was compiled will crash on a test file.
> When adding one, give the test-file path an explicit early return.

## Tests

There is **no language-server-specific test target**.
`toolchain/language_server/BUILD` only declares
`filegroup(name = "testdata")`, which is pulled into
`//toolchain/testing:all_testdata` and run by `//toolchain/testing:file_test`.

```bash
# Run just the language server tests (or any subset).
bazelisk test //toolchain/testing:file_test \
    --test_arg=--file_tests=toolchain/language_server/testdata/position/hover_and_goto.carbon

# See the raw output, which is much easier to read than a test failure.
bazelisk run //toolchain/testing:file_test -- --dump_output \
    --file_tests=toolchain/language_server/testdata/position/hover_and_goto.carbon

# Update expectations. Never hand-write CHECK lines.
./toolchain/autoupdate_testdata.py toolchain/language_server/testdata/...
```

These tests run serially, because clangd's logging is a global singleton.

### Test file shape

The request stream is a `// --- STDIN` split written with the `[[@LSP-*]]`
keywords, and the responses land in a trailing `// --- AUTOUPDATE-SPLIT`.
Documents come from other splits by way of `"text": "FROM_FILE_SPLIT"`, which
is substituted with the content of the split whose name matches the `uri`.

```carbon
// --- position.carbon
fn Abs(n: i32) -> i32 { return n; }

// --- STDIN
[[@LSP-CALL:initialize:"capabilities": {}]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/position.carbon",
    "languageId": "carbon",
    "text": "FROM_FILE_SPLIT"
  }
]]
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/position.carbon"},
  "position": {"line": 0, "character": 3}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT
```

Full keyword documentation is in
[`testing/file_test/README.md`](/testing/file_test/README.md).

### Traps

> [!WARNING]
> **A blank line inside the `STDIN` split breaks the JSON transport.** It
> terminates a header block, so clangd logs a timestamped
> `Warning: Missing Content-Length header, or zero-length message.` The
> timestamp makes the test unreproducible, so it fails on the next run.
> Comment lines between messages are fine; blank lines are not. A single blank
> line immediately before `// --- AUTOUPDATE-SPLIT` is also fine.

Other things worth knowing:

-   **`positionEncoding` is UTF-16.** A `character` is a UTF-16 code unit
    offset, not a byte offset.
-   **Line and character numbers in requests are 0-based**, while the `locN_M`
    suffixes in SemIR output are 1-based. Off-by-ones here are silent: the
    request succeeds and returns the wrong thing.
-   **A split can hold a document that itself contains `// CHECK:STDOUT:`
    lines**, because `CHECK` lines only form expectations inside the
    `AUTOUPDATE-SPLIT`. Such a document still can't contain a literal `// ---`
    line, which would split the enclosing test file; write it as
    `[[@0x2f]]/ --- name.carbon`.

## The SemIR text reader

`sem_ir_text.cpp` indexes the formatted SemIR inside a test file's
`// CHECK:STDOUT:` lines so that hover and go-to-definition work on operand
names. It is a heuristic reader, not a parser, and its correctness rests on
facts about `toolchain/sem_ir/formatter.cpp` and
`toolchain/sem_ir/inst_namer.cpp`. Re-check these if the formatter changes:

-   There are exactly four scope keywords: `file`, `generated`, `imports`, and
    `constants` (`InstNamer::GetScopeName`). Everything else is `@entityname`.
-   A reference is `%name` within its own scope and `scope.%name` otherwise
    (`InstNamer::GetNameFor`). Names may contain `.` and may _start_ with one,
    as in `%.Self.frozen`.
-   **Type annotations are printed in the `constants` scope.**
    `Formatter::FormatTypeOfInst` does
    `llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::Constants)`,
    so in `%x: %foo = ...` a bare `%foo` means `constants.%foo`. This does not
    apply to ordinary operands or to `[concrete = ...]` annotations.
-   **`*_decl` braces hold the declared entity's scope.** The braces of
    `%F.decl: ... = fn_decl @F [...] { ... } { ... }` are lexically inside
    `file { }`, but their names belong to `@F`.
-   **`!with Self:` switches scope without a brace**, until `!members:`
    switches it back. Brace counting alone cannot see this.
-   A `specific @F(args) { }` block uses `@F`'s scope and defines nothing; each
    `%name => value` row references an instruction of the generic.

### Validating a change to the reader

The unit tests only cover a handful of cases. To check a change against the
real corpus, drive the server over a sample of check testdata, hovering on
every `%name`, and compare the resolved fraction before and after. Roughly 99%
of names resolve; the residue are names the formatter references but never
emits a definition line for, such as `%I.WithSelf.F`, which only ever appears
inside a `[symbolic = ...]` annotation.

Two things to get right in such a harness:

-   Feed the request stream from a **file**, not a pipe. The server reads stdin
    as a file and reports `error: Input/output error` on a pipe.
-   Read the output as **bytes**. Python's `text=True` rewrites the `\r\n`
    framing, and `Content-Length` counts bytes.

## VS Code extension

[`utils/vscode/`](/utils/vscode/) declares three languages in `package.json`:

| Language id      | Applies to                        |
| ---------------- | --------------------------------- |
| `carbon`         | `*.carbon`                        |
| `carbon-testdata`| `**/testdata/**/*.carbon`         |
| `semir`          | `*.semir`                         |

> [!NOTE]
> The TextMate _scope_ for SemIR is `source.carbon-semir`, but the
> _language id_ is `semir`. Markdown code fences in hover text resolve language
> ids, so a fence must say ` ```semir `.

`extension.ts` launches the server over stdio, using the `carbonPath` setting
(default `./bazel-bin/toolchain/carbon`). Its `documentSelector` controls which
files are sent to the server at all; a new document kind has to be added there
as well as in the server.

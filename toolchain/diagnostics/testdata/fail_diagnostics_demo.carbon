// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// ARGS: --include-diagnostic-kind --diagnostic-snippets --terminal-unicode=always compile --phase=check %s
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/diagnostics/testdata/fail_diagnostics_demo.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/diagnostics/testdata/fail_diagnostics_demo.carbon

// A tour of how diagnostics are drawn. None of the code below is meant to
// compile: every line is here to pull a different part of the rendering into
// the output. See /toolchain/docs/diagnostics_rendering.md for what each part
// is and why it looks the way it does.
//
// This is an ordinary Carbon file, so it runs outside the test suite too:
//
//   carbon compile --include-diagnostic-kind --terminal-unicode=always \
//       --phase=check toolchain/diagnostics/testdata/fail_diagnostics_demo.carbon
//
// The `CHECK` lines below are that output, captured, so the tour reads without
// running anything. With no terminal to measure, layout targets the width code
// is formatted to, so the fitting behaviors appear here too: a long source
// line windowed to fit, a label line slid left inside its range, and a label
// wrapped inside its frame are all below. What can't appear is anything that
// needs a real terminal's width -- an out-dented label needs a narrower
// terminal than the fixed target; `renderer_test` covers it. A diagnostic
// reaching into a second file, with the import path drawn above it, needs a
// library to import and so isn't here either.
//
// What each line is for:
//
// - `Blend(1)`: a label elsewhere in the same file, shown with the error in
//   one view of the file, with the lines between them elided.
// - `Blend(0, "...")`: a context leading the diagnostic, so the message is read
//   against the code as a label. The underline is measured in columns rather
//   than bytes.
// - `Cpp.Paint(0)`: a diagnostic from Clang, drawn here rather than by Clang.
//   It is located at the callee the message names, as Clang does, and its note
//   becomes two labels: the candidate considered, on its name, and what was
//   wrong with it, on the parameters the count it reports came from.
// - `missing`: the message placed under its own range with nothing else on
//   the line, which is the common case. The warning below it has the same
//   shape on the binding it initializes.
// - `count * flag`: the message placed under the whole expression it reports,
//   which is a binary operator and both its operands. The other two errors on
//   this line are separate diagnostics, so each is drawn in its own frame,
//   which shows the line again.
// - `cell[n]`: the same shape for an index, with the message under the object
//   and the index together.
// - `-flag`: the same shape for a prefix operator, with the message under the
//   operator and its operand.

library "demo";

import Cpp inline '''
  void Paint(int layer, int depth);
''';

fn Blend(first_operand: i32, second_operand: Core.Optional(i32));

fn Tour(count: i32, flag: bool, cell: {.a: i32}, n: i32) -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:     64 │ fn Blend(first_operand: i32, second_operand: Core.Optional(i32));
  // CHECK:STDERR:        · ────────────────────────────────┬────────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ calling function declared here
  // CHECK:STDERR:        ·                                    │ [InCallToEntity]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->  79 │   Blend(1);
  // CHECK:STDERR:        ·   ━━━━┳━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: 1 argument passed to function expecting 2 arguments
  // CHECK:STDERR:        ·          ┃        [CallArgCountMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Blend(1);
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+16]]:12
  // CHECK:STDERR:     64 │ fn Blend(first_operand: i32, second_operand: Core.Optional(i32));
  // CHECK:STDERR:        ·                              ───────────────┬──────────────────
  // CHECK:STDERR:        ·                                             ╰──┤ initializing function parameter
  // CHECK:STDERR:        ·                                                │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->  96 │   Blend(0, "全角文字の幅も列で数えるので、下線はずれず、長すぎる行は両端を...
  // CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·            ┗━━┫ error: cannot implicitly convert expression of type `str` to
  // CHECK:STDERR:        ·            │  ┃        `Core.Optional(i32 as Core.OptionalStorage)`
  // CHECK:STDERR:        ·            │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·            ╰────┤ type `str` does not implement interface
  // CHECK:STDERR:        ·                 │ `Core.ImplicitAs(Core.Optional(i32 as Core.OptionalStorage))`
  // CHECK:STDERR:        ·                 │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Blend(0, "全角文字の幅も列で数えるので、下線はずれず、長すぎる行は両端を切り詰めて窓のように表示されます");
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:     61 │   void Paint(int layer, int depth);
  // CHECK:STDERR:        ·        ┬     ──────────┬─────────
  // CHECK:STDERR:        ·        ╰──────────────────┤ candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR:        ·                        ╰────┤ requires 2 arguments, but 1 was provided
  // CHECK:STDERR:        ·                             │ [CppInteropParseNoteReason]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: -> 110 │   Cpp.Paint(0);
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: no matching function for call to 'Paint'
  // CHECK:STDERR:        ·          ┃        [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.Paint(0);
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+6]]:19
  // CHECK:STDERR: -> 117 │   var tint: i32 = missing;
  // CHECK:STDERR:        ·                   ━━━┳━━━
  // CHECK:STDERR:        ·                      ┗━━┫ error: name `missing` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var tint: i32 = missing;
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+32]]:10
  // CHECK:STDERR: -> 150 │   return count * flag + cell[n] + -flag;
  // CHECK:STDERR:        ·          ┳━━━━━━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot access member of interface `Core.MulWith(bool)` in
  // CHECK:STDERR:        ·             ┃        type `i32` that does not implement that interface
  // CHECK:STDERR:        ·             ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+24]]:25
  // CHECK:STDERR: -> 150 │   return count * flag + cell[n] + -flag;
  // CHECK:STDERR:        ·                         ┳━━━━━━
  // CHECK:STDERR:        ·                         ┗━━┫ error: cannot access member of interface
  // CHECK:STDERR:        ·                            ┃        `Core.IndexWith(i32)` in type `{.a: i32}`
  // CHECK:STDERR:        ·                            ┃        that does not implement that interface
  // CHECK:STDERR:        ·                            ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE+15]]:35
  // CHECK:STDERR: -> 150 │   return count * flag + cell[n] + -flag;
  // CHECK:STDERR:        ·                                   ┳━━━━
  // CHECK:STDERR:        ·                                   ┗━━┫ error: cannot access member of interface
  // CHECK:STDERR:        ·                                      ┃        `Core.Negate` in type `bool` that
  // CHECK:STDERR:        ·                                      ┃        does not implement that interface
  // CHECK:STDERR:        ·                                      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_diagnostics_demo.carbon:[[@LINE-27]]:7
  // CHECK:STDERR: -> 117 │   var tint: i32 = missing;
  // CHECK:STDERR:        ·       ━━━━┳━━━━
  // CHECK:STDERR:        ·           ┗━━┫ warning: binding `tint` unused [UnusedBinding]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return count * flag + cell[n] + -flag;
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ARGS: --include-diagnostic-kind --diagnostic-snippets --terminal-unicode=always compile --phase=parse %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/diagnostics/testdata/fail_recovery_demo.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/diagnostics/testdata/fail_recovery_demo.carbon

// A tour of how a missing token is drawn, which is the case where nothing in
// the source is wrong: what the reader needs is not a range to look at but a
// place to put something. See /toolchain/docs/diagnostics_rendering.md for
// the rendering and /toolchain/docs/diagnostics.md for the convention.
//
// The rule is that the message is located at the token the missing one goes
// after, which is the last token the developer wrote. The token after it can be
// the end of the file, which has no source to locate the message at. For an
// opening delimiter, the message is located at the introducer that requires it,
// which the message already names.
//
// This is a parse-phase tour, so nothing here reaches checking. What each line
// is for:
//
// - `fn Paint`: a separator missing between two elements of a list, located at
//   the element it goes after.
// - `if true`: an opening delimiter missing after its introducer, located at
//   the introducer the message names.
// - `var n`: the lexer inserting a `)` to recover, which the label reports,
//   because everything reported after that point is about that reading of the
//   file rather than about what was written.
// - `class Widget`: a `;` missing at the end of a declaration, located at the
//   last token of the declaration. This is last in the file because recovery
//   skips to what looks like the end of the declaration, which consumes
//   everything after it. It is also the case where the next token is the end
//   of the file.

library "recovery_demo";

// CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+6]]:13
// CHECK:STDERR: ->  49 │ fn Paint(a: i32 b: i32);
// CHECK:STDERR:        ·             ━┳━
// CHECK:STDERR:        ·              ┗━━┫ error: expected `,` or `)` [UnexpectedTokenAfterListElement]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Paint(a: i32 b: i32);

fn Blend() {
  // CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->  58 │   if true { }
  // CHECK:STDERR:        ·   ━┳
  // CHECK:STDERR:        ·    ┗━━┫ error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  if true { }
}

fn Tint() {
  // CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+9]]:16
  // CHECK:STDERR: ->  71 │   var n: i32 = (1;
  // CHECK:STDERR:        ·                ┳ ┬
  // CHECK:STDERR:        ·                ┗━━━━┫ error: opening symbol without a corresponding closing
  // CHECK:STDERR:        ·                  │  ┃        symbol
  // CHECK:STDERR:        ·                  │  ┃        [UnmatchedOpening]
  // CHECK:STDERR:        ·                  ╰────┤ possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var n: i32 = (1;
}

// CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+8]]:7
// CHECK:STDERR: ->  82 │ class Widget
// CHECK:STDERR:        ·       ┳━━━━━
// CHECK:STDERR:        ·       ┗━━┫ error: `class` declarations must either end with a `;` or have a
// CHECK:STDERR:        ·          ┃        `{ ... }` block for a definition
// CHECK:STDERR:        ·          ┃        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class Widget

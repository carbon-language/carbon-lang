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
// The rule is that the message is located at the point where the missing
// token would be inserted, which covers no text and is drawn as a lone `┳`. A
// token that goes after another is inserted just past it, even when the next
// token is on a later line or is the end of the file. An opening delimiter goes
// against the token after it when that token is on the same line.
//
// This is a parse-phase tour, so nothing here reaches checking. What each line
// is for:
//
// - `fn Paint`: a separator missing between two elements of a list, located
//   just past the element it goes after.
// - `if true`: an opening delimiter missing after its introducer, located just
//   before the token it goes against.
// - `var n`: the lexer inserting a `)` to recover, which the label reports,
//   because everything reported after that point is about that reading of the
//   file rather than about what was written.
// - `class Widget`: a `;` missing at the end of a declaration, located just
//   past the last token of the declaration. This is last in the file because
//   recovery skips to what looks like the end of the declaration, which
//   consumes everything after it. It is also the case where the next token is
//   the end of the file.

library "recovery_demo";

// CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+6]]:16
// CHECK:STDERR: ->  49 │ fn Paint(a: i32 b: i32);
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: expected `,` or `)` [UnexpectedTokenAfterListElement]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Paint(a: i32 b: i32);

fn Blend() {
  // CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+6]]:6
  // CHECK:STDERR: ->  58 │   if true { }
  // CHECK:STDERR:        ·      ┳
  // CHECK:STDERR:        ·      ┗━━┫ error: expected `(` after `if` [ExpectedParenAfter]
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

// CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+8]]:13
// CHECK:STDERR: ->  82 │ class Widget
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ error: `class` declarations must either end with a `;` or have
// CHECK:STDERR:        ·                ┃        a `{ ... }` block for a definition
// CHECK:STDERR:        ·                ┃        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class Widget

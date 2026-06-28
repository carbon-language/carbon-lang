// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/trailing_comments.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/trailing_comments.carbon

// --- test.carbon

// A trailing comment stays on the line of the code it annotates.
var count: i32 = 0; // A local variable.

// Trailing comments may follow a sequence of statements.
var x: i32 = 0; // first
var yy: i32 = 10; // second
var zzz: i32 = 100; // third

fn Render(frame: Frame) {
  Draw(frame); // A function call,
  Flush(); // another call,
} // and a closing brace.

var a: i32 = 1; // alpha

var bb: i32 = 22; // beta

// With several statements on one source line, a trailing comment attaches to
// the statement it follows -- the last one -- not the first to be rendered.
fn Several() { var s: i32 = 1; var t: i32 = 2; // both split, comment on `t`
}

// A trailing comment in the middle of a statement flushes the line so far and
// stays attached to it; the statement continues on the next line.
fn Mid() {
  Call(aaa, // a mid-statement trailing comment
       bbb);
}

// A trailing comment on a block string literal's introducer line is carried
// within the literal token and is preserved, with or without a file type
// indicator. A trailing comment after the literal's closing line stays
// attached to that line.
var query: String = '''sql // TODO: switch to a prepared statement
  SELECT * FROM t
  ''';

var poem: String = ''' // No file type indicator, just a comment.
  Roses are red.
  '''; // A trailing comment after the closing line.

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A trailing comment stays on the line of the code it annotates.
// CHECK:STDOUT: var count: i32 = 0; // A local variable.
// CHECK:STDOUT:
// CHECK:STDOUT: // Trailing comments may follow a sequence of statements.
// CHECK:STDOUT: var x: i32 = 0; // first
// CHECK:STDOUT: var yy: i32 = 10; // second
// CHECK:STDOUT: var zzz: i32 = 100; // third
// CHECK:STDOUT:
// CHECK:STDOUT: fn Render(frame: Frame) {
// CHECK:STDOUT:   Draw(frame); // A function call,
// CHECK:STDOUT:   Flush(); // another call,
// CHECK:STDOUT: } // and a closing brace.
// CHECK:STDOUT:
// CHECK:STDOUT: var a: i32 = 1; // alpha
// CHECK:STDOUT:
// CHECK:STDOUT: var bb: i32 = 22; // beta
// CHECK:STDOUT:
// CHECK:STDOUT: // With several statements on one source line, a trailing comment attaches to
// CHECK:STDOUT: // the statement it follows -- the last one -- not the first to be rendered.
// CHECK:STDOUT: fn Several() {
// CHECK:STDOUT:   var s: i32 = 1;
// CHECK:STDOUT:   var t: i32 = 2; // both split, comment on `t`
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // A trailing comment in the middle of a statement flushes the line so far and
// CHECK:STDOUT: // stays attached to it; the statement continues on the next line.
// CHECK:STDOUT: fn Mid() {
// CHECK:STDOUT:   Call(aaa, // a mid-statement trailing comment
// CHECK:STDOUT:   bbb);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // A trailing comment on a block string literal's introducer line is carried
// CHECK:STDOUT: // within the literal token and is preserved, with or without a file type
// CHECK:STDOUT: // indicator. A trailing comment after the literal's closing line stays
// CHECK:STDOUT: // attached to that line.
// CHECK:STDOUT: var query: String = '''sql // TODO: switch to a prepared statement
// CHECK:STDOUT:   SELECT * FROM t
// CHECK:STDOUT:   ''';
// CHECK:STDOUT:
// CHECK:STDOUT: var poem: String = ''' // No file type indicator, just a comment.
// CHECK:STDOUT:   Roses are red.
// CHECK:STDOUT:   '''; // A trailing comment after the closing line.

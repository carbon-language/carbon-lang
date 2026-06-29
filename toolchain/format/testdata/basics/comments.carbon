// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/comments.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/comments.carbon

// --- test.carbon

// A comment
fn F() {}

// Another comment

  // Block
  // comment


class C {
    // Internal comment
}


  // Another
  // Block
  //
  // Comment

// A trailing comment may follow a variable, a function call, or a closing
// brace, and the formatter keeps it on the line it annotates.
var count: i32 = 0;          // a) A local variable,

fn Render(frame: Frame) {
  Draw(frame);               // b) A function call,
  Flush();
}                            // c) And a closing brace.

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A comment
// CHECK:STDOUT: fn F () {}
// CHECK:STDOUT: // Another comment
// CHECK:STDOUT: // Block
// CHECK:STDOUT:   // comment
// CHECK:STDOUT: class C {
// CHECK:STDOUT:   // Internal comment
// CHECK:STDOUT: }
// CHECK:STDOUT: // Another
// CHECK:STDOUT:   // Block
// CHECK:STDOUT:   //
// CHECK:STDOUT:   // Comment
// CHECK:STDOUT: // A trailing comment may follow a variable, a function call, or a closing
// CHECK:STDOUT: // brace, and the formatter keeps it on the line it annotates.
// CHECK:STDOUT: var count: i32 = 0; // a) A local variable,
// CHECK:STDOUT: fn Render (frame: Frame) {
// CHECK:STDOUT:   Draw (frame); // b) A function call,
// CHECK:STDOUT:   Flush ();
// CHECK:STDOUT: } // c) And a closing brace.

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
  class D {
            // Deeply mis-indented comment
    var z: i32 = 3;
  }
}

fn IndentedComments() {
        // Over-indented comment.
  var x: i32 = 1;
// Under-indented comment block,
// spanning two lines.
  var y: i32 = 2;
}


  // Another
  // Block
  //
  // Comment

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A comment
// CHECK:STDOUT: fn F() {}
// CHECK:STDOUT:
// CHECK:STDOUT: // Another comment
// CHECK:STDOUT:
// CHECK:STDOUT: // Block
// CHECK:STDOUT: // comment
// CHECK:STDOUT:
// CHECK:STDOUT: class C {
// CHECK:STDOUT:   // Internal comment
// CHECK:STDOUT:   class D {
// CHECK:STDOUT:     // Deeply mis-indented comment
// CHECK:STDOUT:     var z: i32 = 3;
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn IndentedComments() {
// CHECK:STDOUT:   // Over-indented comment.
// CHECK:STDOUT:   var x: i32 = 1;
// CHECK:STDOUT:   // Under-indented comment block,
// CHECK:STDOUT:   // spanning two lines.
// CHECK:STDOUT:   var y: i32 = 2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // Another
// CHECK:STDOUT: // Block
// CHECK:STDOUT: //
// CHECK:STDOUT: // Comment

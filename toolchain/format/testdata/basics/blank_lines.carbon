// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/blank_lines.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/blank_lines.carbon

// --- test.carbon

fn F() {



  var x: i32 = 1;


  var y: i32 = 2;

}



fn G() {}

// Multiple statements or declarations on one source line each get their own
// line, with no blank line invented between them.
fn H() { var a: i32 = 1; var b: i32 = 2; } fn I() {}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var x: i32 = 1;
// CHECK:STDOUT:
// CHECK:STDOUT:   var y: i32 = 2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() {}
// CHECK:STDOUT:
// CHECK:STDOUT: // Multiple statements or declarations on one source line each get their own
// CHECK:STDOUT: // line, with no blank line invented between them.
// CHECK:STDOUT: fn H() {
// CHECK:STDOUT:   var a: i32 = 1;
// CHECK:STDOUT:   var b: i32 = 2;
// CHECK:STDOUT: }
// CHECK:STDOUT: fn I() {}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/braces.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/braces.carbon

// --- test.carbon

fn F() {

}

fn G() -> i32 {

return 3;

}


fn H(x: i32, y: i32) -> i32 {

var z: i32 = x + y;
return z;

}

class C {
    class D {
        class E {}
    }
}

// `else` (and `else if`) continues the close-brace line rather than starting
// its own.
fn Sign(x: i32) -> i32 {
  if (x > 0) { return 1; }
  else if (x < 0) { return -1; }
  else { return 0; }
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() -> i32 {
// CHECK:STDOUT:   return 3;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn H(x: i32, y: i32) -> i32 {
// CHECK:STDOUT:   var z: i32 = x + y;
// CHECK:STDOUT:   return z;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class C {
// CHECK:STDOUT:   class D {
// CHECK:STDOUT:     class E {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // `else` (and `else if`) continues the close-brace line rather than starting
// CHECK:STDOUT: // its own.
// CHECK:STDOUT: fn Sign(x: i32) -> i32 {
// CHECK:STDOUT:   if (x > 0) {
// CHECK:STDOUT:     return 1;
// CHECK:STDOUT:   } else if (x < 0) {
// CHECK:STDOUT:     return -1;
// CHECK:STDOUT:   } else {
// CHECK:STDOUT:     return 0;
// CHECK:STDOUT:   }
// CHECK:STDOUT: }

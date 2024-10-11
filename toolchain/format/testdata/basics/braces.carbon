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

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F () {}
// CHECK:STDOUT: fn G () -> i32 {
// CHECK:STDOUT:   return 3;
// CHECK:STDOUT: }
// CHECK:STDOUT: fn H (x: i32, y: i32) -> i32 {
// CHECK:STDOUT:   var z: i32 = x + y;
// CHECK:STDOUT:   return z;
// CHECK:STDOUT: }
// CHECK:STDOUT: class C {
// CHECK:STDOUT:   class D {
// CHECK:STDOUT:     class E {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }

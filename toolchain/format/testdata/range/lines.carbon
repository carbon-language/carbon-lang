// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind format --output=- --lines=3:3 %s
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/range/lines.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/range/lines.carbon

// --- test.carbon

fn F() {
  var x: i32 = 1+1;
  var y: i32 = 2+2;
  var z: i32 = 3+3;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT:
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var x: i32 = 1 + 1;
// CHECK:STDOUT:   var y: i32 = 2+2;
// CHECK:STDOUT:   var z: i32 = 3+3;
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/match.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/match.carbon

// --- test.carbon

// Every non-empty block expands, so a packed `match` becomes one arm per line
// with three-line arm bodies. The `.Red` designator keeps a space before its
// `.` and none after.
fn Choose(color: Color) -> i32 {
  match (color) { case .Red => { return 1; } default => { return 0; } }
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // Every non-empty block expands, so a packed `match` becomes one arm per line
// CHECK:STDOUT: // with three-line arm bodies. The `.Red` designator keeps a space before its
// CHECK:STDOUT: // `.` and none after.
// CHECK:STDOUT: fn Choose(color: Color) -> i32 {
// CHECK:STDOUT:   match (color) {
// CHECK:STDOUT:     case .Red => {
// CHECK:STDOUT:       return 1;
// CHECK:STDOUT:     }
// CHECK:STDOUT:     default => {
// CHECK:STDOUT:       return 0;
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }

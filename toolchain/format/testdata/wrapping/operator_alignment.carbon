// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/operator_alignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/operator_alignment.carbon

// --- test.carbon

fn F() {
  // Breaks at the loosest operator (`+`, not `*`), keeps the `*` groups intact,
  // and aligns the continued operands under the first operand.
  return aaaaaaaaaa * bbbbbbbbbb + cccccccccc * dddddddddd + eeeeeeeeee * ffffffffff;

  // Keeps the right-hand side on the `=` line and breaks within it, aligning
  // operands under the first operand rather than the assignment continuation.
  total_value_xyz = aaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbb * ccccc;

  // Operands of an expression inside a call align after the open bracket.
  process_data(aaaaaaaaaa + bbbbbbbbbb + cccccccccc + dddddddddd + eeeeeeeeee + ffffff);
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   // Breaks at the loosest operator (`+`, not `*`), keeps the `*` groups intact,
// CHECK:STDOUT:   // and aligns the continued operands under the first operand.
// CHECK:STDOUT:   return aaaaaaaaaa * bbbbbbbbbb + cccccccccc * dddddddddd +
// CHECK:STDOUT:          eeeeeeeeee * ffffffffff;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Keeps the right-hand side on the `=` line and breaks within it, aligning
// CHECK:STDOUT:   // operands under the first operand rather than the assignment continuation.
// CHECK:STDOUT:   total_value_xyz =
// CHECK:STDOUT:       aaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbb * ccccc;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Operands of an expression inside a call align after the open bracket.
// CHECK:STDOUT:   process_data(aaaaaaaaaa + bbbbbbbbbb + cccccccccc + dddddddddd + eeeeeeeeee +
// CHECK:STDOUT:                ffffff);
// CHECK:STDOUT: }

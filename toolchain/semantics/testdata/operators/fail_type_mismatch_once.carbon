// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, body: {block2}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   12,
// CHECK:STDOUT:   12,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 34, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type1},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: nodeError, arg1: node+2, type: typeError},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+3, arg1: node+4, type: typeError},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+5},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %65 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %71: i32 = int_value 12
// CHECK:STDOUT:   %71.15: f64 = real_value 34e-1
// CHECK:STDOUT:   %71.13: <error> = add <error>, %71.15
// CHECK:STDOUT:   %71.21: i32 = int_value 12
// CHECK:STDOUT:   %71.19: <error> = add %71.13, %71.21
// CHECK:STDOUT:   return %71.19
// CHECK:STDOUT: }

fn Main() -> i32 {
  // The following line has two mismatches, but after the first, it shouldn't
  // keep erroring.
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+3]]:13: Cannot implicitly convert from `i32` to `f64`.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:             ^
  return 12 + 3.4 + 12;
}

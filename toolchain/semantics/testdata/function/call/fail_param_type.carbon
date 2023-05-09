// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT:   {param_ir: block1, param_refs: block2},
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 10, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   Run,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str1, arg1: callable0},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+2, arg1: block0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str2, arg1: callable1},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+4, arg1: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Run(a: i32) {}

fn Main() {
  // CHECK:STDERR: toolchain/semantics/testdata/function/call/fail_param_type.carbon:[[@LINE+2]]:6: No matching callable was found.
  // CHECK:STDERR: toolchain/semantics/testdata/function/call/fail_param_type.carbon:[[@LINE-4]]:1: Callable cannot be used: Cannot implicityly convert argument 0 from `f64` to `i32`.
  Run(1.0);
}

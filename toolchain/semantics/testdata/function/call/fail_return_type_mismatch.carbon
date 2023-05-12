// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT:   {arg_ir: block0, arg_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0, return_type: nodeFloatingPointType},
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 10, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Foo,
// CHECK:STDOUT:   Run,
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str0, arg1: callable0},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+1, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+0, arg1: block2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str1, arg1: callable1},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+5, type: nodeIntegerType},
// CHECK:STDOUT:   {kind: Call, arg0: call0, arg1: callable0, type: nodeFloatingPointType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+5, arg1: nodeInvalidType, type: nodeInvalidType},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+4, arg1: block3},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Foo() -> f64 { return 1.0; }

fn Run() {
  // CHECK:STDERR: toolchain/semantics/testdata/function/call/fail_return_type_mismatch.carbon:[[@LINE+1]]:21: Cannot implicitly convert from `f64` to `i32`.
  var x: i32 = Foo();
}

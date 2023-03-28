// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-semantics}
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT:   {param_ir: block1, param_refs: block2, return_type: nodeEmptyStruct},
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   Echo,
// CHECK:STDOUT:   Run,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+0, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str1, arg1: callable0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: nodeInvalidType, type: nodeInvalidType},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+2, arg1: block4},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str2, arg1: callable1},
// CHECK:STDOUT:   {kind: VarStorage, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+6, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: Assign, arg0: node+6, arg1: nodeInvalidType, type: nodeInvalidType},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+5, arg1: block5},
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
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Echo(a: {}) -> {} {
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_empty_struct.carbon:[[@LINE+1]]:10: Name a not found
  return a;
}

fn Run() {
  // TODO: Should succeed.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_empty_struct.carbon:[[@LINE+2]]:19: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_empty_struct.carbon:[[@LINE-8]]:1: Callable cannot be used: Received 0 argument(s), but require 1 argument(s).
  var a: {} = Echo({});
}

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
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node0, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node1, type: node1},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node2, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node3, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node4, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node5, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node6, type: node5},
// CHECK:STDOUT:   {kind: VarStorage, type: node2},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node7, type: node2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str1, arg1: callable0},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node9, arg1: block0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str2, arg1: callable1},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: node3},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node11, arg1: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node7,
// CHECK:STDOUT:     node8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node9,
// CHECK:STDOUT:     node10,
// CHECK:STDOUT:     node11,
// CHECK:STDOUT:     node13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Run(a: i32) {}

fn Main() {
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_type.carbon:[[@LINE+2]]:6: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_type.carbon:[[@LINE-4]]:1: Type mismatch: cannot convert argument 0 from node3 to node2.
  Run(1.0);
}

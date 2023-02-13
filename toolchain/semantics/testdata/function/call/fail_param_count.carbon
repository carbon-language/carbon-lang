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
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT:   {param_ir: block2, param_refs: block3},
// CHECK:STDOUT:   {param_ir: block4, param_refs: block5},
// CHECK:STDOUT:   {param_ir: block0, param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Run0,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   Run1,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   Run2,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node0, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node1, type: node1},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node2, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node3, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node4, type: node0},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: node5, type: node0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: callable0},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node6},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node6, arg1: block0},
// CHECK:STDOUT:   {kind: VarStorage, type: node2},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node9, type: node2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: callable1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node11},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node11, arg1: block0},
// CHECK:STDOUT:   {kind: VarStorage, type: node2},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node14, type: node2},
// CHECK:STDOUT:   {kind: VarStorage, type: node2},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node16, type: node2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: callable2},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node18},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node18, arg1: block0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: callable3},
// CHECK:STDOUT:   {kind: BindName, arg0: str5, arg1: node21},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: node2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: node2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: node2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: node2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: node2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: node2},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node21, arg1: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node6,
// CHECK:STDOUT:     node7,
// CHECK:STDOUT:     node8,
// CHECK:STDOUT:     node11,
// CHECK:STDOUT:     node12,
// CHECK:STDOUT:     node13,
// CHECK:STDOUT:     node18,
// CHECK:STDOUT:     node19,
// CHECK:STDOUT:     node20,
// CHECK:STDOUT:     node21,
// CHECK:STDOUT:     node22,
// CHECK:STDOUT:     node29,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node9,
// CHECK:STDOUT:     node10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node14,
// CHECK:STDOUT:     node15,
// CHECK:STDOUT:     node16,
// CHECK:STDOUT:     node17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node15,
// CHECK:STDOUT:     node17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node23,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node23,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node24,
// CHECK:STDOUT:     node25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node24,
// CHECK:STDOUT:     node25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node26,
// CHECK:STDOUT:     node27,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node26,
// CHECK:STDOUT:     node27,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node28,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node28,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Run0() {}
fn Run1(a: i32) {}
// CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:9: Redefining a in the same scope.
// CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-2]]:9: Previous definition is here.
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-8]]:1: Received 1 arguments, but require 0 parameters.
  Run0(1);
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-11]]:1: Received 2 arguments, but require 0 parameters.
  Run0(0, 1);

  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-14]]:1: Received 0 arguments, but require 1 parameters.
  Run1();
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-17]]:1: Received 2 arguments, but require 1 parameters.
  Run1(0, 1);

  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-18]]:1: Received 0 arguments, but require 2 parameters.
  Run2();
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: {{.*}}/toolchain/semantics/testdata/function/call/fail_param_count.carbon:[[@LINE-21]]:1: Received 1 arguments, but require 2 parameters.
  Run2(0);
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, body: block2},
// CHECK:STDOUT:   {name: str1, param_refs: block4, body: block5},
// CHECK:STDOUT:   {name: str3, param_refs: block7, body: block8},
// CHECK:STDOUT:   {name: str5, param_refs: block0, body: block9},
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
// CHECK:STDOUT:   Run1,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   Run2,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeEmptyTupleType,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+1, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+4, type: type1},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node+6, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+10, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+12, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+14, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+16, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+18, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+20, type: type1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Run0() {}
fn Run1(a: i32) {}
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-6]]:1: Function cannot be used: Received 1 argument(s), but require 0 argument(s).
  Run0(1);
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-9]]:1: Function cannot be used: Received 2 argument(s), but require 0 argument(s).
  Run0(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-12]]:1: Function cannot be used: Received 0 argument(s), but require 1 argument(s).
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-15]]:1: Function cannot be used: Received 2 argument(s), but require 1 argument(s).
  Run1(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-18]]:1: Function cannot be used: Received 0 argument(s), but require 2 argument(s).
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+2]]:7: No matching callable was found.
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-21]]:1: Function cannot be used: Received 1 argument(s), but require 2 argument(s).
  Run2(0);
}

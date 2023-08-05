// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, body: {block2}}},
// CHECK:STDOUT:   {name: str1, param_refs: block4, body: {block5}}},
// CHECK:STDOUT:   {name: str3, param_refs: block7, body: {block8}}},
// CHECK:STDOUT:   {name: str5, param_refs: block0, body: {block9}}},
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
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+6, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str4, arg1: node+8, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+13, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+15, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+21, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+23, type: type0},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %169 = fn_decl @Run0
// CHECK:STDOUT:   %170 = fn_decl @Run1
// CHECK:STDOUT:   %171 = fn_decl @Run2
// CHECK:STDOUT:   %173 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run0() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run1(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run2(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %180: i32 = int_value 1
// CHECK:STDOUT:   %180.8: i32 = stub_reference %180
// CHECK:STDOUT:   %187: i32 = int_value 0
// CHECK:STDOUT:   %187.8: i32 = stub_reference %187
// CHECK:STDOUT:   %187.11: i32 = int_value 1
// CHECK:STDOUT:   %187.11.0: i32 = stub_reference %187.11
// CHECK:STDOUT:   %202: i32 = int_value 0
// CHECK:STDOUT:   %202.8: i32 = stub_reference %202
// CHECK:STDOUT:   %202.11: i32 = int_value 1
// CHECK:STDOUT:   %202.11.0: i32 = stub_reference %202.11
// CHECK:STDOUT:   %217: i32 = int_value 0
// CHECK:STDOUT:   %217.8: i32 = stub_reference %217
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

fn Run0() {}
fn Run1(a: i32) {}
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: No matching callable was found.
  // CHECK:STDERR:   Run0(1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-8]]:1: Function cannot be used: Received 1 argument(s), but require 0 argument(s).
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^
  Run0(1);
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: No matching callable was found.
  // CHECK:STDERR:   Run0(0, 1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-15]]:1: Function cannot be used: Received 2 argument(s), but require 0 argument(s).
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^
  Run0(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: No matching callable was found.
  // CHECK:STDERR:   Run1();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-22]]:1: Function cannot be used: Received 0 argument(s), but require 1 argument(s).
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: No matching callable was found.
  // CHECK:STDERR:   Run1(0, 1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-29]]:1: Function cannot be used: Received 2 argument(s), but require 1 argument(s).
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^
  Run1(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: No matching callable was found.
  // CHECK:STDERR:   Run2();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-36]]:1: Function cannot be used: Received 0 argument(s), but require 2 argument(s).
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: No matching callable was found.
  // CHECK:STDERR:   Run2(0);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-43]]:1: Function cannot be used: Received 1 argument(s), but require 2 argument(s).
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^
  Run2(0);
}

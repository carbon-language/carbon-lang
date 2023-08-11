// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn Main() {
  var x: i32 = 0;
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: Name is previously declared here.
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^
  var x: i32 = 0;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, body: {block2}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Main,
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+1, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+3, arg1: node+1, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+1, arg1: node+4},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+6, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+8, arg1: node+6, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+6, arg1: node+9},
// CHECK:STDOUT:   {kind: Return},
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
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc8 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.loc9: i32 = var
// CHECK:STDOUT:   %.loc9_16.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_16.2: i32 = initialize_from %.loc9_16.1, %x.loc9
// CHECK:STDOUT:   assign %x.loc9, %.loc9_16.2
// CHECK:STDOUT:   %x.loc16: i32 = var
// CHECK:STDOUT:   %.loc16_16.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc16_16.2: i32 = initialize_from %.loc16_16.1, %x.loc16
// CHECK:STDOUT:   assign %x.loc16, %.loc16_16.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

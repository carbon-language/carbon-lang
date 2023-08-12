// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var n: i32 = 0;
  return *&*&n;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, return_slot: node+0, body: {block3}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   n,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+7,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+4, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+5},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+2, type: type1},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+8, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+9, type: type1},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+10, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+11, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+12, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+13},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n: i32 = var
// CHECK:STDOUT:   %.loc8_16.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc8_16.2: i32 = initialize_from %.loc8_16.1 to %n
// CHECK:STDOUT:   assign %n, %.loc8_16.2
// CHECK:STDOUT:   %.loc9_13.1: type = ptr_type i32
// CHECK:STDOUT:   %.loc9_13.2: i32* = address_of %n
// CHECK:STDOUT:   %.loc9_12: i32 = dereference %.loc9_13.2
// CHECK:STDOUT:   %.loc9_11: i32* = address_of %.loc9_12
// CHECK:STDOUT:   %.loc9_10.1: i32 = dereference %.loc9_11
// CHECK:STDOUT:   %.loc9_10.2: i32 = value_binding %.loc9_10.1
// CHECK:STDOUT:   %.loc9_10.3: i32 = initialize_from %.loc9_10.2 to %return
// CHECK:STDOUT:   return %.loc9_10.3
// CHECK:STDOUT: }

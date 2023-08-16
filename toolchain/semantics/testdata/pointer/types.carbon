// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, return_slot: node+4, body: {block4}}},
// CHECK:STDOUT:   {name: str2, param_refs: block6, return_type: type3, return_slot: node+15, body: {block7}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Ptr,
// CHECK:STDOUT:   p,
// CHECK:STDOUT:   ConstPtr,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT:   node+9,
// CHECK:STDOUT:   node+10,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+1, type: type1},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+1, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+6, arg1: node+4, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+7},
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type2, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+11, type: type3},
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type2, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+11, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+17, arg1: node+15, type: type3},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+18},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @Ptr
// CHECK:STDOUT:   %.loc11 = fn_decl @ConstPtr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p: i32*) -> %return: i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_8.1: i32* = value_binding %p
// CHECK:STDOUT:   %.loc7_8.2: i32* = initialize_from %.loc7_8.1 to %return
// CHECK:STDOUT:   return %.loc7_8.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p: const i32*) -> %return: const i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_13.1: const i32* = value_binding %p
// CHECK:STDOUT:   %.loc11_13.2: const i32* = initialize_from %.loc11_13.1 to %return
// CHECK:STDOUT:   return %.loc11_13.2
// CHECK:STDOUT: }

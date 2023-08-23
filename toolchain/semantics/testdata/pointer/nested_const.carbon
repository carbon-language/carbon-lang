// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, body: [block4]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   p,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   node+4,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: ConstType, arg0: type2, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type3, type: typeTypeType},
// CHECK:STDOUT:   {kind: ConstType, arg0: type4, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type5},
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+5, type: type3},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+8, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+9},
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
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc8 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: const (const (const i32*)*)) -> const i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_11: const (const i32*) = dereference %p
// CHECK:STDOUT:   %.loc9_10: const i32 = dereference %.loc9_11
// CHECK:STDOUT:   return %.loc9_10
// CHECK:STDOUT: }

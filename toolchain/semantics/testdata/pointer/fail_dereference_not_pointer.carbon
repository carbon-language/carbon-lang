// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Deref(n: i32) {
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  *n;
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: Cannot dereference operand of non-pointer type `() as type`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  *();
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: Cannot dereference operand of non-pointer type `{} as type`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  *{};
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, body: {block4}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Deref,
// CHECK:STDOUT:   n,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+5,
// CHECK:STDOUT:   node+8,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+3, type: typeError},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type1},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+6, type: typeError},
// CHECK:STDOUT:   {kind: StructType, arg0: block0, type: typeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: type2},
// CHECK:STDOUT:   {kind: Dereference, arg0: node+9, type: typeError},
// CHECK:STDOUT:   {kind: Return},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:   %.loc7 = fn_decl @Deref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = value_binding %n
// CHECK:STDOUT:   %.loc11: <error> = dereference %.loc7
// CHECK:STDOUT:   %.loc15_5.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc15_5.2: () = tuple_value ()
// CHECK:STDOUT:   %.loc15_3: <error> = dereference %.loc15_5.2
// CHECK:STDOUT:   %.loc19_5.1: type = struct_type {}
// CHECK:STDOUT:   %.loc19_5.2: {} = struct_value ()
// CHECK:STDOUT:   %.loc19_3: <error> = dereference %.loc19_5.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

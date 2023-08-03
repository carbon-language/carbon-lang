// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type4, body: {block4}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   ConstMismatch,
// CHECK:STDOUT:   p,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   node+7,
// CHECK:STDOUT:   node+8,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructType, arg0: block0, type: typeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: type0},
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+4, type: type2},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: type0},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: ConstType, arg0: type3, type: typeTypeType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: nodeError},
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
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %0: {} = struct_value ()
// CHECK:STDOUT:   %1: type = ptr_type {}
// CHECK:STDOUT:   %2: type = const_type {}*
// CHECK:STDOUT:   %3 = fn_decl @ConstMismatch
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p: const {}*) -> const ({}*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   ret <error>
// CHECK:STDOUT: }

fn ConstMismatch(p: const {}*) -> const ({}*) {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:11: Cannot implicitly convert from `const {}*` to `const ({}*)`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:           ^
  return p;
}

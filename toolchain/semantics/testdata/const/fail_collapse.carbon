// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:9: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: fn G(p: const (const i32)**) -> i32** {
// CHECK:STDERR:         ^
fn G(p: const (const i32)**) -> i32** {
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:11: Cannot implicitly convert from `const i32**` to `i32**`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:           ^
  return p;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type5, body: [block4]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   G,
// CHECK:STDOUT:   p,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   node+5,
// CHECK:STDOUT:   node+6,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: ConstType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type2, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type3},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type4, type: typeTypeType},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc10_36: type = ptr_type i32
// CHECK:STDOUT:   %.loc10_37: type = ptr_type i32*
// CHECK:STDOUT:   %.loc10_1 = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: const i32**) -> i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }

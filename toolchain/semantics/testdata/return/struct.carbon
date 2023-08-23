// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> {.a: i32} {
  return {.a = 3};
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type1, body: [block4]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   3,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Main,
// CHECK:STDOUT:   a,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block2, type: typeTypeType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+3, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block6, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+6},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_14: i32 = stub_reference %.loc8_16
// CHECK:STDOUT:   %.loc8_17: {.a: i32} = struct_value (%.loc8_14)
// CHECK:STDOUT:   return %.loc8_17
// CHECK:STDOUT: }

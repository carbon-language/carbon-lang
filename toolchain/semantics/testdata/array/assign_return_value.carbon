// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type2, body: {block3}}},
// CHECK:STDOUT:   {name: str1, param_refs: block0, body: {block5}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   Run,
// CHECK:STDOUT:   t,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   node+11,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+5, type: type1},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type2},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+7},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+10, arg1: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type3},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+12, type: type3},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type2},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+14, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+12, arg1: node+11},
// CHECK:STDOUT:   {kind: Return},
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
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_12: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_16.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_16.2: (type,) = tuple_value (%.loc7_12)
// CHECK:STDOUT:   %.loc7_16.3: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_1 = fn_decl @F
// CHECK:STDOUT:   %.loc9 = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28.1: i32 = int_value 0
// CHECK:STDOUT:   %.loc7_28.2: i32 = stub_reference %.loc7_28.1
// CHECK:STDOUT:   %.loc7_30: (i32,) = tuple_value (%.loc7_28.2)
// CHECK:STDOUT:   return %.loc7_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_value 1
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32
// CHECK:STDOUT:   %t: array [i32; 1] = var
// CHECK:STDOUT:   %.loc10_22.1: (i32,) = call @F()
// CHECK:STDOUT:   %.loc10_22.2: array [i32; 1] = array_value %.loc10_22.1
// CHECK:STDOUT:   assign %t, %.loc10_17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

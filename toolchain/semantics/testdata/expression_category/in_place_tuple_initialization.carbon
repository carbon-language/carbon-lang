// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
  return G()[0];
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type2, return_slot: node+5},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type2, return_slot: node+10, body: [block6]},
// CHECK:STDOUT:   {name: str4, param_refs: block0, return_type: type1, body: [block12]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   return,
// CHECK:STDOUT:   G,
// CHECK:STDOUT:   v,
// CHECK:STDOUT:   H,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+4,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block5, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block7, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str3, type: type2},
// CHECK:STDOUT:   {kind: NoOp},
// CHECK:STDOUT:   {kind: Call, arg0: block8, arg1: function0, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+15, arg1: node+17},
// CHECK:STDOUT:   {kind: NoOp},
// CHECK:STDOUT:   {kind: Call, arg0: block9, arg1: function0, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+15, arg1: node+20},
// CHECK:STDOUT:   {kind: NoOp},
// CHECK:STDOUT:   {kind: Call, arg0: block10, arg1: function0, type: type2},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+23},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: TemporaryStorage, type: type2},
// CHECK:STDOUT:   {kind: Call, arg0: block13, arg1: function1, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: Temporary, arg0: node+27, arg1: node+28, type: type2},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+30, arg1: node+29, type: type1},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+31, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+32},
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
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+33,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc9 = fn_decl @G
// CHECK:STDOUT:   %.loc15 = fn_decl @H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_11: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_16: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_19: (type, type) = tuple_value (%.loc10_11, %.loc10_16)
// CHECK:STDOUT:   %v: ref (i32, i32) = var "v"
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc10_24: init (i32, i32) = call @F() to %v
// CHECK:STDOUT:   assign %v, %.loc10_24
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc11: init (i32, i32) = call @F() to %v
// CHECK:STDOUT:   assign %v, %.loc11
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc12: init (i32, i32) = call @F() to %return
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_11.1: ref (i32, i32) = temporary_storage
// CHECK:STDOUT:   %.loc16_11.2: init (i32, i32) = call @G() to %.loc16_11.1
// CHECK:STDOUT:   %.loc16_14: i32 = int_literal 0
// CHECK:STDOUT:   %.loc16_11.3: ref (i32, i32) = temporary %.loc16_11.1, %.loc16_11.2
// CHECK:STDOUT:   %.loc16_15.1: ref i32 = tuple_index %.loc16_11.3, %.loc16_14
// CHECK:STDOUT:   %.loc16_15.2: i32 = bind_value %.loc16_15.1
// CHECK:STDOUT:   return %.loc16_15.2
// CHECK:STDOUT: }

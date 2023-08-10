// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G() -> i32;

fn H() -> {.a: i32};

fn AddressOfLiteral() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &0;
  // CHECK:STDERR:   ^
  &0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &true;
  // CHECK:STDERR:   ^
  &true;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &1.0;
  // CHECK:STDERR:   ^
  &1.0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &"Hello";
  // CHECK:STDERR:   ^
  &"Hello";
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1, 2);
  // CHECK:STDERR:   ^
  &(1, 2);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &{.a = 5};
  // CHECK:STDERR:   ^
  &{.a = 5};
}

fn AddressOfOperator() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1 + 1);
  // CHECK:STDERR:   ^
  &(1 + 1);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &H().a;
  // CHECK:STDERR:   ^
  &H().a;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(not true);
  // CHECK:STDERR:   ^
  &(not true);
}

fn AddressOfCall() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &G();
  // CHECK:STDERR:   ^
  &G();
}

fn AddressOfType() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &i32;
  // CHECK:STDERR:   ^
  &i32;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(const i32*);
  // CHECK:STDERR:   ^
  &(const i32*);
}

fn AddressOfTupleElementValue() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &((1, 2)[0]);
  // CHECK:STDERR:   ^
  &((1, 2)[0]);
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0},
// CHECK:STDOUT:   {name: str1, param_refs: block0, return_type: type1},
// CHECK:STDOUT:   {name: str3, param_refs: block0, body: {block3}}},
// CHECK:STDOUT:   {name: str5, param_refs: block0, body: {block7}}},
// CHECK:STDOUT:   {name: str6, param_refs: block0, body: {block8}}},
// CHECK:STDOUT:   {name: str7, param_refs: block0, body: {block9}}},
// CHECK:STDOUT:   {name: str8, param_refs: block0, body: {block10}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   5,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 10, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   G,
// CHECK:STDOUT:   H,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   AddressOfLiteral,
// CHECK:STDOUT:   Hello,
// CHECK:STDOUT:   AddressOfOperator,
// CHECK:STDOUT:   AddressOfCall,
// CHECK:STDOUT:   AddressOfType,
// CHECK:STDOUT:   AddressOfTupleElementValue,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+6,
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT:   node+9,
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT:   node+12,
// CHECK:STDOUT:   nodeStringType,
// CHECK:STDOUT:   node+15,
// CHECK:STDOUT:   node+21,
// CHECK:STDOUT:   node+23,
// CHECK:STDOUT:   node+29,
// CHECK:STDOUT:   node+49,
// CHECK:STDOUT:   node+51,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block2, type: typeTypeType},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+5, type: type2},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type3},
// CHECK:STDOUT:   {kind: PointerType, arg0: type3, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+8, type: type4},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type5},
// CHECK:STDOUT:   {kind: PointerType, arg0: type5, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+11, type: type6},
// CHECK:STDOUT:   {kind: StringLiteral, arg0: str4, type: type7},
// CHECK:STDOUT:   {kind: PointerType, arg0: type7, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+14, type: type8},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type9},
// CHECK:STDOUT:   {kind: PointerType, arg0: type9, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+22, type: type10},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str2, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+25, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block6, type: type1},
// CHECK:STDOUT:   {kind: PointerType, arg0: type1, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+28, type: type11},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function3},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+33, arg1: node+34, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+35, type: type2},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type1},
// CHECK:STDOUT:   {kind: StructAccess, arg0: node+37, arg1: member0, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+38, type: type2},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type3},
// CHECK:STDOUT:   {kind: UnaryOperatorNot, arg0: node+40, type: type3},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+41, type: type4},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function4},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+45, type: type2},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function5},
// CHECK:STDOUT:   {kind: PointerType, arg0: typeTypeType, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: nodeIntegerType, type: type12},
// CHECK:STDOUT:   {kind: ConstType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: PointerType, arg0: type13, type: typeTypeType},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+52, type: type12},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function6},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int6, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+56, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int7, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+58, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block11, type: type9},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int8, type: type0},
// CHECK:STDOUT:   {kind: TupleIndex, arg0: node+60, arg1: node+61, type: type0},
// CHECK:STDOUT:   {kind: AddressOf, arg0: node+62, type: type2},
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
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+44,
// CHECK:STDOUT:     node+48,
// CHECK:STDOUT:     node+55,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+33,
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+38,
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:     node+41,
// CHECK:STDOUT:     node+42,
// CHECK:STDOUT:     node+43,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+45,
// CHECK:STDOUT:     node+46,
// CHECK:STDOUT:     node+47,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+49,
// CHECK:STDOUT:     node+50,
// CHECK:STDOUT:     node+51,
// CHECK:STDOUT:     node+52,
// CHECK:STDOUT:     node+53,
// CHECK:STDOUT:     node+54,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+56,
// CHECK:STDOUT:     node+57,
// CHECK:STDOUT:     node+58,
// CHECK:STDOUT:     node+59,
// CHECK:STDOUT:     node+60,
// CHECK:STDOUT:     node+61,
// CHECK:STDOUT:     node+62,
// CHECK:STDOUT:     node+63,
// CHECK:STDOUT:     node+64,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+57,
// CHECK:STDOUT:     node+59,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @G
// CHECK:STDOUT:   %.loc9_19: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc9_1 = fn_decl @H
// CHECK:STDOUT:   %.loc11 = fn_decl @AddressOfLiteral
// CHECK:STDOUT:   %.loc38 = fn_decl @AddressOfOperator
// CHECK:STDOUT:   %.loc53 = fn_decl @AddressOfCall
// CHECK:STDOUT:   %.loc60 = fn_decl @AddressOfType
// CHECK:STDOUT:   %.loc71 = fn_decl @AddressOfTupleElementValue
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> {.a: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc15_3.1: type = ptr_type i32
// CHECK:STDOUT:   %.loc15_3.2: i32* = address_of %.loc15_4
// CHECK:STDOUT:   %.loc19_4: bool = bool_literal true
// CHECK:STDOUT:   %.loc19_3.1: type = ptr_type bool
// CHECK:STDOUT:   %.loc19_3.2: bool* = address_of %.loc19_4
// CHECK:STDOUT:   %.loc23_4: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc23_3.1: type = ptr_type f64
// CHECK:STDOUT:   %.loc23_3.2: f64* = address_of %.loc23_4
// CHECK:STDOUT:   %.loc27_4: String = string_literal "Hello"
// CHECK:STDOUT:   %.loc27_3.1: type = ptr_type String
// CHECK:STDOUT:   %.loc27_3.2: String* = address_of %.loc27_4
// CHECK:STDOUT:   %.loc31_5.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc31_5.2: i32 = stub_reference %.loc31_5.1
// CHECK:STDOUT:   %.loc31_8.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc31_8.2: i32 = stub_reference %.loc31_8.1
// CHECK:STDOUT:   %.loc31_9.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc31_9.2: (i32, i32) = tuple_value (%.loc31_5.2, %.loc31_8.2)
// CHECK:STDOUT:   %.loc31_3.1: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc31_3.2: (i32, i32)* = address_of %.loc31_9.2
// CHECK:STDOUT:   %.loc35_10: i32 = int_literal 5
// CHECK:STDOUT:   %.loc35_8: i32 = stub_reference %.loc35_10
// CHECK:STDOUT:   %.loc35_11: {.a: i32} = struct_value (%.loc35_8)
// CHECK:STDOUT:   %.loc35_3.1: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc35_3.2: {.a: i32}* = address_of %.loc35_11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfOperator() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc42_5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc42_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc42_7: i32 = add %.loc42_5, %.loc42_9
// CHECK:STDOUT:   %.loc42_3: i32* = address_of %.loc42_7
// CHECK:STDOUT:   %.loc46_5: {.a: i32} = call @H()
// CHECK:STDOUT:   %.loc46_7: i32 = struct_access %.loc46_5, member0
// CHECK:STDOUT:   %.loc46_3: i32* = address_of %.loc46_7
// CHECK:STDOUT:   %.loc50_9: bool = bool_literal true
// CHECK:STDOUT:   %.loc50_5: bool = not %.loc50_9
// CHECK:STDOUT:   %.loc50_3: bool* = address_of %.loc50_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfCall() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc57_5: i32 = call @G()
// CHECK:STDOUT:   %.loc57_3: i32* = address_of %.loc57_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc64_3.1: type = ptr_type type
// CHECK:STDOUT:   %.loc64_3.2: type* = address_of i32
// CHECK:STDOUT:   %.loc68_5: type = const_type i32
// CHECK:STDOUT:   %.loc68_14: type = ptr_type const i32
// CHECK:STDOUT:   %.loc68_3: type* = address_of %.loc68_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfTupleElementValue() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc75_6.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc75_6.2: i32 = stub_reference %.loc75_6.1
// CHECK:STDOUT:   %.loc75_9.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc75_9.2: i32 = stub_reference %.loc75_9.1
// CHECK:STDOUT:   %.loc75_10: (i32, i32) = tuple_value (%.loc75_6.2, %.loc75_9.2)
// CHECK:STDOUT:   %.loc75_12: i32 = int_literal 0
// CHECK:STDOUT:   %.loc75_13: i32 = tuple_index %.loc75_10, %.loc75_12
// CHECK:STDOUT:   %.loc75_3: i32* = address_of %.loc75_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

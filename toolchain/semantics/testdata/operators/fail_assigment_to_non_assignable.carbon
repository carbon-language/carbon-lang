// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32;

fn Main() {
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:3: Expression is not assignable.
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  1 = 2;
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:5: Expression is not assignable.
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:     ^
  F() = 1;
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:8: Expression is not assignable.
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:        ^
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:8: Expression is not assignable.
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:        ^
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:3: Expression is not assignable.
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^
  i32 = i32*;
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:18: Expression is not assignable.
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:                  ^
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:25: Expression is not assignable.
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:                         ^
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:25: Expression is not assignable.
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:                         ^
  (if true then a else a) = 10;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0},
// CHECK:STDOUT:   {name: str1, param_refs: block0, body: {block2, block11, block12, block13, block14, block15, block16}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   10,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT:   n,
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   y,
// CHECK:STDOUT:   a,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+12,
// CHECK:STDOUT:   node+41,
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+3},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+5, arg1: node+6},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+10, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+14, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int6, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+16, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+13, arg1: node+18},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+20, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int7, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+20, arg1: node+22},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+20, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+20, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block5, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int8, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+27, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int9, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+29, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block6, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+26, arg1: node+31},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: Assign, arg0: nodeIntegerType, arg1: node+33},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int10, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+35, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int11, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+38, type: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block7, type: typeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block8, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int12, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+43, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int13, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+46, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block10, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+42, arg1: node+49},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type3},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block11, arg1: node+51},
// CHECK:STDOUT:   {kind: Branch, arg0: block12},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int14, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int15, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block13, arg1: node+54},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block13, arg1: node+55},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block13, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int16, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+58, arg1: node+59},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str5, arg1: node+61, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type3},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block14, arg1: node+63},
// CHECK:STDOUT:   {kind: Branch, arg0: block15},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block16, arg1: node+61},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block16, arg1: node+61},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block16, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int17, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+68, arg1: node+69},
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
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+33,
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+38,
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:     node+41,
// CHECK:STDOUT:     node+42,
// CHECK:STDOUT:     node+43,
// CHECK:STDOUT:     node+45,
// CHECK:STDOUT:     node+46,
// CHECK:STDOUT:     node+48,
// CHECK:STDOUT:     node+49,
// CHECK:STDOUT:     node+50,
// CHECK:STDOUT:     node+51,
// CHECK:STDOUT:     node+52,
// CHECK:STDOUT:     node+53,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+44,
// CHECK:STDOUT:     node+47,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+45,
// CHECK:STDOUT:     node+48,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+54,
// CHECK:STDOUT:     node+56,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+55,
// CHECK:STDOUT:     node+57,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+58,
// CHECK:STDOUT:     node+59,
// CHECK:STDOUT:     node+60,
// CHECK:STDOUT:     node+61,
// CHECK:STDOUT:     node+62,
// CHECK:STDOUT:     node+63,
// CHECK:STDOUT:     node+64,
// CHECK:STDOUT:     node+65,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+66,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+67,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+68,
// CHECK:STDOUT:     node+69,
// CHECK:STDOUT:     node+70,
// CHECK:STDOUT:     node+71,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

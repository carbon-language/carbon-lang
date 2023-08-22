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
// CHECK:STDOUT:   {name: str1, param_refs: block0, body: [block2, block11, block12, block13, block14, block15, block16]},
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
// CHECK:STDOUT:   node+40,
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
// CHECK:STDOUT:   {kind: VarStorage, arg0: str2, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int7, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+20, arg1: node+21},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+20, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+20, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block5, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int8, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+26, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int9, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+28, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block6, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+25, arg1: node+30},
// CHECK:STDOUT:   {kind: PointerType, arg0: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: Assign, arg0: nodeIntegerType, arg1: node+32},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int10, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+34, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int11, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+37, type: type0},
// CHECK:STDOUT:   {kind: StructType, arg0: block7, type: typeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block8, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int12, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str3, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+42, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int13, type: type0},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str4, arg1: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+45, type: type0},
// CHECK:STDOUT:   {kind: StructValue, arg0: block10, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+41, arg1: node+48},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type3},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block11, arg1: node+50},
// CHECK:STDOUT:   {kind: Branch, arg0: block12},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int14, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int15, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block13, arg1: node+53},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block13, arg1: node+54},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block13, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int16, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+57, arg1: node+58},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str5, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type3},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block14, arg1: node+61},
// CHECK:STDOUT:   {kind: Branch, arg0: block15},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block16, arg1: node+60},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block16, arg1: node+60},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block16, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int17, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+66, arg1: node+67},
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
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:     node+41,
// CHECK:STDOUT:     node+42,
// CHECK:STDOUT:     node+44,
// CHECK:STDOUT:     node+45,
// CHECK:STDOUT:     node+47,
// CHECK:STDOUT:     node+48,
// CHECK:STDOUT:     node+49,
// CHECK:STDOUT:     node+50,
// CHECK:STDOUT:     node+51,
// CHECK:STDOUT:     node+52,
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
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:     node+38,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+43,
// CHECK:STDOUT:     node+46,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+44,
// CHECK:STDOUT:     node+47,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+53,
// CHECK:STDOUT:     node+55,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+54,
// CHECK:STDOUT:     node+56,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+57,
// CHECK:STDOUT:     node+58,
// CHECK:STDOUT:     node+59,
// CHECK:STDOUT:     node+60,
// CHECK:STDOUT:     node+61,
// CHECK:STDOUT:     node+62,
// CHECK:STDOUT:     node+63,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+64,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+65,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+66,
// CHECK:STDOUT:     node+67,
// CHECK:STDOUT:     node+68,
// CHECK:STDOUT:     node+69,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc9 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_3: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_7: i32 = int_literal 2
// CHECK:STDOUT:   assign %.loc13_3, %.loc13_7
// CHECK:STDOUT:   %.loc17_4: i32 = call @F()
// CHECK:STDOUT:   %.loc17_9: i32 = int_literal 1
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_9
// CHECK:STDOUT:   %.loc21_4.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc21_4.2: i32 = stub_reference %.loc21_4.1
// CHECK:STDOUT:   %.loc21_7.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc21_7.2: i32 = stub_reference %.loc21_7.1
// CHECK:STDOUT:   %.loc21_8.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc21_8.2: (i32, i32) = tuple_value (%.loc21_4.2, %.loc21_7.2)
// CHECK:STDOUT:   %.loc21_13.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc21_13.2: i32 = stub_reference %.loc21_13.1
// CHECK:STDOUT:   %.loc21_16.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc21_16.2: i32 = stub_reference %.loc21_16.1
// CHECK:STDOUT:   %.loc21_17: (i32, i32) = tuple_value (%.loc21_13.2, %.loc21_16.2)
// CHECK:STDOUT:   assign %.loc21_8.2, %.loc21_17
// CHECK:STDOUT:   %n: i32 = var "n"
// CHECK:STDOUT:   %.loc22: i32 = int_literal 0
// CHECK:STDOUT:   assign %n, %.loc22
// CHECK:STDOUT:   %.loc26_4: i32 = stub_reference %n
// CHECK:STDOUT:   %.loc26_7: i32 = stub_reference %n
// CHECK:STDOUT:   %.loc26_8: (i32, i32) = tuple_value (%.loc26_4, %.loc26_7)
// CHECK:STDOUT:   %.loc26_13.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc26_13.2: i32 = stub_reference %.loc26_13.1
// CHECK:STDOUT:   %.loc26_16.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc26_16.2: i32 = stub_reference %.loc26_16.1
// CHECK:STDOUT:   %.loc26_17: (i32, i32) = tuple_value (%.loc26_13.2, %.loc26_16.2)
// CHECK:STDOUT:   assign %.loc26_8, %.loc26_17
// CHECK:STDOUT:   %.loc30: type = ptr_type i32
// CHECK:STDOUT:   assign i32, %.loc30
// CHECK:STDOUT:   %.loc34_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc34_7: i32 = stub_reference %.loc34_9
// CHECK:STDOUT:   %.loc34_17: i32 = int_literal 2
// CHECK:STDOUT:   %.loc34_15: i32 = stub_reference %.loc34_17
// CHECK:STDOUT:   %.loc34_18.1: type = struct_type {.x: i32, .y: i32}
// CHECK:STDOUT:   %.loc34_18.2: {.x: i32, .y: i32} = struct_value (%.loc34_7, %.loc34_15)
// CHECK:STDOUT:   %.loc34_28: i32 = int_literal 3
// CHECK:STDOUT:   %.loc34_26: i32 = stub_reference %.loc34_28
// CHECK:STDOUT:   %.loc34_36: i32 = int_literal 4
// CHECK:STDOUT:   %.loc34_34: i32 = stub_reference %.loc34_36
// CHECK:STDOUT:   %.loc34_37: {.x: i32, .y: i32} = struct_value (%.loc34_26, %.loc34_34)
// CHECK:STDOUT:   assign %.loc34_18.2, %.loc34_37
// CHECK:STDOUT:   %.loc38_7: bool = bool_literal true
// CHECK:STDOUT:   if %.loc38_7 br !if.expr.then.loc38 else br !if.expr.else.loc38
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc38:
// CHECK:STDOUT:   %.loc38_17: i32 = int_literal 1
// CHECK:STDOUT:   br !if.expr.result.loc38(%.loc38_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc38:
// CHECK:STDOUT:   %.loc38_24: i32 = int_literal 2
// CHECK:STDOUT:   br !if.expr.result.loc38(%.loc38_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc38:
// CHECK:STDOUT:   %.loc38_4: i32 = block_arg !if.expr.result.loc38
// CHECK:STDOUT:   %.loc38_29: i32 = int_literal 3
// CHECK:STDOUT:   assign %.loc38_4, %.loc38_29
// CHECK:STDOUT:   %a: i32 = var "a"
// CHECK:STDOUT:   %.loc45_7: bool = bool_literal true
// CHECK:STDOUT:   if %.loc45_7 br !if.expr.then.loc45 else br !if.expr.else.loc45
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc45:
// CHECK:STDOUT:   br !if.expr.result.loc45(%a)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc45:
// CHECK:STDOUT:   br !if.expr.result.loc45(%a)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc45:
// CHECK:STDOUT:   %.loc45_4: i32 = block_arg !if.expr.result.loc45
// CHECK:STDOUT:   %.loc45_29: i32 = int_literal 10
// CHECK:STDOUT:   assign %.loc45_4, %.loc45_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

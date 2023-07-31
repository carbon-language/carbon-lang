// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, body: {block4}}},
// CHECK:STDOUT:   {name: str4, param_refs: block0, body: {block5}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   5,
// CHECK:STDOUT:   6,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   Foo,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+21,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+4, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+9, arg1: node+10, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+11, arg1: node+12, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+13, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+15, arg1: node+16, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: Call, arg0: block6, arg1: function0, type: type1},
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
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %0 = FunctionDeclaration @Foo
// CHECK:STDOUT:   %1 = FunctionDeclaration @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32, %c: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   Return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %0: i32 = IntegerLiteral 1
// CHECK:STDOUT:   %1: i32 = IntegerLiteral 2
// CHECK:STDOUT:   %2: i32 = BinaryOperatorAdd %0, %1
// CHECK:STDOUT:   %3: i32 = IntegerLiteral 3
// CHECK:STDOUT:   %4: i32 = BinaryOperatorAdd %2, %3
// CHECK:STDOUT:   %5: i32 = StubReference %4
// CHECK:STDOUT:   %6: i32 = IntegerLiteral 4
// CHECK:STDOUT:   %7: i32 = IntegerLiteral 5
// CHECK:STDOUT:   %8: i32 = BinaryOperatorAdd %6, %7
// CHECK:STDOUT:   %9: i32 = StubReference %8
// CHECK:STDOUT:   %10: i32 = IntegerLiteral 6
// CHECK:STDOUT:   %11: i32 = StubReference %10
// CHECK:STDOUT:   %12: type = TupleType ()
// CHECK:STDOUT:   %13: () as type = Call @Foo(%5, %9, %11)
// CHECK:STDOUT:   Return
// CHECK:STDOUT: }

fn Foo(a: i32, b: i32, c: i32) {}

fn Main() {
  // Generates multiple IR instructions for the first two parameters.
  Foo(1 + 2 + 3, 4 + 5, 6);
}

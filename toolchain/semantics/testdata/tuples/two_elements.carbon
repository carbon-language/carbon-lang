// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   4,
// CHECK:STDOUT:   102,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   y,
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
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+5, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+7, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+9, type: type1},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+5, arg1: node+11},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+16, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+16, arg1: node+5},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %0: type = StubReference <noderef 3>
// CHECK:STDOUT:   %1: type = StubReference <noderef 3>
// CHECK:STDOUT:   %2: type = TupleType {type, type}
// CHECK:STDOUT:   %3: (type, type) as type = TupleValue {%0, %1}
// CHECK:STDOUT:   %4: type = TupleType {i32, i32}
// CHECK:STDOUT:   %x: (i32, i32) as type = VarStorage
// CHECK:STDOUT:   %5: i32 = IntegerLiteral 4
// CHECK:STDOUT:   %6: i32 = StubReference %5
// CHECK:STDOUT:   %7: i32 = IntegerLiteral 102
// CHECK:STDOUT:   %8: i32 = StubReference %7
// CHECK:STDOUT:   %9: (i32, i32) as type = TupleValue {%6, %8}
// CHECK:STDOUT:   Assign (i32, i32) as type, %x, %9
// CHECK:STDOUT:   %10: type = StubReference <noderef 3>
// CHECK:STDOUT:   %11: type = StubReference <noderef 3>
// CHECK:STDOUT:   %12: (type, type) as type = TupleValue {%10, %11}
// CHECK:STDOUT:   %y: (i32, i32) as type = VarStorage
// CHECK:STDOUT:   Assign (i32, i32) as type, %y, %x
// CHECK:STDOUT: }

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
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
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+9,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+11,
// CHECK:STDOUT:   node+12,
// CHECK:STDOUT:   node+21,
// CHECK:STDOUT:   node+32,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:     type2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type5,
// CHECK:STDOUT:     type5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+7, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type1},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock2, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock3, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type4},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+13, type: type4},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+15, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+17, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+19, type: type2},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock4, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block5, type: type5},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+22, type: type5},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+24, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+26, type: type2},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type2},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+28, type: type2},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block7, type: type5},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+30, type: type5},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock5, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block6, type: type6},
// CHECK:STDOUT:   {kind: Assign, arg0: node+13, arg1: nodeError},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %0: type = StubReference i32
// CHECK:STDOUT:   %1: type = StubReference i32
// CHECK:STDOUT:   %2: type = TupleType (type, type)
// CHECK:STDOUT:   %3: (type, type) as type = TupleValue (%0, %1)
// CHECK:STDOUT:   %4: (type, type) as type = StubReference %3
// CHECK:STDOUT:   %5: type = StubReference i32
// CHECK:STDOUT:   %6: type = StubReference i32
// CHECK:STDOUT:   %7: (type, type) as type = TupleValue (%5, %6)
// CHECK:STDOUT:   %8: (type, type) as type = StubReference %7
// CHECK:STDOUT:   %9: type = TupleType ((type, type) as type, (type, type) as type)
// CHECK:STDOUT:   %10: ((type, type), (type, type)) as type = TupleValue (%4, %8)
// CHECK:STDOUT:   %11: type = TupleType (i32, i32)
// CHECK:STDOUT:   %12: type = TupleType ((i32, i32) as type, (i32, i32) as type)
// CHECK:STDOUT:   %x: ((i32, i32), (i32, i32)) as type = VarStorage
// CHECK:STDOUT:   %13: i32 = IntegerLiteral 1
// CHECK:STDOUT:   %14: i32 = StubReference %13
// CHECK:STDOUT:   %15: i32 = IntegerLiteral 2
// CHECK:STDOUT:   %16: i32 = StubReference %15
// CHECK:STDOUT:   %17: i32 = IntegerLiteral 3
// CHECK:STDOUT:   %18: i32 = StubReference %17
// CHECK:STDOUT:   %19: type = TupleType (i32, i32, i32)
// CHECK:STDOUT:   %20: (i32, i32, i32) as type = TupleValue (%14, %16, %18)
// CHECK:STDOUT:   %21: (i32, i32, i32) as type = StubReference %20
// CHECK:STDOUT:   %22: i32 = IntegerLiteral 4
// CHECK:STDOUT:   %23: i32 = StubReference %22
// CHECK:STDOUT:   %24: i32 = IntegerLiteral 5
// CHECK:STDOUT:   %25: i32 = StubReference %24
// CHECK:STDOUT:   %26: i32 = IntegerLiteral 6
// CHECK:STDOUT:   %27: i32 = StubReference %26
// CHECK:STDOUT:   %28: (i32, i32, i32) as type = TupleValue (%23, %25, %27)
// CHECK:STDOUT:   %29: (i32, i32, i32) as type = StubReference %28
// CHECK:STDOUT:   %30: type = TupleType ((i32, i32, i32) as type, (i32, i32, i32) as type)
// CHECK:STDOUT:   %31: ((i32, i32, i32), (i32, i32, i32)) as type = TupleValue (%21, %29)
// CHECK:STDOUT:   Assign %x, <error>
// CHECK:STDOUT: }

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:57: Cannot implicitly convert from `((i32, i32, i32), (i32, i32, i32)) as type` to `((i32, i32), (i32, i32)) as type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                                         ^
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:57: Cannot implicitly convert from `((i32, i32, i32), (i32, i32, i32)) as type` to `((i32, i32), (i32, i32)) as type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                                         ^
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

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
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: nodeError, arg1: node+13, type: typeError},
// CHECK:STDOUT:   {kind: Assign, arg0: node+13, arg1: node+34},
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
// CHECK:STDOUT:     node+35,
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
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc10_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc10_18.2: (type, type) = tuple_value (%.loc10_10, %.loc10_15)
// CHECK:STDOUT:   %.loc10_18.3: (type, type) = stub_reference %.loc10_18.2
// CHECK:STDOUT:   %.loc10_22: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_27: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_30.1: (type, type) = tuple_value (%.loc10_22, %.loc10_27)
// CHECK:STDOUT:   %.loc10_30.2: (type, type) = stub_reference %.loc10_30.1
// CHECK:STDOUT:   %.loc10_31.1: type = tuple_type ((type, type), (type, type))
// CHECK:STDOUT:   %.loc10_31.2: ((type, type), (type, type)) = tuple_value (%.loc10_18.3, %.loc10_30.2)
// CHECK:STDOUT:   %.loc10_18.4: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc10_31.3: type = tuple_type ((i32, i32), (i32, i32))
// CHECK:STDOUT:   %x: ((i32, i32), (i32, i32)) = var
// CHECK:STDOUT:   %.loc10_37.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_37.2: i32 = stub_reference %.loc10_37.1
// CHECK:STDOUT:   %.loc10_40.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_40.2: i32 = stub_reference %.loc10_40.1
// CHECK:STDOUT:   %.loc10_43.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc10_43.2: i32 = stub_reference %.loc10_43.1
// CHECK:STDOUT:   %.loc10_44.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_44.2: (i32, i32, i32) = tuple_value (%.loc10_37.2, %.loc10_40.2, %.loc10_43.2)
// CHECK:STDOUT:   %.loc10_44.3: (i32, i32, i32) = stub_reference %.loc10_44.2
// CHECK:STDOUT:   %.loc10_48.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc10_48.2: i32 = stub_reference %.loc10_48.1
// CHECK:STDOUT:   %.loc10_51.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc10_51.2: i32 = stub_reference %.loc10_51.1
// CHECK:STDOUT:   %.loc10_54.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc10_54.2: i32 = stub_reference %.loc10_54.1
// CHECK:STDOUT:   %.loc10_55.1: (i32, i32, i32) = tuple_value (%.loc10_48.2, %.loc10_51.2, %.loc10_54.2)
// CHECK:STDOUT:   %.loc10_55.2: (i32, i32, i32) = stub_reference %.loc10_55.1
// CHECK:STDOUT:   %.loc10_56.1: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT:   %.loc10_56.2: ((i32, i32, i32), (i32, i32, i32)) = tuple_value (%.loc10_44.3, %.loc10_55.2)
// CHECK:STDOUT:   %.1: <error> = initialize_from <error>, %x
// CHECK:STDOUT:   assign %x, %.1
// CHECK:STDOUT: }

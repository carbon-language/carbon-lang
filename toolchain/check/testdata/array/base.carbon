// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (1,);
var b: [f64; 2] = (11.1, 2.2,);
var c: [(); 5] = ((), (), (), (), (),);

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   5,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 111, exponent: -1, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 22, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   node+5,
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT:   node+10,
// CHECK:STDOUT:   node+16,
// CHECK:STDOUT:   node+20,
// CHECK:STDOUT:   node+23,
// CHECK:STDOUT:   node+35,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type6,
// CHECK:STDOUT:     type6,
// CHECK:STDOUT:     type6,
// CHECK:STDOUT:     type6,
// CHECK:STDOUT:     type6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+0, arg1: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str0, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+3, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type2},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+6, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+7},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+9, arg1: type3, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type4},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+12, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real1, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+14, type: type3},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type5},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+17, type: type4},
// CHECK:STDOUT:   {kind: Assign, arg0: node+11, arg1: node+18},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock2, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type6},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+22, arg1: type6, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str2, type: type7},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type6},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+25, type: type6},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type6},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+27, type: type6},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type6},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+29, type: type6},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type6},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+31, type: type6},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type6},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+33, type: type6},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock3, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type8},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+36, type: type7},
// CHECK:STDOUT:   {kind: Assign, arg0: node+24, arg1: node+37},
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
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:     node+38,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_22.2: (i32,) = tuple_value (%.loc7_20.2)
// CHECK:STDOUT:   %.loc7_22.3: [i32; 1] = array_value %.loc7_22.2
// CHECK:STDOUT:   assign %a, %.loc7_22.3
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, f64
// CHECK:STDOUT:   %b: ref [f64; 2] = var "b"
// CHECK:STDOUT:   %.loc8_20.1: f64 = real_literal 111e-1
// CHECK:STDOUT:   %.loc8_20.2: f64 = stub_reference %.loc8_20.1
// CHECK:STDOUT:   %.loc8_26.1: f64 = real_literal 22e-1
// CHECK:STDOUT:   %.loc8_26.2: f64 = stub_reference %.loc8_26.1
// CHECK:STDOUT:   %.loc8_30.1: type = tuple_type (f64, f64)
// CHECK:STDOUT:   %.loc8_30.2: (f64, f64) = tuple_value (%.loc8_20.2, %.loc8_26.2)
// CHECK:STDOUT:   %.loc8_30.3: [f64; 2] = array_value %.loc8_30.2
// CHECK:STDOUT:   assign %b, %.loc8_30.3
// CHECK:STDOUT:   %.loc9_10.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc9_10.2: () = tuple_value ()
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 5
// CHECK:STDOUT:   %.loc9_14: type = array_type %.loc9_13, ()
// CHECK:STDOUT:   %c: ref [(); 5] = var "c"
// CHECK:STDOUT:   %.loc9_20.1: () = tuple_value ()
// CHECK:STDOUT:   %.loc9_20.2: () = stub_reference %.loc9_20.1
// CHECK:STDOUT:   %.loc9_24.1: () = tuple_value ()
// CHECK:STDOUT:   %.loc9_24.2: () = stub_reference %.loc9_24.1
// CHECK:STDOUT:   %.loc9_28.1: () = tuple_value ()
// CHECK:STDOUT:   %.loc9_28.2: () = stub_reference %.loc9_28.1
// CHECK:STDOUT:   %.loc9_32.1: () = tuple_value ()
// CHECK:STDOUT:   %.loc9_32.2: () = stub_reference %.loc9_32.1
// CHECK:STDOUT:   %.loc9_36.1: () = tuple_value ()
// CHECK:STDOUT:   %.loc9_36.2: () = stub_reference %.loc9_36.1
// CHECK:STDOUT:   %.loc9_38.1: type = tuple_type ((), (), (), (), ())
// CHECK:STDOUT:   %.loc9_38.2: ((), (), (), (), ()) = tuple_value (%.loc9_20.2, %.loc9_24.2, %.loc9_28.2, %.loc9_32.2, %.loc9_36.2)
// CHECK:STDOUT:   %.loc9_38.3: [(); 5] = array_value %.loc9_38.2
// CHECK:STDOUT:   assign %c, %.loc9_38.3
// CHECK:STDOUT: }

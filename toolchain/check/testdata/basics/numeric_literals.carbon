// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 5] = (
    8,
    9,
    0x8,
    0b1000,
    39999999999999999993,
  );
  var floats: [f64; 7] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
    39999999999999999993.0e39999999999999999993,
  );
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, body: [block2]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   5,
// CHECK:STDOUT:   8,
// CHECK:STDOUT:   9,
// CHECK:STDOUT:   8,
// CHECK:STDOUT:   8,
// CHECK:STDOUT:   39999999999999999993,
// CHECK:STDOUT:   7,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 9, exponent: -1, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 80, exponent: -1, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 800, exponent: -1, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 10, exponent: 6, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 10, exponent: 7, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 10, exponent: -9, is_decimal: 1},
// CHECK:STDOUT:   {mantissa: 399999999999999999930, exponent: 39999999999999999992, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   ints,
// CHECK:STDOUT:   floats,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   node+14,
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT:   node+19,
// CHECK:STDOUT:   node+35,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:     type3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+1, arg1: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+4, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+6, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int4, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+10, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int5, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+12, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type2},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+15, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+3, arg1: node+16},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int6, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+18, arg1: type3, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str2, type: type4},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+21, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real1, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+23, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real2, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+25, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real3, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+27, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real4, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+29, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real5, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+31, type: type3},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real6, type: type3},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+33, type: type3},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block4, type: type5},
// CHECK:STDOUT:   {kind: ArrayValue, arg0: node+36, type: type4},
// CHECK:STDOUT:   {kind: Assign, arg0: node+20, arg1: node+37},
// CHECK:STDOUT:   {kind: Return},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 5
// CHECK:STDOUT:   %.loc10_20: type = array_type %.loc10_19, i32
// CHECK:STDOUT:   %ints: ref [i32; 5] = var "ints"
// CHECK:STDOUT:   %.loc11_5.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc11_5.2: i32 = stub_reference %.loc11_5.1
// CHECK:STDOUT:   %.loc12_5.1: i32 = int_literal 9
// CHECK:STDOUT:   %.loc12_5.2: i32 = stub_reference %.loc12_5.1
// CHECK:STDOUT:   %.loc13_5.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc13_5.2: i32 = stub_reference %.loc13_5.1
// CHECK:STDOUT:   %.loc14_5.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc14_5.2: i32 = stub_reference %.loc14_5.1
// CHECK:STDOUT:   %.loc15_5.1: i32 = int_literal 39999999999999999993
// CHECK:STDOUT:   %.loc15_5.2: i32 = stub_reference %.loc15_5.1
// CHECK:STDOUT:   %.loc16_3.1: type = tuple_type (i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc16_3.2: (i32, i32, i32, i32, i32) = tuple_value (%.loc11_5.2, %.loc12_5.2, %.loc13_5.2, %.loc14_5.2, %.loc15_5.2)
// CHECK:STDOUT:   %.loc16_3.3: [i32; 5] = array_value %.loc16_3.2
// CHECK:STDOUT:   assign %ints, %.loc16_3.3
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 7
// CHECK:STDOUT:   %.loc17_22: type = array_type %.loc17_21, f64
// CHECK:STDOUT:   %floats: ref [f64; 7] = var "floats"
// CHECK:STDOUT:   %.loc18_5.1: f64 = real_literal 9e-1
// CHECK:STDOUT:   %.loc18_5.2: f64 = stub_reference %.loc18_5.1
// CHECK:STDOUT:   %.loc19_5.1: f64 = real_literal 80e-1
// CHECK:STDOUT:   %.loc19_5.2: f64 = stub_reference %.loc19_5.1
// CHECK:STDOUT:   %.loc20_5.1: f64 = real_literal 800e-1
// CHECK:STDOUT:   %.loc20_5.2: f64 = stub_reference %.loc20_5.1
// CHECK:STDOUT:   %.loc21_5.1: f64 = real_literal 10e6
// CHECK:STDOUT:   %.loc21_5.2: f64 = stub_reference %.loc21_5.1
// CHECK:STDOUT:   %.loc22_5.1: f64 = real_literal 10e7
// CHECK:STDOUT:   %.loc22_5.2: f64 = stub_reference %.loc22_5.1
// CHECK:STDOUT:   %.loc23_5.1: f64 = real_literal 10e-9
// CHECK:STDOUT:   %.loc23_5.2: f64 = stub_reference %.loc23_5.1
// CHECK:STDOUT:   %.loc24_5.1: f64 = real_literal 399999999999999999930e39999999999999999992
// CHECK:STDOUT:   %.loc24_5.2: f64 = stub_reference %.loc24_5.1
// CHECK:STDOUT:   %.loc25_3.1: type = tuple_type (f64, f64, f64, f64, f64, f64, f64)
// CHECK:STDOUT:   %.loc25_3.2: (f64, f64, f64, f64, f64, f64, f64) = tuple_value (%.loc18_5.2, %.loc19_5.2, %.loc20_5.2, %.loc21_5.2, %.loc22_5.2, %.loc23_5.2, %.loc24_5.2)
// CHECK:STDOUT:   %.loc25_3.3: [f64; 7] = array_value %.loc25_3.2
// CHECK:STDOUT:   assign %floats, %.loc25_3.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

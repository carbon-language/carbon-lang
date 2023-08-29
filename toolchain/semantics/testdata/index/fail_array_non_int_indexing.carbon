// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (12,);
// CHECK:STDERR: fail_array_non_int_indexing.carbon:[[@LINE+3]]:16: Cannot implicitly convert from `f64` to `i32`.
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                ^
var b: i32 = a[2.6];

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   12,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT:   {mantissa: 26, exponent: -1, is_decimal: 1},
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   node+5,
// CHECK:STDOUT:   nodeFloatingPointType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
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
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: RealLiteral, arg0: real0, type: type3},
// CHECK:STDOUT:   {kind: ArrayIndex, arg0: node+2, arg1: nodeError, type: type0},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+11, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+9, arg1: node+12},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_23.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_23.2: (i32,) = tuple_value (%.loc7_20.2)
// CHECK:STDOUT:   %.loc7_23.3: [i32; 1] = array_value %.loc7_23.2
// CHECK:STDOUT:   assign %a, %.loc7_23.3
// CHECK:STDOUT:   %b: i32 = var "b"
// CHECK:STDOUT:   %.loc11_16: f64 = real_literal 26e-1
// CHECK:STDOUT:   %.loc11_19.1: i32 = array_index %a, <error>
// CHECK:STDOUT:   %.loc11_19.2: i32 = bind_value %.loc11_19.1
// CHECK:STDOUT:   assign %b, %.loc11_19.2
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_out_of_bound.carbon:[[@LINE+3]]:28: Cannot implicitly convert from `(i32, i32, i32) as type` to `array [i32; 1] as type`.
// CHECK:STDERR: var a: [i32; 1] = (1, 2, 3);
// CHECK:STDERR:                            ^
var a: [i32; 1] = (1, 2, 3);

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   node+10,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:     type0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ArrayType, arg0: node+0, arg1: type0, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+2, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+4, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+6, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type0},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+8, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type2},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: nodeError},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc10_14: i32 = int_value 1
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, i32
// CHECK:STDOUT:   %a: array [i32; 1] = var
// CHECK:STDOUT:   %.loc10_20.1: i32 = int_value 1
// CHECK:STDOUT:   %.loc10_20.2: i32 = stub_reference %.loc10_20.1
// CHECK:STDOUT:   %.loc10_23.1: i32 = int_value 2
// CHECK:STDOUT:   %.loc10_23.2: i32 = stub_reference %.loc10_23.1
// CHECK:STDOUT:   %.loc10_26.1: i32 = int_value 3
// CHECK:STDOUT:   %.loc10_26.2: i32 = stub_reference %.loc10_26.1
// CHECK:STDOUT:   %.loc10_27.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_27.2: (i32, i32, i32) = tuple_value (%.loc10_20.2, %.loc10_23.2, %.loc10_26.2)
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   2,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+2,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+4,
// CHECK:STDOUT:   node+9,
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
// CHECK:STDOUT:   [
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
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock2, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block3, type: type3},
// CHECK:STDOUT:   {kind: Assign, arg0: node+5, arg1: nodeError},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %0: type = StubReference i32
// CHECK:STDOUT:   %1: type = StubReference i32
// CHECK:STDOUT:   %2: type = TupleType (type, type)
// CHECK:STDOUT:   %3: (type, type) = TupleValue (%0, %1)
// CHECK:STDOUT:   %4: type = TupleType (i32, i32)
// CHECK:STDOUT:   %x: (i32, i32) = VarStorage
// CHECK:STDOUT:   %5: i32 = IntegerLiteral 2
// CHECK:STDOUT:   %6: i32 = StubReference %5
// CHECK:STDOUT:   %7: type = TupleType (i32)
// CHECK:STDOUT:   %8: (i32,) = TupleValue (%6)
// CHECK:STDOUT:   Assign %x, <error>
// CHECK:STDOUT: }

// CHECK:STDERR: fail_too_few_element.carbon:[[@LINE+3]]:26: Cannot implicitly convert from `(i32,) as type` to `(i32, i32) as type`.
// CHECK:STDERR: var x: (i32, i32) = (2, );
// CHECK:STDERR:                          ^
var x: (i32, i32) = (2, );

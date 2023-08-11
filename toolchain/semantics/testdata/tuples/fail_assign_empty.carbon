// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_empty.carbon:[[@LINE+3]]:19: Cannot implicitly convert from `() as type` to `(i32,) as type`.
// CHECK:STDERR: var x: (i32,) = ();
// CHECK:STDERR:                   ^
var x: (i32,) = ();

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+1,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+3,
// CHECK:STDOUT:   node+6,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:     typeTypeType,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     type1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StubReference, arg0: nodeIntegerType, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block2, type: type0},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:   {kind: VarStorage, type: type2},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+4, type: type2},
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock2, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type3},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: nodeError, arg1: node+4, type: typeError},
// CHECK:STDOUT:   {kind: Assign, arg0: node+4, arg1: node+8},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc10_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc10_13.2: (type,) = tuple_value (%.loc10_9)
// CHECK:STDOUT:   %.loc10_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %x: (i32,) = var
// CHECK:STDOUT:   %.loc10_18.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc10_18.2: () = tuple_value ()
// CHECK:STDOUT:   %.1: <error> = initialize_from <error>, %x
// CHECK:STDOUT:   assign %x, %.1
// CHECK:STDOUT: }

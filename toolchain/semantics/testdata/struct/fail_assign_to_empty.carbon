// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:21: Cannot implicitly convert from `{.a: i32}` to `{} as type`.
// CHECK:STDERR: var x: {} = {.a = 1};
// CHECK:STDERR:                     ^
var x: {} = {.a = 1};

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   a,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   node+7,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructType, arg0: block0, type: typeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: StructTypeField, arg0: str1, arg1: type1},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+4, type: type1},
// CHECK:STDOUT:   {kind: StructType, arg0: block2, type: typeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block3, type: type2},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: nodeError, arg1: node+2, type: typeError},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+9},
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
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc10_9.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_9.2: {} = struct_value ()
// CHECK:STDOUT:   %x: {} = var
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_17: i32 = stub_reference %.loc10_19
// CHECK:STDOUT:   %.loc10_20.1: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc10_20.2: {.a: i32} = struct_value (%.loc10_17)
// CHECK:STDOUT:   %.1: <error> = initialize_from <error>, %x
// CHECK:STDOUT:   assign %x, %.1
// CHECK:STDOUT: }

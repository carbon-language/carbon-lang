// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: () = ();
var y: () = x;

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   x,
// CHECK:STDOUT:   y,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+4, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+2, arg1: node+5},
// CHECK:STDOUT:   {kind: TupleValue, arg0: block0, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+8, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+2, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+10, arg1: node+8, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+8, arg1: node+11},
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
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7_9.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_9.2: () = tuple_value ()
// CHECK:STDOUT:   %x: () = var
// CHECK:STDOUT:   %.loc7_14.1: () = tuple_value ()
// CHECK:STDOUT:   %.loc7_14.2: () = initialize_from %.loc7_14.1 to %x
// CHECK:STDOUT:   assign %x, %.loc7_14.2
// CHECK:STDOUT:   %.loc8: () = tuple_value ()
// CHECK:STDOUT:   %y: () = var
// CHECK:STDOUT:   %.loc7_5.1: () = value_binding %x
// CHECK:STDOUT:   %.loc7_5.2: () = initialize_from %.loc7_5.1 to %y
// CHECK:STDOUT:   assign %y, %.loc7_5.2
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE

fn Foo(n: i32) -> (i32, f64) {
  return (n + 2, 3.4);
}

// CHECK:STDOUT: - filename: raw_ir.carbon
// CHECK:STDOUT:   sem_ir:
// CHECK:STDOUT:   - cross_reference_irs_size: 1
// CHECK:STDOUT:     functions: [
// CHECK:STDOUT:       {name: str0, param_refs: block1, return_type: type3, return_slot: node+4, body: [block4]},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     classes: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     integers: [
// CHECK:STDOUT:       2,
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     reals: [
// CHECK:STDOUT:       {mantissa: 34, exponent: -1, is_decimal: 1},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     strings: [
// CHECK:STDOUT:       Foo,
// CHECK:STDOUT:       n,
// CHECK:STDOUT:       return,
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     types: [
// CHECK:STDOUT:       {node: nodeIntegerType, value_rep: {kind: copy, type: type0}},
// CHECK:STDOUT:       {node: node+1, value_rep: {kind: unknown, type: type<invalid>}},
// CHECK:STDOUT:       {node: nodeFloatingPointType, value_rep: {kind: copy, type: type2}},
// CHECK:STDOUT:       {node: node+3, value_rep: {kind: pointer, type: type4}},
// CHECK:STDOUT:       {node: node+5, value_rep: {kind: copy, type: type4}},
// CHECK:STDOUT:       {node: nodeFunctionType, value_rep: {kind: copy, type: type5}},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     type_blocks: [
// CHECK:STDOUT:       [
// CHECK:STDOUT:         typeTypeType,
// CHECK:STDOUT:         typeTypeType,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         type0,
// CHECK:STDOUT:         type2,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     nodes: [
// CHECK:STDOUT:       {kind: Parameter, arg0: str1, type: type0},
// CHECK:STDOUT:       {kind: TupleType, arg0: typeBlock0, type: typeTypeType},
// CHECK:STDOUT:       {kind: TupleLiteral, arg0: block2, type: type1},
// CHECK:STDOUT:       {kind: TupleType, arg0: typeBlock1, type: typeTypeType},
// CHECK:STDOUT:       {kind: VarStorage, arg0: str2, type: type3},
// CHECK:STDOUT:       {kind: PointerType, arg0: type3, type: typeTypeType},
// CHECK:STDOUT:       {kind: FunctionDeclaration, arg0: function0, type: type5},
// CHECK:STDOUT:       {kind: NameReference, arg0: str1, arg1: node+0, type: type0},
// CHECK:STDOUT:       {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:       {kind: BinaryOperatorAdd, arg0: node+7, arg1: node+8, type: type0},
// CHECK:STDOUT:       {kind: RealLiteral, arg0: real0, type: type2},
// CHECK:STDOUT:       {kind: TupleLiteral, arg0: block5, type: type3},
// CHECK:STDOUT:       {kind: TupleAccess, arg0: node+4, arg1: member0, type: type0},
// CHECK:STDOUT:       {kind: InitializeFrom, arg0: node+9, arg1: node+12, type: type0},
// CHECK:STDOUT:       {kind: TupleAccess, arg0: node+4, arg1: member1, type: type2},
// CHECK:STDOUT:       {kind: InitializeFrom, arg0: node+10, arg1: node+14, type: type2},
// CHECK:STDOUT:       {kind: TupleInit, arg0: node+11, arg1: block6, type: type3},
// CHECK:STDOUT:       {kind: ReturnExpression, arg0: node+16},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     node_blocks: [
// CHECK:STDOUT:       [
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+0,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         nodeIntegerType,
// CHECK:STDOUT:         nodeFloatingPointType,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+0,
// CHECK:STDOUT:         node+1,
// CHECK:STDOUT:         node+2,
// CHECK:STDOUT:         node+3,
// CHECK:STDOUT:         node+4,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+7,
// CHECK:STDOUT:         node+8,
// CHECK:STDOUT:         node+9,
// CHECK:STDOUT:         node+10,
// CHECK:STDOUT:         node+11,
// CHECK:STDOUT:         node+12,
// CHECK:STDOUT:         node+13,
// CHECK:STDOUT:         node+14,
// CHECK:STDOUT:         node+15,
// CHECK:STDOUT:         node+16,
// CHECK:STDOUT:         node+17,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+9,
// CHECK:STDOUT:         node+10,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+13,
// CHECK:STDOUT:         node+15,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+5,
// CHECK:STDOUT:         node+6,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:     ]

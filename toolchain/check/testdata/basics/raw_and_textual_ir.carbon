// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-sem-ir --dump-raw-sem-ir %s
//
// Check that we can combine textual IR and raw IR dumping in one compile.
//
// AUTOUPDATE

fn Foo(n: i32) -> (i32, f64) {
  return (n + 2, 3.4);
}

// CHECK:STDOUT: - filename: raw_and_textual_ir.carbon
// CHECK:STDOUT:   sem_ir:
// CHECK:STDOUT:   - cross_reference_irs_size: 1
// CHECK:STDOUT:     functions: [
// CHECK:STDOUT:       {name: str0, param_refs: block1, return_type: type3, return_slot: node+4, body: [block4]},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     integer_literals: [
// CHECK:STDOUT:       2,
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     real_literals: [
// CHECK:STDOUT:       {mantissa: 34, exponent: -1, is_decimal: 1},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     strings: [
// CHECK:STDOUT:       Foo,
// CHECK:STDOUT:       n,
// CHECK:STDOUT:       return,
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     types: [
// CHECK:STDOUT:       nodeIntegerType,
// CHECK:STDOUT:       node+1,
// CHECK:STDOUT:       nodeFloatingPointType,
// CHECK:STDOUT:       node+3,
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
// CHECK:STDOUT:       {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:       {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:       {kind: BinaryOperatorAdd, arg0: node+0, arg1: node+6, type: type0},
// CHECK:STDOUT:       {kind: RealLiteral, arg0: real0, type: type2},
// CHECK:STDOUT:       {kind: TupleLiteral, arg0: block5, type: type3},
// CHECK:STDOUT:       {kind: TupleAccess, arg0: node+4, arg1: member0, type: type0},
// CHECK:STDOUT:       {kind: InitializeFrom, arg0: node+7, arg1: node+10, type: type0},
// CHECK:STDOUT:       {kind: TupleAccess, arg0: node+4, arg1: member1, type: type2},
// CHECK:STDOUT:       {kind: InitializeFrom, arg0: node+8, arg1: node+12, type: type2},
// CHECK:STDOUT:       {kind: TupleInit, arg0: node+9, arg1: block6, type: type3},
// CHECK:STDOUT:       {kind: ReturnExpression, arg0: node+14},
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
// CHECK:STDOUT:         node+6,
// CHECK:STDOUT:         node+7,
// CHECK:STDOUT:         node+8,
// CHECK:STDOUT:         node+9,
// CHECK:STDOUT:         node+10,
// CHECK:STDOUT:         node+11,
// CHECK:STDOUT:         node+12,
// CHECK:STDOUT:         node+13,
// CHECK:STDOUT:         node+14,
// CHECK:STDOUT:         node+15,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+7,
// CHECK:STDOUT:         node+8,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+11,
// CHECK:STDOUT:         node+13,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+5,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:     ]
// CHECK:STDOUT:
// CHECK:STDOUT: file "raw_and_textual_ir.carbon" {
// CHECK:STDOUT:   %.loc11 = fn_decl @Foo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: i32) -> %return: (i32, f64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_15: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_13: i32 = add %n, %.loc12_15
// CHECK:STDOUT:   %.loc12_18: f64 = real_literal 34e-1
// CHECK:STDOUT:   %.loc12_21: (i32, f64) = tuple_literal (%.loc12_13, %.loc12_18)
// CHECK:STDOUT:   %.loc12_22.1: ref i32 = tuple_access %return, member0
// CHECK:STDOUT:   %.loc12_22.2: init i32 = initialize_from %.loc12_13 to %.loc12_22.1
// CHECK:STDOUT:   %.loc12_22.3: ref f64 = tuple_access %return, member1
// CHECK:STDOUT:   %.loc12_22.4: init f64 = initialize_from %.loc12_18 to %.loc12_22.3
// CHECK:STDOUT:   %.loc12_22.5: init (i32, f64) = tuple_init %.loc12_21, (%.loc12_22.2, %.loc12_22.4)
// CHECK:STDOUT:   return %.loc12_22.5
// CHECK:STDOUT: }

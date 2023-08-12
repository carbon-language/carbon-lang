// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, return_slot: node+0, body: {block3}}},
// CHECK:STDOUT:   {name: str1, param_refs: block0, return_type: type0, return_slot: node+5, body: {block5}}},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type0, return_slot: node+10, body: {block7, block9, block10}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   G,
// CHECK:STDOUT:   Or,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+2, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+3},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+7, arg1: node+5, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+8},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: Call, arg0: block8, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, arg0: node+12, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+13, type: type0},
// CHECK:STDOUT:   {kind: UnaryOperatorNot, arg0: node+14, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block9, arg1: node+15},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block10, arg1: node+16},
// CHECK:STDOUT:   {kind: Call, arg0: block11, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+20, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block10, arg1: node+21},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block10, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+23, arg1: node+10, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+24},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc8 = fn_decl @G
// CHECK:STDOUT:   %.loc10 = fn_decl @Or
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_25.1: bool = bool_literal true
// CHECK:STDOUT:   %.loc7_25.2: bool = initialize_from %.loc7_25.1 to %return
// CHECK:STDOUT:   return %.loc7_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_25.1: bool = bool_literal true
// CHECK:STDOUT:   %.loc8_25.2: bool = initialize_from %.loc8_25.1 to %return
// CHECK:STDOUT:   return %.loc8_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> %return: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_11.1: bool = call @F() to %.loc11_11.2
// CHECK:STDOUT:   %.loc11_11.2: bool = materialize_temporary %.loc11_11.1
// CHECK:STDOUT:   %.loc11_11.3: bool = value_binding %.loc11_11.2
// CHECK:STDOUT:   %.loc11_14.1: bool = not %.loc11_11.3
// CHECK:STDOUT:   %.loc11_14.2: bool = bool_literal true
// CHECK:STDOUT:   if %.loc11_14.1 br !or.rhs else br !or.result(%.loc11_14.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %.loc11_18.1: bool = call @G() to %.loc11_18.2
// CHECK:STDOUT:   %.loc11_18.2: bool = materialize_temporary %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.3: bool = value_binding %.loc11_18.2
// CHECK:STDOUT:   br !or.result(%.loc11_18.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc11_14.3: bool = block_arg !or.result
// CHECK:STDOUT:   %.loc11_14.4: bool = initialize_from %.loc11_14.3 to %return
// CHECK:STDOUT:   return %.loc11_14.4
// CHECK:STDOUT: }

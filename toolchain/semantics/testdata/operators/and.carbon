// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, body: [block3]},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type0, body: [block5]},
// CHECK:STDOUT:   {name: str3, param_refs: block0, return_type: type0, body: [block7, block8, block9]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   return,
// CHECK:STDOUT:   G,
// CHECK:STDOUT:   And,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+2},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+6},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+11, arg1: node+10},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+11, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: false, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block8, arg1: node+13},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block9, arg1: node+14},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, type: type0},
// CHECK:STDOUT:   {kind: Assign, arg0: node+18, arg1: node+17},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+18, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block9, arg1: node+20},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block9, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+22},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc8 = fn_decl @G
// CHECK:STDOUT:   %.loc10 = fn_decl @And
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: bool = bool_literal true
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_11.1: init bool = call @F()
// CHECK:STDOUT:   %.loc11_11.2: ref bool = materialize_temporary
// CHECK:STDOUT:   assign %.loc11_11.2, %.loc11_11.1
// CHECK:STDOUT:   %.loc11_11.3: bool = bind_value %.loc11_11.2
// CHECK:STDOUT:   %.loc11_14.1: bool = bool_literal false
// CHECK:STDOUT:   if %.loc11_11.3 br !and.rhs else br !and.result(%.loc11_14.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %.loc11_19.1: init bool = call @G()
// CHECK:STDOUT:   %.loc11_19.2: ref bool = materialize_temporary
// CHECK:STDOUT:   assign %.loc11_19.2, %.loc11_19.1
// CHECK:STDOUT:   %.loc11_19.3: bool = bind_value %.loc11_19.2
// CHECK:STDOUT:   br !and.result(%.loc11_19.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_14.2: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc11_14.2
// CHECK:STDOUT: }

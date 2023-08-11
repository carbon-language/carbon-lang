// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool) -> i32 {
  return 0;

  // Exercise various forms of control flow to check that we still consider the
  // end of the function unreachable after them.
  var n: i32 = if b then 1 else 2;
  if (b and (b or b)) { }
  if (b) { } else { }

  // Unreachable, no error on missing return.
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, return_slot: node+2, body: {block5}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   0,
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   n,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+4, arg1: node+2, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+5},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+7, type: type1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type1},
// CHECK:STDOUT:   {kind: BlockArg, arg0: unreachable, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+12, arg1: node+7, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+7, arg1: node+13},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: false, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: UnaryOperatorNot, arg0: node+17, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: unreachable, arg1: node+20},
// CHECK:STDOUT:   {kind: BlockArg, arg0: unreachable, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: unreachable, arg1: node+22},
// CHECK:STDOUT:   {kind: BlockArg, arg0: unreachable, type: type0},
// CHECK:STDOUT:   {kind: Branch, arg0: unreachable},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_10.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc8_10.2: i32 = initialize_from %.loc8_10.1, %return
// CHECK:STDOUT:   return %.loc8_10.2
// CHECK:STDOUT: }

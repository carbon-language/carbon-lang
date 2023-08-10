// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: bool, b: bool, c: bool) -> i32 {
  return if a then if b then 1 else 2 else if c then 3 else 4;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, body: {block4, block5, block7, block8, block9, block6, block10, block11, block12, block13}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   3,
// CHECK:STDOUT:   4,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   c,
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
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+2, type: type0},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+4, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block5, arg1: node+0},
// CHECK:STDOUT:   {kind: Branch, arg0: block6},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block7, arg1: node+2},
// CHECK:STDOUT:   {kind: Branch, arg0: block8},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block9, arg1: node+11},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block9, arg1: node+12},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block9, type: type1},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block10, arg1: node+4},
// CHECK:STDOUT:   {kind: Branch, arg0: block11},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int3, type: type1},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block12, arg1: node+18},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block12, arg1: node+19},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block12, type: type1},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block13, arg1: node+15},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block13, arg1: node+22},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block13, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+25},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: bool, %b: bool, %c: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %a br !.loc8_20.1 else br !.loc8_44.1
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_20.1:
// CHECK:STDOUT:   if %b br !.loc8_30 else br !.loc8_37
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_30:
// CHECK:STDOUT:   %.loc8_30: i32 = int_literal 1
// CHECK:STDOUT:   br !.loc8_20.2(%.loc8_30)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_37:
// CHECK:STDOUT:   %.loc8_37: i32 = int_literal 2
// CHECK:STDOUT:   br !.loc8_20.2(%.loc8_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_20.2:
// CHECK:STDOUT:   %.loc8_20: i32 = block_arg !.loc8_20.2
// CHECK:STDOUT:   br !.loc8_10(%.loc8_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_44.1:
// CHECK:STDOUT:   if %c br !.loc8_54 else br !.loc8_61
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_54:
// CHECK:STDOUT:   %.loc8_54: i32 = int_literal 3
// CHECK:STDOUT:   br !.loc8_44.2(%.loc8_54)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_61:
// CHECK:STDOUT:   %.loc8_61: i32 = int_literal 4
// CHECK:STDOUT:   br !.loc8_44.2(%.loc8_61)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_44.2:
// CHECK:STDOUT:   %.loc8_44: i32 = block_arg !.loc8_44.2
// CHECK:STDOUT:   br !.loc8_10(%.loc8_44)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8_10:
// CHECK:STDOUT:   %.loc8_10: i32 = block_arg !.loc8_10
// CHECK:STDOUT:   return %.loc8_10
// CHECK:STDOUT: }

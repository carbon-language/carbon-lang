// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F() -> i32 {
  return if true then A() else B();
}

fn G() -> i32 {
  return if false then A() else B();
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, body: {block2}}},
// CHECK:STDOUT:   {name: str1, param_refs: block0, return_type: type0, body: {block3}}},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type0, body: {block4, block5, block6, block7}}},
// CHECK:STDOUT:   {name: str3, param_refs: block0, return_type: type0, body: {block8, block9, block10, block11}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   A,
// CHECK:STDOUT:   B,
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   G,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeIntegerType,
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+4},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type1},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block5, arg1: node+7},
// CHECK:STDOUT:   {kind: Branch, arg0: block6},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block7, arg1: node+10},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block7, arg1: node+11},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block7, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+14},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function3},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: false, type: type1},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block9, arg1: node+17},
// CHECK:STDOUT:   {kind: Branch, arg0: block10},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block11, arg1: node+20},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block11, arg1: node+21},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block11, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+24},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @A
// CHECK:STDOUT:   %.loc8 = fn_decl @B
// CHECK:STDOUT:   %.loc10 = fn_decl @F
// CHECK:STDOUT:   %.loc14 = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_value 1
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_value 2
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_13: bool = bool_value true
// CHECK:STDOUT:   if %.loc11_13 br !.loc11_24 else br !.loc11_33
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_24:
// CHECK:STDOUT:   %.loc11_24: i32 = call @A()
// CHECK:STDOUT:   br !.loc11_10(%.loc11_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_33:
// CHECK:STDOUT:   %.loc11_33: i32 = call @B()
// CHECK:STDOUT:   br !.loc11_10(%.loc11_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_10:
// CHECK:STDOUT:   %.loc11_10: i32 = block_arg !.loc11_10
// CHECK:STDOUT:   return %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_value false
// CHECK:STDOUT:   if %.loc15_13 br !.loc15_25 else br !.loc15_34
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc15_25:
// CHECK:STDOUT:   %.loc15_25: i32 = call @A()
// CHECK:STDOUT:   br !.loc15_10(%.loc15_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc15_34:
// CHECK:STDOUT:   %.loc15_34: i32 = call @B()
// CHECK:STDOUT:   br !.loc15_10(%.loc15_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc15_10:
// CHECK:STDOUT:   %.loc15_10: i32 = block_arg !.loc15_10
// CHECK:STDOUT:   return %.loc15_10
// CHECK:STDOUT: }

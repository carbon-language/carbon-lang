// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool, n: i32, m: i32) -> i32 {
  return if b then n + m else m + n;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, body: {block4, block5, block6, block7}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   n,
// CHECK:STDOUT:   m,
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
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+2, type: type1},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+4, type: type1},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block5, arg1: node+8},
// CHECK:STDOUT:   {kind: Branch, arg0: block6},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+2, type: type1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+4, type: type1},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+11, arg1: node+12, type: type1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+4, type: type1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+2, type: type1},
// CHECK:STDOUT:   {kind: BinaryOperatorAdd, arg0: node+14, arg1: node+15, type: type1},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block7, arg1: node+13},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block7, arg1: node+16},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block7, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+19},
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool, %n: i32, %m: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_6: bool = value_binding %b
// CHECK:STDOUT:   if %.loc7_6 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc7_15.1: i32 = value_binding %n
// CHECK:STDOUT:   %.loc7_23.1: i32 = value_binding %m
// CHECK:STDOUT:   %.loc8_22: i32 = add %.loc7_15.1, %.loc7_23.1
// CHECK:STDOUT:   br !if.expr.result(%.loc8_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc7_23.2: i32 = value_binding %m
// CHECK:STDOUT:   %.loc7_15.2: i32 = value_binding %n
// CHECK:STDOUT:   %.loc8_33: i32 = add %.loc7_23.2, %.loc7_15.2
// CHECK:STDOUT:   br !if.expr.result(%.loc8_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc8_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc8_10
// CHECK:STDOUT: }

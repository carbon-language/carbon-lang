// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F(b: bool) -> i32 {
  return if b then A() else B();
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, body: {block2}}},
// CHECK:STDOUT:   {name: str1, param_refs: block0, return_type: type0, body: {block3}}},
// CHECK:STDOUT:   {name: str2, param_refs: block5, return_type: type0, body: {block6, block7, block8, block9}}},
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
// CHECK:STDOUT:   b,
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
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+1, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+2},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+5, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+6},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str3, arg1: node+8, type: type1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+8, type: type1},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block7, arg1: node+11},
// CHECK:STDOUT:   {kind: Branch, arg0: block8},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block9, arg1: node+14},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block9, arg1: node+15},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block9, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+18, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+19},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @A
// CHECK:STDOUT:   %.loc8 = fn_decl @B
// CHECK:STDOUT:   %.loc10 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_24.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_24.2: i32 = initialize_from %.loc7_24.1
// CHECK:STDOUT:   return %.loc7_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_24.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_24.2: i32 = initialize_from %.loc8_24.1
// CHECK:STDOUT:   return %.loc8_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: bool = value_binding %b
// CHECK:STDOUT:   if %.loc10 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc11_21: i32 = call @A()
// CHECK:STDOUT:   br !if.expr.result(%.loc11_21)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc11_30: i32 = call @B()
// CHECK:STDOUT:   br !if.expr.result(%.loc11_30)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10.1: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc11_10.2: i32 = initialize_from %.loc11_10.1
// CHECK:STDOUT:   return %.loc11_10.2
// CHECK:STDOUT: }

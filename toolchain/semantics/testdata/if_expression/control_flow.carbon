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
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, body: [block3]},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type0, body: [block5]},
// CHECK:STDOUT:   {name: str3, param_refs: block7, return_type: type0, body: [block8, block9, block10, block11]},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   A,
// CHECK:STDOUT:   return,
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
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+2},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+6},
// CHECK:STDOUT:   {kind: Parameter, arg0: str4, type: type1},
// CHECK:STDOUT:   {kind: VarStorage, arg0: str1, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block9, arg1: node+8},
// CHECK:STDOUT:   {kind: Branch, arg0: block10},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: TemporaryStorage, type: type0},
// CHECK:STDOUT:   {kind: Temporary, arg0: node+14, arg1: node+13, type: type0},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+15, type: type0},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: TemporaryStorage, type: type0},
// CHECK:STDOUT:   {kind: Temporary, arg0: node+18, arg1: node+17, type: type0},
// CHECK:STDOUT:   {kind: BindValue, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block11, arg1: node+16},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block11, arg1: node+20},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block11, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+23},
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
// CHECK:STDOUT:     node+10,
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
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
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
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc11_21.1: init i32 = call @A()
// CHECK:STDOUT:   %.loc11_21.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_21.3: ref i32 = temporary %.loc11_21.2, %.loc11_21.1
// CHECK:STDOUT:   %.loc11_21.4: i32 = bind_value %.loc11_21.3
// CHECK:STDOUT:   br !if.expr.result(%.loc11_21.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc11_30.1: init i32 = call @B()
// CHECK:STDOUT:   %.loc11_30.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_30.3: ref i32 = temporary %.loc11_30.2, %.loc11_30.1
// CHECK:STDOUT:   %.loc11_30.4: i32 = bind_value %.loc11_30.3
// CHECK:STDOUT:   br !if.expr.result(%.loc11_30.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc11_10
// CHECK:STDOUT: }

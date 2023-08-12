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
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, return_slot: node+0, body: {block3}}},
// CHECK:STDOUT:   {name: str1, param_refs: block0, return_type: type0, return_slot: node+5, body: {block5}}},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type0, return_slot: node+10, body: {block7, block8, block9, block12}}},
// CHECK:STDOUT:   {name: str3, param_refs: block0, return_type: type0, return_slot: node+26, body: {block14, block15, block16, block19}}},
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
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+2, arg1: node+0, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+3},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+7, arg1: node+5, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+8},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type1},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block8, arg1: node+12},
// CHECK:STDOUT:   {kind: Branch, arg0: block9},
// CHECK:STDOUT:   {kind: Call, arg0: block10, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, arg0: node+15, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+16, type: type0},
// CHECK:STDOUT:   {kind: Call, arg0: block11, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, arg0: node+18, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+19, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block12, arg1: node+17},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block12, arg1: node+20},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block12, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+23, arg1: node+10, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+24},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function3},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: false, type: type1},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block15, arg1: node+28},
// CHECK:STDOUT:   {kind: Branch, arg0: block16},
// CHECK:STDOUT:   {kind: Call, arg0: block17, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, arg0: node+31, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+32, type: type0},
// CHECK:STDOUT:   {kind: Call, arg0: block18, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: MaterializeTemporary, arg0: node+34, type: type0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+35, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block19, arg1: node+33},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block19, arg1: node+36},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block19, type: type0},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+39, arg1: node+26, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+40},
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
// CHECK:STDOUT:     node+27,
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
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:     node+29,
// CHECK:STDOUT:     node+30,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+31,
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:     node+33,
// CHECK:STDOUT:     node+37,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+34,
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:     node+36,
// CHECK:STDOUT:     node+38,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+32,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+35,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+39,
// CHECK:STDOUT:     node+40,
// CHECK:STDOUT:     node+41,
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
// CHECK:STDOUT: fn @A() -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_24.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_24.2: i32 = initialize_from %.loc7_24.1 to %return
// CHECK:STDOUT:   return %.loc7_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_24.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_24.2: i32 = initialize_from %.loc8_24.1 to %return
// CHECK:STDOUT:   return %.loc8_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_13: bool = bool_literal true
// CHECK:STDOUT:   if %.loc11_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc11_24.1: i32 = call @A() to %.loc11_24.2
// CHECK:STDOUT:   %.loc11_24.2: i32 = materialize_temporary %.loc11_24.1
// CHECK:STDOUT:   %.loc11_24.3: i32 = value_binding %.loc11_24.2
// CHECK:STDOUT:   br !if.expr.result(%.loc11_24.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc11_33.1: i32 = call @B() to %.loc11_33.2
// CHECK:STDOUT:   %.loc11_33.2: i32 = materialize_temporary %.loc11_33.1
// CHECK:STDOUT:   %.loc11_33.3: i32 = value_binding %.loc11_33.2
// CHECK:STDOUT:   br !if.expr.result(%.loc11_33.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10.1: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc11_10.2: i32 = initialize_from %.loc11_10.1 to %return
// CHECK:STDOUT:   return %.loc11_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal false
// CHECK:STDOUT:   if %.loc15_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc15_25.1: i32 = call @A() to %.loc15_25.2
// CHECK:STDOUT:   %.loc15_25.2: i32 = materialize_temporary %.loc15_25.1
// CHECK:STDOUT:   %.loc15_25.3: i32 = value_binding %.loc15_25.2
// CHECK:STDOUT:   br !if.expr.result(%.loc15_25.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc15_34.1: i32 = call @B() to %.loc15_34.2
// CHECK:STDOUT:   %.loc15_34.2: i32 = materialize_temporary %.loc15_34.1
// CHECK:STDOUT:   %.loc15_34.3: i32 = value_binding %.loc15_34.2
// CHECK:STDOUT:   br !if.expr.result(%.loc15_34.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_10.1: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc15_10.2: i32 = initialize_from %.loc15_10.1 to %return
// CHECK:STDOUT:   return %.loc15_10.2
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn If1(b: bool) -> i32 {
  if (b) {
    return 1;
  } else {
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

fn If2(b: bool) -> i32 {
  if (b) {
  } else {
    return 2;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

fn If3(b: bool) -> i32 {
  if (b) {
    return 1;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, body: {block4, block5, block6, block7}}},
// CHECK:STDOUT:   {name: str2, param_refs: block9, return_type: type1, body: {block10, block11, block12, block13}}},
// CHECK:STDOUT:   {name: str3, param_refs: block15, return_type: type1, body: {block16, block17, block18}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   1,
// CHECK:STDOUT:   2,
// CHECK:STDOUT:   1,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   If1,
// CHECK:STDOUT:   b,
// CHECK:STDOUT:   If2,
// CHECK:STDOUT:   If3,
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
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block5, arg1: node+3},
// CHECK:STDOUT:   {kind: Branch, arg0: block6},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+6, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+7},
// CHECK:STDOUT:   {kind: Branch, arg0: block7},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+10, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+10, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block11, arg1: node+13},
// CHECK:STDOUT:   {kind: Branch, arg0: block12},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int1, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+16, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+17},
// CHECK:STDOUT:   {kind: Branch, arg0: block13},
// CHECK:STDOUT:   {kind: VarStorage, type: type0},
// CHECK:STDOUT:   {kind: BindName, arg0: str1, arg1: node+20, type: type0},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+20, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block17, arg1: node+23},
// CHECK:STDOUT:   {kind: Branch, arg0: block18},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int2, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+26, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+27},
// CHECK:STDOUT:   {kind: Branch, arg0: block18},
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
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+22,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:     node+15,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+19,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:     node+18,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+20,
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+21,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+23,
// CHECK:STDOUT:     node+24,
// CHECK:STDOUT:     node+25,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+26,
// CHECK:STDOUT:     node+27,
// CHECK:STDOUT:     node+28,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @If1
// CHECK:STDOUT:   %.loc17 = fn_decl @If2
// CHECK:STDOUT:   %.loc27 = fn_decl @If3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If1(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: bool = value_binding %b
// CHECK:STDOUT:   if %.loc7 br !.loc9 else br !.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9:
// CHECK:STDOUT:   %.loc9_12.1: i32 = int_value 1
// CHECK:STDOUT:   %.loc9_12.2: i32 = initialize_from %.loc9_12.1
// CHECK:STDOUT:   return %.loc9_12.2
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc8:
// CHECK:STDOUT:   br !.1
// CHECK:STDOUT:
// CHECK:STDOUT: !.1:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If2(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17: bool = value_binding %b
// CHECK:STDOUT:   if %.loc17 br !.loc18 else br !.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc18:
// CHECK:STDOUT:   br !.1
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20:
// CHECK:STDOUT:   %.loc20_12.1: i32 = int_value 2
// CHECK:STDOUT:   %.loc20_12.2: i32 = initialize_from %.loc20_12.1
// CHECK:STDOUT:   return %.loc20_12.2
// CHECK:STDOUT:
// CHECK:STDOUT: !.1:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If3(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27: bool = value_binding %b
// CHECK:STDOUT:   if %.loc27 br !.loc29 else br !.1
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc29:
// CHECK:STDOUT:   %.loc29_12.1: i32 = int_value 1
// CHECK:STDOUT:   %.loc29_12.2: i32 = initialize_from %.loc29_12.1
// CHECK:STDOUT:   return %.loc29_12.2
// CHECK:STDOUT:
// CHECK:STDOUT: !.1:
// CHECK:STDOUT: }

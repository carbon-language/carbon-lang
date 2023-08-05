// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block0, return_type: type0, body: {block2}}},
// CHECK:STDOUT:   {name: str1, param_refs: block0, return_type: type0, body: {block3}}},
// CHECK:STDOUT:   {name: str2, param_refs: block0, return_type: type0, body: {block4, block5, block6}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   F,
// CHECK:STDOUT:   G,
// CHECK:STDOUT:   And,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   nodeBoolType,
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+1},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function1},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: true, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+4},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: function2},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function0, type: type0},
// CHECK:STDOUT:   {kind: BoolLiteral, arg0: false, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block5, arg1: node+7},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block6, arg1: node+8},
// CHECK:STDOUT:   {kind: Call, arg0: block0, arg1: function1, type: type0},
// CHECK:STDOUT:   {kind: BranchWithArg, arg0: block6, arg1: node+11},
// CHECK:STDOUT:   {kind: BlockArg, arg0: block6, type: type0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+13},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:     node+6,
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
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %108 = fn_decl @F
// CHECK:STDOUT:   %109 = fn_decl @G
// CHECK:STDOUT:   %111 = fn_decl @And
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %108: bool = bool_value true
// CHECK:STDOUT:   return %108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %109: bool = bool_value true
// CHECK:STDOUT:   return %109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %112: bool = call @F()
// CHECK:STDOUT:   %112.14: bool = bool_value false
// CHECK:STDOUT:   if %112 br !112 else br !112.14(%112.14)
// CHECK:STDOUT:
// CHECK:STDOUT: !112:
// CHECK:STDOUT:   %112.19: bool = call @G()
// CHECK:STDOUT:   br !112.14(%112.19)
// CHECK:STDOUT:
// CHECK:STDOUT: !112.14:
// CHECK:STDOUT:   %112.14.0: bool = block_arg !112.14
// CHECK:STDOUT:   return %112.14.0
// CHECK:STDOUT: }

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

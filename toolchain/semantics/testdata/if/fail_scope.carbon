// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn VarScope(b: bool) -> i32 {
  if (b) {
    var n: i32 = 2;
    return n;
  }
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: Name `n` not found
  // CHECK:STDERR:   return n;
  // CHECK:STDERR:          ^
  return n;
}

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT:   {name: str0, param_refs: block2, return_type: type1, return_slot: node+2, body: {block5, block6, block7}}},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT:   2,
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   VarScope,
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
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+0, type: type0},
// CHECK:STDOUT:   {kind: BranchIf, arg0: block6, arg1: node+4},
// CHECK:STDOUT:   {kind: Branch, arg0: block7},
// CHECK:STDOUT:   {kind: VarStorage, type: type1},
// CHECK:STDOUT:   {kind: BindName, arg0: str2, arg1: node+7, type: type1},
// CHECK:STDOUT:   {kind: IntegerLiteral, arg0: int0, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+9, arg1: node+7, type: type1},
// CHECK:STDOUT:   {kind: Assign, arg0: node+7, arg1: node+10},
// CHECK:STDOUT:   {kind: ValueBinding, arg0: node+7, type: type1},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: node+12, arg1: node+2, type: type1},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+13},
// CHECK:STDOUT:   {kind: Branch, arg0: block7},
// CHECK:STDOUT:   {kind: InitializeFrom, arg0: nodeError, arg1: node+2, type: typeError},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+16},
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
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+16,
// CHECK:STDOUT:     node+17,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]
// CHECK:STDOUT:
// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @VarScope
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @VarScope(%b: bool) -> %return: i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: bool = value_binding %b
// CHECK:STDOUT:   if %.loc7 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n: i32 = var
// CHECK:STDOUT:   %.loc9_18.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc9_18.2: i32 = initialize_from %.loc9_18.1, %n
// CHECK:STDOUT:   assign %n, %.loc9_18.2
// CHECK:STDOUT:   %.loc9_9.1: i32 = value_binding %n
// CHECK:STDOUT:   %.loc9_9.2: i32 = initialize_from %.loc9_9.1, %return
// CHECK:STDOUT:   return %.loc9_9.2
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.1: <error> = initialize_from <error>, %return
// CHECK:STDOUT:   return %.1
// CHECK:STDOUT: }

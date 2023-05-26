// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: callables: [
// CHECK:STDOUT:   {param_refs: block2, return_type: node+4},
// CHECK:STDOUT:   {param_refs: block0},
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   a,
// CHECK:STDOUT:   Echo,
// CHECK:STDOUT:   Main,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT:   node+0,
// CHECK:STDOUT:   node+4,
// CHECK:STDOUT:   node+10,
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: StructType, arg0: block0, type: nodeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: node+0},
// CHECK:STDOUT:   {kind: VarStorage, type: node+0},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+2, type: node+0},
// CHECK:STDOUT:   {kind: StructType, arg0: block0, type: nodeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: node+4},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str1, arg1: callable0},
// CHECK:STDOUT:   {kind: ReturnExpression, arg0: node+2, type: node+0},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+6, arg1: block4},
// CHECK:STDOUT:   {kind: FunctionDeclaration, arg0: str2, arg1: callable1},
// CHECK:STDOUT:   {kind: StructType, arg0: block0, type: nodeTypeType},
// CHECK:STDOUT:   {kind: StructValue, arg0: block0, type: node+10},
// CHECK:STDOUT:   {kind: StubReference, arg0: node+11, type: node+10},
// CHECK:STDOUT:   {kind: Call, arg0: block6, arg1: callable0, type: node+4},
// CHECK:STDOUT:   {kind: FunctionDefinition, arg0: node+9, arg1: block5},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+3,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+4,
// CHECK:STDOUT:     node+5,
// CHECK:STDOUT:     node+6,
// CHECK:STDOUT:     node+8,
// CHECK:STDOUT:     node+9,
// CHECK:STDOUT:     node+14,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+7,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+10,
// CHECK:STDOUT:     node+11,
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:     node+13,
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+12,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

fn Echo(a: {}) -> {} {
  return a;
}

fn Main() {
  Echo({});
}

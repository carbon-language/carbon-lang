// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-sem-ir --dump-raw-sem-ir %s
//
// Check that we can combine textual IR and raw IR dumping in one compile.
//
// AUTOUPDATE

// --- a.carbon
fn A() {}

// --- b.carbon
fn B() {}

// CHECK:STDOUT: - filename: a.carbon
// CHECK:STDOUT:   sem_ir:
// CHECK:STDOUT:   - cross_reference_irs_size: 1
// CHECK:STDOUT:     functions: [
// CHECK:STDOUT:       {name: str0, param_refs: block0, body: [block1]},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     classes: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     integers: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     reals: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     strings: [
// CHECK:STDOUT:       A,
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     types: [
// CHECK:STDOUT:       {node: nodeFunctionType, value_rep: {kind: copy, type: type0}},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     type_blocks: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     nodes: [
// CHECK:STDOUT:       {kind: FunctionDeclaration, arg0: function0, type: type0},
// CHECK:STDOUT:       {kind: Return},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     node_blocks: [
// CHECK:STDOUT:       [
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+1,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+0,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:     ]
// CHECK:STDOUT:
// CHECK:STDOUT: file "a.carbon" {
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT: - filename: b.carbon
// CHECK:STDOUT:   sem_ir:
// CHECK:STDOUT:   - cross_reference_irs_size: 1
// CHECK:STDOUT:     functions: [
// CHECK:STDOUT:       {name: str0, param_refs: block0, body: [block1]},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     classes: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     integers: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     reals: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     strings: [
// CHECK:STDOUT:       B,
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     types: [
// CHECK:STDOUT:       {node: nodeFunctionType, value_rep: {kind: copy, type: type0}},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     type_blocks: [
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     nodes: [
// CHECK:STDOUT:       {kind: FunctionDeclaration, arg0: function0, type: type0},
// CHECK:STDOUT:       {kind: Return},
// CHECK:STDOUT:     ]
// CHECK:STDOUT:     node_blocks: [
// CHECK:STDOUT:       [
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+1,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:       [
// CHECK:STDOUT:         node+0,
// CHECK:STDOUT:       ],
// CHECK:STDOUT:     ]
// CHECK:STDOUT:
// CHECK:STDOUT: file "b.carbon" {
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

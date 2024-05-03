// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import  --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE

// --- a.carbon
package A api;

fn A() {}

// --- b.carbon
package B api;

fn B() {}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        a.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+1}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, enclosing_scope: name_scope0, param_refs: empty, body: [block3]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template inst+2, value_rep: {kind: none, type: type2}}
// CHECK:STDOUT:     type2:           {constant: template inst+3, value_rep: {kind: none, type: type2}}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     typeBlock0:      {}
// CHECK:STDOUT:     typeBlock1:      {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     'inst+1':          {kind: FunctionDecl, arg0: function0, arg1: empty, type: type1}
// CHECK:STDOUT:     'inst+2':          {kind: FunctionType, arg0: function0, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: TupleType, arg0: typeBlock0, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: StructValue, arg0: empty, type: type1}
// CHECK:STDOUT:     'inst+5':          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          template inst+0
// CHECK:STDOUT:     'inst+1':          template inst+4
// CHECK:STDOUT:     'inst+2':          template inst+2
// CHECK:STDOUT:     'inst+3':          template inst+3
// CHECK:STDOUT:     'inst+4':          template inst+4
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+1}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, enclosing_scope: name_scope0, param_refs: empty, body: [block3]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template inst+2, value_rep: {kind: none, type: type2}}
// CHECK:STDOUT:     type2:           {constant: template inst+3, value_rep: {kind: none, type: type2}}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     typeBlock0:      {}
// CHECK:STDOUT:     typeBlock1:      {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     'inst+1':          {kind: FunctionDecl, arg0: function0, arg1: empty, type: type1}
// CHECK:STDOUT:     'inst+2':          {kind: FunctionType, arg0: function0, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: TupleType, arg0: typeBlock0, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: StructValue, arg0: empty, type: type1}
// CHECK:STDOUT:     'inst+5':          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          template inst+0
// CHECK:STDOUT:     'inst+1':          template inst+4
// CHECK:STDOUT:     'inst+2':          template inst+2
// CHECK:STDOUT:     'inst+3':          template inst+3
// CHECK:STDOUT:     'inst+4':          template inst+4
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT: ...

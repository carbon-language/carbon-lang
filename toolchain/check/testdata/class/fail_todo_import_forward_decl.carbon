// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// ARGS: compile --dump-raw-sem-ir --phase=check %s

// --- a.carbon

library "a" api;

class ForwardDecl;

// --- b.carbon

library "b" api;

import library "a";

// TODO: This should probably have a valid form.
// CHECK:STDERR: b.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class ForwardDecl {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: a.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class ForwardDecl;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
class ForwardDecl {
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        a.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name1: inst+1}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class0:          {name: name1, enclosing_scope: name_scope0}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template inst+2, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst+0:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     inst+1:          {kind: ClassDecl, arg0: class0, arg1: empty}
// CHECK:STDOUT:     inst+2:          {kind: ClassType, arg0: class0, type: typeTypeType}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:     inst+2:          template inst+2
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 2
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name2: inst+1}}
// CHECK:STDOUT:     name_scope1:     {inst: inst+4, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class0:          {name: name2, enclosing_scope: name_scope<invalid>}
// CHECK:STDOUT:     class1:          {name: name<invalid>, enclosing_scope: name_scope<invalid>}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template inst+3, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type2:           {constant: template inst+5, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type3:           {constant: template inst+6, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst+0:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     inst+1:          {kind: ImportRefUsed, arg0: ir1, arg1: inst+1}
// CHECK:STDOUT:     inst+2:          {kind: ClassDecl, arg0: class0, arg1: empty}
// CHECK:STDOUT:     inst+3:          {kind: ClassType, arg0: class0, type: typeTypeType}
// CHECK:STDOUT:     inst+4:          {kind: ClassDecl, arg0: class1, arg1: block3}
// CHECK:STDOUT:     inst+5:          {kind: ClassType, arg0: class1, type: typeTypeType}
// CHECK:STDOUT:     inst+6:          {kind: StructType, arg0: block5, type: typeTypeType}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:     inst+1:          template inst+3
// CHECK:STDOUT:     inst+3:          template inst+3
// CHECK:STDOUT:     inst+5:          template inst+5
// CHECK:STDOUT:     inst+6:          template inst+6
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block4:          {}
// CHECK:STDOUT:     block5:          {}
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT:       2:               inst+4
// CHECK:STDOUT: ...

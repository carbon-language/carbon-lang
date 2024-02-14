// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --dump-raw-sem-ir --phase=check %s
// AUTOUPDATE

// --- a.carbon

library "a" api;

class Cycle {
  var a: Cycle*;
}

// --- b.carbon

library "b" api;

import library "a";

fn Run() {
  var a: Cycle*;
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        a.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name1: inst+1}}
// CHECK:STDOUT:     name_scope1:     {inst: inst+1, enclosing_scope: name_scope0, has_error: false, extended_scopes: [], names: {name0: inst+7}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class0:          {name: name1, enclosing_scope: name_scope0}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template inst+2, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type2:           {constant: template inst+5, value_rep: {kind: copy, type: type2}}
// CHECK:STDOUT:     type3:           {constant: template inst+6, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type4:           {constant: template inst+10, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst+0:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     inst+1:          {kind: ClassDecl, arg0: class0, arg1: empty}
// CHECK:STDOUT:     inst+2:          {kind: ClassType, arg0: class0, type: typeTypeType}
// CHECK:STDOUT:     inst+3:          {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     inst+4:          {kind: PointerType, arg0: type1, type: typeTypeType}
// CHECK:STDOUT:     inst+5:          {kind: PointerType, arg0: type1, type: typeTypeType}
// CHECK:STDOUT:     inst+6:          {kind: UnboundElementType, arg0: type1, arg1: type2, type: typeTypeType}
// CHECK:STDOUT:     inst+7:          {kind: FieldDecl, arg0: name0, arg1: element0, type: type3}
// CHECK:STDOUT:     inst+8:          {kind: StructTypeField, arg0: name0, arg1: type2}
// CHECK:STDOUT:     inst+9:          {kind: StructTypeField, arg0: name0, arg1: type2}
// CHECK:STDOUT:     inst+10:         {kind: StructType, arg0: block5, type: typeTypeType}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:     inst+2:          template inst+2
// CHECK:STDOUT:     inst+3:          template inst+2
// CHECK:STDOUT:     inst+4:          template inst+5
// CHECK:STDOUT:     inst+5:          template inst+5
// CHECK:STDOUT:     inst+6:          template inst+6
// CHECK:STDOUT:     inst+7:          template inst+7
// CHECK:STDOUT:     inst+8:          template inst+9
// CHECK:STDOUT:     inst+9:          template inst+9
// CHECK:STDOUT:     inst+10:         template inst+10
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:       1:               inst+4
// CHECK:STDOUT:       2:               inst+7
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+8
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+9
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 2
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name2: inst+2, name3: inst+1}}
// CHECK:STDOUT:     name_scope1:     {inst: inst+3, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name1: inst+9}}
// CHECK:STDOUT:   bind_names:
// CHECK:STDOUT:     bindName0:       {name: name1, enclosing_scope: name_scope<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name2, enclosing_scope: name_scope0, param_refs: empty, body: [block3]}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class0:          {name: name3, enclosing_scope: name_scope<invalid>}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template instFunctionType, value_rep: {kind: copy, type: type1}}
// CHECK:STDOUT:     type2:           {constant: template inst+4, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type3:           {constant: template inst+5, value_rep: {kind: copy, type: type3}}
// CHECK:STDOUT:     type4:           {constant: template inst+8, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst+0:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     inst+1:          {kind: ImportRefUsed, arg0: ir1, arg1: inst+1}
// CHECK:STDOUT:     inst+2:          {kind: FunctionDecl, arg0: function0, type: type1}
// CHECK:STDOUT:     inst+3:          {kind: ClassDecl, arg0: class0, arg1: empty}
// CHECK:STDOUT:     inst+4:          {kind: ClassType, arg0: class0, type: typeTypeType}
// CHECK:STDOUT:     inst+5:          {kind: PointerType, arg0: type2, type: typeTypeType}
// CHECK:STDOUT:     inst+6:          {kind: StructTypeField, arg0: name1, arg1: type3}
// CHECK:STDOUT:     inst+7:          {kind: StructTypeField, arg0: name1, arg1: type3}
// CHECK:STDOUT:     inst+8:          {kind: StructType, arg0: block5, type: typeTypeType}
// CHECK:STDOUT:     inst+9:          {kind: ImportRefUnused, arg0: ir1, arg1: inst+7}
// CHECK:STDOUT:     inst+10:         {kind: NameRef, arg0: name3, arg1: inst+4, type: typeTypeType}
// CHECK:STDOUT:     inst+11:         {kind: PointerType, arg0: type2, type: typeTypeType}
// CHECK:STDOUT:     inst+12:         {kind: VarStorage, arg0: name1, type: type3}
// CHECK:STDOUT:     inst+13:         {kind: BindName, arg0: bindName0, arg1: inst+12, type: type3}
// CHECK:STDOUT:     inst+14:         {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:     inst+1:          template inst+4
// CHECK:STDOUT:     inst+2:          template inst+2
// CHECK:STDOUT:     inst+4:          template inst+4
// CHECK:STDOUT:     inst+5:          template inst+5
// CHECK:STDOUT:     inst+6:          template inst+7
// CHECK:STDOUT:     inst+7:          template inst+7
// CHECK:STDOUT:     inst+8:          template inst+8
// CHECK:STDOUT:     inst+9:          <invalid>
// CHECK:STDOUT:     inst+10:         template inst+4
// CHECK:STDOUT:     inst+11:         template inst+5
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:       1:               inst+10
// CHECK:STDOUT:       2:               inst+11
// CHECK:STDOUT:       3:               inst+12
// CHECK:STDOUT:       4:               inst+13
// CHECK:STDOUT:       5:               inst+14
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+6
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+7
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+9
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT:       2:               inst+2
// CHECK:STDOUT: ...

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import  --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/multifile_raw_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/multifile_raw_ir.carbon

// --- a.carbon
package A;

fn A() {}

// --- b.carbon
package B;

import A;

fn B() {
  A.A();
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        a.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<invalid>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+1}}
// CHECK:STDOUT:   entity_names:    {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, body: [block4]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:     'type(inst+2)':    {kind: none, type: type(inst+3)}
// CHECK:STDOUT:     'type(inst+3)':    {kind: none, type: type(inst+3)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+1':          {kind: FunctionDecl, arg0: function0, arg1: empty, type: type(inst+2)}
// CHECK:STDOUT:     'inst+2':          {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: StructValue, arg0: empty, type: type(inst+2)}
// CHECK:STDOUT:     'inst+5':          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:     'inst+1':          templateConstant(inst+4)
// CHECK:STDOUT:     'inst+2':          templateConstant(inst+2)
// CHECK:STDOUT:     'inst+3':          templateConstant(inst+3)
// CHECK:STDOUT:     'inst+4':          templateConstant(inst+4)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<invalid>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst+1, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir1, inst_id: inst+1}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: ir1, inst_id: inst+1}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name1: inst+2, name0: inst+3}}
// CHECK:STDOUT:     name_scope1:     {inst: inst+2, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name1: inst+8}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope1, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, body: [block4]}
// CHECK:STDOUT:     function1:       {name: name1, parent_scope: name_scope1}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:     'type(inst+4)':    {kind: none, type: type(inst+5)}
// CHECK:STDOUT:     'type(inst+5)':    {kind: none, type: type(inst+5)}
// CHECK:STDOUT:     'type(inst+10)':   {kind: none, type: type(inst+5)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+1':          {kind: ImportDecl, arg0: name1}
// CHECK:STDOUT:     'inst+2':          {kind: Namespace, arg0: name_scope1, arg1: inst+1, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+3':          {kind: FunctionDecl, arg0: function0, arg1: empty, type: type(inst+4)}
// CHECK:STDOUT:     'inst+4':          {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+5':          {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+6':          {kind: StructValue, arg0: empty, type: type(inst+4)}
// CHECK:STDOUT:     'inst+7':          {kind: NameRef, arg0: name1, arg1: inst+2, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+8':          {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name0, type: type(inst+10)}
// CHECK:STDOUT:     'inst+9':          {kind: FunctionDecl, arg0: function1, arg1: empty, type: type(inst+10)}
// CHECK:STDOUT:     'inst+10':         {kind: FunctionType, arg0: function1, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+11':         {kind: StructValue, arg0: empty, type: type(inst+10)}
// CHECK:STDOUT:     'inst+12':         {kind: NameRef, arg0: name1, arg1: inst+8, type: type(inst+10)}
// CHECK:STDOUT:     'inst+13':         {kind: Call, arg0: inst+12, arg1: empty, type: type(inst+5)}
// CHECK:STDOUT:     'inst+14':         {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:     'inst+2':          templateConstant(inst+2)
// CHECK:STDOUT:     'inst+3':          templateConstant(inst+6)
// CHECK:STDOUT:     'inst+4':          templateConstant(inst+4)
// CHECK:STDOUT:     'inst+5':          templateConstant(inst+5)
// CHECK:STDOUT:     'inst+6':          templateConstant(inst+6)
// CHECK:STDOUT:     'inst+7':          templateConstant(inst+2)
// CHECK:STDOUT:     'inst+8':          templateConstant(inst+11)
// CHECK:STDOUT:     'inst+9':          templateConstant(inst+11)
// CHECK:STDOUT:     'inst+10':         templateConstant(inst+10)
// CHECK:STDOUT:     'inst+11':         templateConstant(inst+11)
// CHECK:STDOUT:     'inst+12':         templateConstant(inst+11)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:     import_refs:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:       1:               inst+8
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+7
// CHECK:STDOUT:       1:               inst+12
// CHECK:STDOUT:       2:               inst+13
// CHECK:STDOUT:       3:               inst+14
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+1
// CHECK:STDOUT:       2:               inst+3
// CHECK:STDOUT: ...

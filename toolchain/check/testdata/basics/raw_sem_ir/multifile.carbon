// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --no-dump-sem-ir
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/multifile.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/multifile.carbon

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
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst16, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst17}}
// CHECK:STDOUT:   entity_names:    {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block_empty, body: [inst_block5]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst18)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst19)}
// CHECK:STDOUT:     'type(inst19)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst19)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst16:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst17:          {kind: FunctionDecl, arg0: function0, arg1: inst_block_empty, type: type(inst18)}
// CHECK:STDOUT:     inst18:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst19:          {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst20:          {kind: StructValue, arg0: inst_block_empty, type: type(inst18)}
// CHECK:STDOUT:     inst21:          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst16:          concrete_constant(inst16)
// CHECK:STDOUT:       inst17:          concrete_constant(inst20)
// CHECK:STDOUT:       inst18:          concrete_constant(inst18)
// CHECK:STDOUT:       inst19:          concrete_constant(inst19)
// CHECK:STDOUT:       inst20:          concrete_constant(inst20)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:       1:               inst17
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir2:             {decl_id: inst17, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir2, inst_id: inst17}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: ir2, inst_id: inst17}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst16, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst18, name0: inst19}}
// CHECK:STDOUT:     name_scope1:     {inst: inst18, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name1: inst24}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope1, index: -1, is_template: 0}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block_empty, body: [inst_block5]}
// CHECK:STDOUT:     function1:       {name: name1, parent_scope: name_scope1}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst20)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst21)}
// CHECK:STDOUT:     'type(inst21)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst21)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst16:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst17:          {kind: ImportDecl, arg0: name1}
// CHECK:STDOUT:     inst18:          {kind: Namespace, arg0: name_scope1, arg1: inst17, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst19:          {kind: FunctionDecl, arg0: function0, arg1: inst_block_empty, type: type(inst20)}
// CHECK:STDOUT:     inst20:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst21:          {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst22:          {kind: StructValue, arg0: inst_block_empty, type: type(inst20)}
// CHECK:STDOUT:     inst23:          {kind: NameRef, arg0: name1, arg1: inst18, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst24:          {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name0, type: type(inst26)}
// CHECK:STDOUT:     inst25:          {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(inst26)}
// CHECK:STDOUT:     inst26:          {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst27:          {kind: StructValue, arg0: inst_block_empty, type: type(inst26)}
// CHECK:STDOUT:     inst28:          {kind: NameRef, arg0: name1, arg1: inst24, type: type(inst26)}
// CHECK:STDOUT:     inst29:          {kind: Call, arg0: inst28, arg1: inst_block_empty, type: type(inst21)}
// CHECK:STDOUT:     inst30:          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst16:          concrete_constant(inst16)
// CHECK:STDOUT:       inst18:          concrete_constant(inst18)
// CHECK:STDOUT:       inst19:          concrete_constant(inst22)
// CHECK:STDOUT:       inst20:          concrete_constant(inst20)
// CHECK:STDOUT:       inst21:          concrete_constant(inst21)
// CHECK:STDOUT:       inst22:          concrete_constant(inst22)
// CHECK:STDOUT:       inst23:          concrete_constant(inst18)
// CHECK:STDOUT:       inst24:          concrete_constant(inst27)
// CHECK:STDOUT:       inst25:          concrete_constant(inst27)
// CHECK:STDOUT:       inst26:          concrete_constant(inst26)
// CHECK:STDOUT:       inst27:          concrete_constant(inst27)
// CHECK:STDOUT:       inst28:          concrete_constant(inst27)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst19
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:       1:               inst24
// CHECK:STDOUT:       2:               inst25
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst23
// CHECK:STDOUT:       1:               inst28
// CHECK:STDOUT:       2:               inst29
// CHECK:STDOUT:       3:               inst30
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:       1:               inst17
// CHECK:STDOUT:       2:               inst19
// CHECK:STDOUT: ...

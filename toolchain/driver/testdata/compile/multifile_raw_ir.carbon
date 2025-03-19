// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-prelude-import  --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/multifile_raw_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/multifile_raw_ir.carbon

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
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst13, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst14}}
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
// CHECK:STDOUT:     'type(inst15)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst16)}
// CHECK:STDOUT:     'type(inst16)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst16)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst13:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst14:          {kind: FunctionDecl, arg0: function0, arg1: inst_block_empty, type: type(inst15)}
// CHECK:STDOUT:     inst15:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst16:          {kind: TupleType, arg0: type_block0, type: type(TypeType)}
// CHECK:STDOUT:     inst17:          {kind: StructValue, arg0: inst_block_empty, type: type(inst15)}
// CHECK:STDOUT:     inst18:          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst13:          concrete_constant(inst13)
// CHECK:STDOUT:     inst14:          concrete_constant(inst17)
// CHECK:STDOUT:     inst15:          concrete_constant(inst15)
// CHECK:STDOUT:     inst16:          concrete_constant(inst16)
// CHECK:STDOUT:     inst17:          concrete_constant(inst17)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst13
// CHECK:STDOUT:       1:               inst14
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst14, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir1, inst_id: inst14}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: ir1, inst_id: inst14}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst13, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst15, name0: inst16}}
// CHECK:STDOUT:     name_scope1:     {inst: inst15, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name1: inst21}}
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
// CHECK:STDOUT:     'type(inst17)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst18)}
// CHECK:STDOUT:     'type(inst18)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst18)}
// CHECK:STDOUT:     'type(inst23)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst18)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst13:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst14:          {kind: ImportDecl, arg0: name1}
// CHECK:STDOUT:     inst15:          {kind: Namespace, arg0: name_scope1, arg1: inst14, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst16:          {kind: FunctionDecl, arg0: function0, arg1: inst_block_empty, type: type(inst17)}
// CHECK:STDOUT:     inst17:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst18:          {kind: TupleType, arg0: type_block0, type: type(TypeType)}
// CHECK:STDOUT:     inst19:          {kind: StructValue, arg0: inst_block_empty, type: type(inst17)}
// CHECK:STDOUT:     inst20:          {kind: NameRef, arg0: name1, arg1: inst15, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst21:          {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name0, type: type(inst23)}
// CHECK:STDOUT:     inst22:          {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(inst23)}
// CHECK:STDOUT:     inst23:          {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst24:          {kind: StructValue, arg0: inst_block_empty, type: type(inst23)}
// CHECK:STDOUT:     inst25:          {kind: NameRef, arg0: name1, arg1: inst21, type: type(inst23)}
// CHECK:STDOUT:     inst26:          {kind: Call, arg0: inst25, arg1: inst_block_empty, type: type(inst18)}
// CHECK:STDOUT:     inst27:          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst13:          concrete_constant(inst13)
// CHECK:STDOUT:     inst15:          concrete_constant(inst15)
// CHECK:STDOUT:     inst16:          concrete_constant(inst19)
// CHECK:STDOUT:     inst17:          concrete_constant(inst17)
// CHECK:STDOUT:     inst18:          concrete_constant(inst18)
// CHECK:STDOUT:     inst19:          concrete_constant(inst19)
// CHECK:STDOUT:     inst20:          concrete_constant(inst15)
// CHECK:STDOUT:     inst21:          concrete_constant(inst24)
// CHECK:STDOUT:     inst22:          concrete_constant(inst24)
// CHECK:STDOUT:     inst23:          concrete_constant(inst23)
// CHECK:STDOUT:     inst24:          concrete_constant(inst24)
// CHECK:STDOUT:     inst25:          concrete_constant(inst24)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:     import_refs:
// CHECK:STDOUT:       0:               inst15
// CHECK:STDOUT:       1:               inst21
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst20
// CHECK:STDOUT:       1:               inst25
// CHECK:STDOUT:       2:               inst26
// CHECK:STDOUT:       3:               inst27
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst13
// CHECK:STDOUT:       1:               inst14
// CHECK:STDOUT:       2:               inst16
// CHECK:STDOUT: ...

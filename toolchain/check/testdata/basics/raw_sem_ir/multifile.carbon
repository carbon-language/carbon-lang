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
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst1610612751}}
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
// CHECK:STDOUT:     'type(inst1610612752)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst1610612753)}
// CHECK:STDOUT:     'type(inst1610612753)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst1610612753)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst1610612751:  {kind: FunctionDecl, arg0: function0, arg1: inst_block_empty, type: type(inst1610612752)}
// CHECK:STDOUT:     inst1610612752:  {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612753:  {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612754:  {kind: StructValue, arg0: inst_block_empty, type: type(inst1610612752)}
// CHECK:STDOUT:     inst1610612755:  {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst1610612751:  concrete_constant(inst1610612754)
// CHECK:STDOUT:       inst1610612752:  concrete_constant(inst1610612752)
// CHECK:STDOUT:       inst1610612753:  concrete_constant(inst1610612753)
// CHECK:STDOUT:       inst1610612754:  concrete_constant(inst1610612754)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst1610612751
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst1610612755
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst1610612751
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir2:             {decl_id: inst1342177295, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir2, inst_id: inst1610612751}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: ir2, inst_id: inst1610612751}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst1342177296, name0: inst1342177297}}
// CHECK:STDOUT:     name_scope1:     {inst: inst1342177296, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name1: inst1342177302}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope1, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
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
// CHECK:STDOUT:     'type(inst1342177298)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst1342177299)}
// CHECK:STDOUT:     'type(inst1342177299)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst1342177299)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst1342177295:  {kind: ImportDecl, arg0: name1}
// CHECK:STDOUT:     inst1342177296:  {kind: Namespace, arg0: name_scope1, arg1: inst1342177295, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst1342177297:  {kind: FunctionDecl, arg0: function0, arg1: inst_block_empty, type: type(inst1342177298)}
// CHECK:STDOUT:     inst1342177298:  {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1342177299:  {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst1342177300:  {kind: StructValue, arg0: inst_block_empty, type: type(inst1342177298)}
// CHECK:STDOUT:     inst1342177301:  {kind: NameRef, arg0: name1, arg1: inst1342177296, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst1342177302:  {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name0, type: type(inst1342177304)}
// CHECK:STDOUT:     inst1342177303:  {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(inst1342177304)}
// CHECK:STDOUT:     inst1342177304:  {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1342177305:  {kind: StructValue, arg0: inst_block_empty, type: type(inst1342177304)}
// CHECK:STDOUT:     inst1342177306:  {kind: NameRef, arg0: name1, arg1: inst1342177302, type: type(inst1342177304)}
// CHECK:STDOUT:     inst1342177307:  {kind: Call, arg0: inst1342177306, arg1: inst_block_empty, type: type(inst1342177299)}
// CHECK:STDOUT:     inst1342177308:  {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst1342177296:  concrete_constant(inst1342177296)
// CHECK:STDOUT:       inst1342177297:  concrete_constant(inst1342177300)
// CHECK:STDOUT:       inst1342177298:  concrete_constant(inst1342177298)
// CHECK:STDOUT:       inst1342177299:  concrete_constant(inst1342177299)
// CHECK:STDOUT:       inst1342177300:  concrete_constant(inst1342177300)
// CHECK:STDOUT:       inst1342177301:  concrete_constant(inst1342177296)
// CHECK:STDOUT:       inst1342177302:  concrete_constant(inst1342177305)
// CHECK:STDOUT:       inst1342177303:  concrete_constant(inst1342177305)
// CHECK:STDOUT:       inst1342177304:  concrete_constant(inst1342177304)
// CHECK:STDOUT:       inst1342177305:  concrete_constant(inst1342177305)
// CHECK:STDOUT:       inst1342177306:  concrete_constant(inst1342177305)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst1342177297
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst1342177296
// CHECK:STDOUT:       1:               inst1342177302
// CHECK:STDOUT:       2:               inst1342177303
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst1342177301
// CHECK:STDOUT:       1:               inst1342177306
// CHECK:STDOUT:       2:               inst1342177307
// CHECK:STDOUT:       3:               inst1342177308
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst1342177295
// CHECK:STDOUT:       2:               inst1342177297
// CHECK:STDOUT: ...

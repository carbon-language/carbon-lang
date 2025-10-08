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
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst0000000E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst6000000F}}
// CHECK:STDOUT:   entity_names:    {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function60000000: {name: name0, parent_scope: name_scope0, call_params_id: inst_block_empty, body: [inst_block5]}
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
// CHECK:STDOUT:     'type(inst60000010)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000011)}
// CHECK:STDOUT:     'type(inst60000011)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000011)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst0000000E:    {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000000F:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000010:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000011:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000013:    {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst0000000E:    concrete_constant(inst0000000E)
// CHECK:STDOUT:       inst6000000F:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000010:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000012:    concrete_constant(inst60000012)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst6000000F
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst60000013
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst0000000E
// CHECK:STDOUT:       1:               inst6000000F
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        b.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir2:      {decl_id: inst5000000F, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir2, inst_id: inst6000000F}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir2, inst_id: inst6000000F}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst0000000E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst50000010, name0: inst50000011}}
// CHECK:STDOUT:     name_scope1:     {inst: inst50000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name1: inst50000016}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope1, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function50000000: {name: name0, parent_scope: name_scope0, call_params_id: inst_block_empty, body: [inst_block5]}
// CHECK:STDOUT:     function50000001: {name: name1, parent_scope: name_scope1}
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
// CHECK:STDOUT:     'type(inst50000012)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000013)}
// CHECK:STDOUT:     'type(inst50000013)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000013)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst0000000E:    {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst5000000F:    {kind: ImportDecl, arg0: name1}
// CHECK:STDOUT:     inst50000010:    {kind: Namespace, arg0: name_scope1, arg1: inst5000000F, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000011:    {kind: FunctionDecl, arg0: function50000000, arg1: inst_block_empty, type: type(inst50000012)}
// CHECK:STDOUT:     inst50000012:    {kind: FunctionType, arg0: function50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000013:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000014:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000012)}
// CHECK:STDOUT:     inst50000015:    {kind: NameRef, arg0: name1, arg1: inst50000010, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000016:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name0, type: type(inst50000018)}
// CHECK:STDOUT:     inst50000017:    {kind: FunctionDecl, arg0: function50000001, arg1: inst_block_empty, type: type(inst50000018)}
// CHECK:STDOUT:     inst50000018:    {kind: FunctionType, arg0: function50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000019:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000018)}
// CHECK:STDOUT:     inst5000001A:    {kind: NameRef, arg0: name1, arg1: inst50000016, type: type(inst50000018)}
// CHECK:STDOUT:     inst5000001B:    {kind: Call, arg0: inst5000001A, arg1: inst_block_empty, type: type(inst50000013)}
// CHECK:STDOUT:     inst5000001C:    {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst0000000E:    concrete_constant(inst0000000E)
// CHECK:STDOUT:       inst50000010:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst50000011:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst50000012:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000013:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst50000014:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst50000015:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst50000016:    concrete_constant(inst50000019)
// CHECK:STDOUT:       inst50000017:    concrete_constant(inst50000019)
// CHECK:STDOUT:       inst50000018:    concrete_constant(inst50000018)
// CHECK:STDOUT:       inst50000019:    concrete_constant(inst50000019)
// CHECK:STDOUT:       inst5000001A:    concrete_constant(inst50000019)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst50000011
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst50000010
// CHECK:STDOUT:       1:               inst50000016
// CHECK:STDOUT:       2:               inst50000017
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst50000015
// CHECK:STDOUT:       1:               inst5000001A
// CHECK:STDOUT:       2:               inst5000001B
// CHECK:STDOUT:       3:               inst5000001C
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst0000000E
// CHECK:STDOUT:       1:               inst5000000F
// CHECK:STDOUT:       2:               inst50000011
// CHECK:STDOUT: ...

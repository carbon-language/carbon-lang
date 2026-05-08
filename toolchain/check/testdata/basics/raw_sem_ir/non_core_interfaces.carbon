// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-prelude-import --dump-raw-sem-ir --no-dump-sem-ir
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/non_core_interfaces.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/non_core_interfaces.carbon

// Tests that a non-core interface with the same name as a core interface does
// not produce a `core_interfce: ${interface_name}` entry. We test this for both
// locally-defined interfaces and imported interfaces.

// --- non_core.carbon
package NonCore;
interface Destroy {
  let T1:! type;
  let T2:! type;
}

// --- main.carbon
import NonCore;

interface Copy {
  let T1:! type;
  let T2:! type;
}

//@dump-sem-ir-begin
fn UseNonCoreDestroy[T:! NonCore.Destroy](_: T.T1, _: T.T2) {}
fn UseLocalCopy[T:! Copy](_: T.T1, _: T.T2) {}
//@dump-sem-ir-end

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        non_core.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           NonCore
// CHECK:STDOUT:     name1:           Destroy
// CHECK:STDOUT:     name2:           T1
// CHECK:STDOUT:     name3:           T2
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   clang_decl_signatures: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst60000010}}
// CHECK:STDOUT:     name_scope60000001: {inst: inst60000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name(SelfType): inst60000012}}
// CHECK:STDOUT:     name_scope60000002: {inst: inst60000015, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {name2: inst60000019, name3: inst6000001D}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name(SelfType), parent_scope: name_scope60000001, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface60000000: {name: name1, parent_scope: name_scope0, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:   associated_constants:
// CHECK:STDOUT:     assoc_const60000000: {name: name2, parent_scope: name_scope60000002, decl_id: inst60000017, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const60000001: {name: name3, parent_scope: name_scope60000002, decl_id: inst6000001C, default_value_id: inst<none>}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic60000000: {decl: inst<none>, bindings: inst_block<none>, self_specific_id: specific<none>, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic60000001: {decl: inst60000015, bindings: inst_block60000005, self_specific_id: specific60000000, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific60000000: {generic: generic60000001, args: inst_block60000006, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:   specific_interfaces: {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(InstType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:   facet_types:
// CHECK:STDOUT:     facet_type60000000: {extends interface: interface60000000}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000010:    {kind: InterfaceDecl, arg0: interface60000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000011:    {kind: FacetType, arg0: facet_type60000000, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000013:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000014:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000015:    {kind: InterfaceWithSelfDecl, arg0: interface60000000}
// CHECK:STDOUT:     inst60000016:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst60000017:    {kind: AssociatedConstantDecl, arg0: assoc_const60000000, arg1: inst_block6000000A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000018:    {kind: AssociatedEntityType, arg0: interface60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000019:    {kind: AssociatedEntity, arg0: element0, arg1: inst60000017, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001A:    {kind: AssociatedEntity, arg0: element0, arg1: inst60000017, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001B:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst6000001C:    {kind: AssociatedConstantDecl, arg0: assoc_const60000001, arg1: inst_block6000000C, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: AssociatedEntity, arg0: element1, arg1: inst6000001C, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001E:    {kind: AssociatedEntity, arg0: element1, arg1: inst6000001C, type: type(inst60000018)}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst60000010:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000012:    symbolic_constant60000000
// CHECK:STDOUT:       inst60000013:    symbolic_constant60000000
// CHECK:STDOUT:       inst60000014:    symbolic_constant60000001
// CHECK:STDOUT:       inst60000015:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst60000016:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst60000017:    concrete_constant(inst60000017)
// CHECK:STDOUT:       inst60000018:    concrete_constant(inst60000018)
// CHECK:STDOUT:       inst60000019:    concrete_constant(inst6000001A)
// CHECK:STDOUT:       inst6000001A:    concrete_constant(inst6000001A)
// CHECK:STDOUT:       inst6000001B:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst6000001C:    concrete_constant(inst6000001C)
// CHECK:STDOUT:       inst6000001D:    concrete_constant(inst6000001E)
// CHECK:STDOUT:       inst6000001E:    concrete_constant(inst6000001E)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant60000000: {inst: inst60000013, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant60000001: {inst: inst60000013, kind: checked, attached: {generic: generic60000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst60000010
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block60000005:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:     inst_block60000006:
// CHECK:STDOUT:       0:               inst60000013
// CHECK:STDOUT:     inst_block60000007:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:       1:               inst60000015
// CHECK:STDOUT:     inst_block60000008:
// CHECK:STDOUT:       0:               inst60000017
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:     inst_block60000009:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:     inst_block6000000A:
// CHECK:STDOUT:       0:               inst60000019
// CHECK:STDOUT:     inst_block6000000B:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:     inst_block6000000C:
// CHECK:STDOUT:       0:               inst6000001D
// CHECK:STDOUT:     inst_block6000000D:
// CHECK:STDOUT:       0:               inst60000017
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:     inst_block6000000E:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst60000010
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     NonCore
// CHECK:STDOUT:         identifier1:     Destroy
// CHECK:STDOUT:         identifier2:     T1
// CHECK:STDOUT:         identifier3:     T2
// CHECK:STDOUT:       strings:         {}
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        main.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           NonCore
// CHECK:STDOUT:     name1:           Copy
// CHECK:STDOUT:     name2:           T1
// CHECK:STDOUT:     name3:           T2
// CHECK:STDOUT:     name4:           UseNonCoreDestroy
// CHECK:STDOUT:     name5:           T
// CHECK:STDOUT:     name6:           Destroy
// CHECK:STDOUT:     name7:           UseLocalCopy
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir50000002: {decl_id: inst50000010, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir50000002, inst_id: inst60000010}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir50000002, inst_id: inst60000010}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir50000002, inst_id: inst60000015}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir50000002, inst_id: inst60000019}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir50000002, inst_id: inst6000001D}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir50000002, inst_id: inst60000017}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir50000002, inst_id: inst6000001C}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir50000002, inst_id: inst60000012}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir50000002, inst_id: inst60000012}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir50000002, inst_id: inst60000017}
// CHECK:STDOUT:     import_ir_instA: {ir_id: import_ir50000002, inst_id: inst60000017}
// CHECK:STDOUT:     import_ir_instB: {ir_id: import_ir50000002, inst_id: inst6000001C}
// CHECK:STDOUT:     import_ir_instC: {ir_id: import_ir50000002, inst_id: inst6000001C}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   clang_decl_signatures: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst50000011, name1: inst50000012, name4: inst50000060, name7: inst50000095}}
// CHECK:STDOUT:     name_scope50000001: {inst: inst50000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name6: inst50000025}}
// CHECK:STDOUT:     name_scope50000002: {inst: inst50000012, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name(SelfType): inst50000014}}
// CHECK:STDOUT:     name_scope50000003: {inst: inst50000017, parent_scope: name_scope50000002, has_error: false, extended_scopes: [], names: {name2: inst5000001B, name3: inst5000001F}}
// CHECK:STDOUT:     name_scope50000004: {inst: inst50000026, parent_scope: name_scope50000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst5000002F}}
// CHECK:STDOUT:     name_scope50000005: {inst: inst50000028, parent_scope: name_scope50000004, has_error: false, extended_scopes: [], names: {name2: inst5000002A, name3: inst5000002B}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name50000000: {name: name(SelfType), parent_scope: name_scope50000002, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000001: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000002: {name: name6, parent_scope: name_scope50000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000003: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000004: {name: name2, parent_scope: name_scope50000005, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000005: {name: name3, parent_scope: name_scope50000005, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000007: {name: name5, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000008: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>}
// CHECK:STDOUT:     entity_name50000009: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>}
// CHECK:STDOUT:     entity_name5000000A: {name: name5, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name5000000B: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>}
// CHECK:STDOUT:     entity_name5000000C: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function50000000: {name: name4, parent_scope: name_scope0, call_param_patterns_id: inst_block50000018, call_params_id: inst_block50000019, body: [inst_block5000001F]}
// CHECK:STDOUT:     function50000001: {name: name7, parent_scope: name_scope0, call_param_patterns_id: inst_block50000028, call_params_id: inst_block50000029, body: [inst_block5000002F]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface50000000: {name: name1, parent_scope: name_scope0, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:     interface50000001: {name: name6, parent_scope: name_scope50000001, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:   associated_constants:
// CHECK:STDOUT:     assoc_const50000000: {name: name2, parent_scope: name_scope50000003, decl_id: inst50000019, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const50000001: {name: name3, parent_scope: name_scope50000003, decl_id: inst5000001E, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const50000002: {name: name2, parent_scope: name_scope50000005, decl_id: inst50000041, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const50000003: {name: name3, parent_scope: name_scope50000005, decl_id: inst50000052, default_value_id: inst<none>}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic50000000: {decl: inst<none>, bindings: inst_block<none>, self_specific_id: specific<none>, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic50000001: {decl: inst50000017, bindings: inst_block50000005, self_specific_id: specific50000000, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic50000002: {decl: inst50000028, bindings: inst_block50000010, self_specific_id: specific50000001, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic50000003: {decl: inst50000060, bindings: inst_block5000001C, self_specific_id: specific50000003, decl_block_id: inst_block5000001D, definition_block_id: inst_block50000020}
// CHECK:STDOUT:     generic50000004: {decl: inst50000095, bindings: inst_block5000002C, self_specific_id: specific50000005, decl_block_id: inst_block5000002D, definition_block_id: inst_block50000030}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific50000000: {generic: generic50000001, args: inst_block50000006, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000001: {generic: generic50000002, args: inst_block50000011, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000002: {generic: generic50000002, args: inst_block50000014, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000003: {generic: generic50000003, args: inst_block50000014, decl_block_id: inst_block5000001E, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific50000004: {generic: generic50000001, args: inst_block50000024, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000005: {generic: generic50000004, args: inst_block50000024, decl_block_id: inst_block5000002E, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:   specific_interfaces:
// CHECK:STDOUT:     specific_interface50000000: {interface_id: interface50000001, specific_id: specific<none>}
// CHECK:STDOUT:     specific_interface50000001: {interface_id: interface50000000, specific_id: specific<none>}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(InstType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000027)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000027)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000061)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000062)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000062)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000062)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant50000009)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant50000009)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(symbolic_constant5000000E)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant5000000E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(inst50000013)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000013)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000096)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000062)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant5000001B)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant5000001B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(symbolic_constant50000020)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant50000020)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:   facet_types:
// CHECK:STDOUT:     facet_type50000000: {extends interface: interface50000000}
// CHECK:STDOUT:     facet_type50000001: {}
// CHECK:STDOUT:     facet_type50000002: {extends interface: interface50000001}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000010:    {kind: ImportDecl, arg0: name0}
// CHECK:STDOUT:     inst50000011:    {kind: Namespace, arg0: name_scope50000001, arg1: inst50000010, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000012:    {kind: InterfaceDecl, arg0: interface50000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000013:    {kind: FacetType, arg0: facet_type50000000, type: type(TypeType)}
// CHECK:STDOUT:     inst50000014:    {kind: SymbolicBinding, arg0: entity_name50000000, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000015:    {kind: SymbolicBinding, arg0: entity_name50000000, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000016:    {kind: SymbolicBinding, arg0: entity_name50000000, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000017:    {kind: InterfaceWithSelfDecl, arg0: interface50000000}
// CHECK:STDOUT:     inst50000018:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst50000019:    {kind: AssociatedConstantDecl, arg0: assoc_const50000000, arg1: inst_block5000000A, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001A:    {kind: AssociatedEntityType, arg0: interface50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001B:    {kind: AssociatedEntity, arg0: element0, arg1: inst50000019, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000001C:    {kind: AssociatedEntity, arg0: element0, arg1: inst50000019, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000001D:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst5000001E:    {kind: AssociatedConstantDecl, arg0: assoc_const50000001, arg1: inst_block5000000C, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001F:    {kind: AssociatedEntity, arg0: element1, arg1: inst5000001E, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000020:    {kind: AssociatedEntity, arg0: element1, arg1: inst5000001E, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000021:    {kind: FacetType, arg0: facet_type50000001, type: type(TypeType)}
// CHECK:STDOUT:     inst50000022:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000021)}
// CHECK:STDOUT:     inst50000023:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000021)}
// CHECK:STDOUT:     inst50000024:    {kind: NameRef, arg0: name0, arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000025:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name50000002, type: type(TypeType)}
// CHECK:STDOUT:     inst50000026:    {kind: InterfaceDecl, arg0: interface50000001, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000027:    {kind: FacetType, arg0: facet_type50000002, type: type(TypeType)}
// CHECK:STDOUT:     inst50000028:    {kind: InterfaceWithSelfDecl, arg0: interface50000001}
// CHECK:STDOUT:     inst50000029:    {kind: SymbolicBinding, arg0: entity_name50000003, arg1: inst<none>, type: type(inst50000027)}
// CHECK:STDOUT:     inst5000002A:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst5000003B)}
// CHECK:STDOUT:     inst5000002B:    {kind: ImportRefLoaded, arg0: import_ir_inst4, arg1: entity_name<none>, type: type(inst5000003B)}
// CHECK:STDOUT:     inst5000002C:    {kind: ImportRefUnloaded, arg0: import_ir_inst5, arg1: entity_name50000004}
// CHECK:STDOUT:     inst5000002D:    {kind: ImportRefUnloaded, arg0: import_ir_inst6, arg1: entity_name50000005}
// CHECK:STDOUT:     inst5000002E:    {kind: ImportRefLoaded, arg0: import_ir_inst7, arg1: entity_name<none>, type: type(inst50000027)}
// CHECK:STDOUT:     inst5000002F:    {kind: ImportRefUnloaded, arg0: import_ir_inst8, arg1: entity_name<none>}
// CHECK:STDOUT:     inst50000030:    {kind: NameRef, arg0: name6, arg1: inst50000025, type: type(TypeType)}
// CHECK:STDOUT:     inst50000031:    {kind: PatternType, arg0: inst50000027, type: type(TypeType)}
// CHECK:STDOUT:     inst50000032:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000027)}
// CHECK:STDOUT:     inst50000033:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000027)}
// CHECK:STDOUT:     inst50000034:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000027)}
// CHECK:STDOUT:     inst50000035:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst50000031)}
// CHECK:STDOUT:     inst50000036:    {kind: NameRef, arg0: name5, arg1: inst50000032, type: type(inst50000027)}
// CHECK:STDOUT:     inst50000037:    {kind: FacetAccessType, arg0: inst50000036, type: type(TypeType)}
// CHECK:STDOUT:     inst50000038:    {kind: FacetAccessType, arg0: inst50000033, type: type(TypeType)}
// CHECK:STDOUT:     inst50000039:    {kind: FacetAccessType, arg0: inst50000034, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003A:    {kind: Converted, arg0: inst50000036, arg1: inst50000037, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003B:    {kind: AssociatedEntityType, arg0: interface50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003C:    {kind: ImportRefLoaded, arg0: import_ir_inst9, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003D:    {kind: AssociatedEntity, arg0: element0, arg1: inst5000003C, type: type(inst5000003B)}
// CHECK:STDOUT:     inst5000003E:    {kind: NameRef, arg0: name2, arg1: inst5000002A, type: type(inst5000003B)}
// CHECK:STDOUT:     inst5000003F:    {kind: LookupImplWitness, arg0: inst50000033, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000040:    {kind: LookupImplWitness, arg0: inst50000033, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000041:    {kind: AssociatedConstantDecl, arg0: assoc_const50000002, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000042:    {kind: ImplWitnessAccess, arg0: inst50000040, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000043:    {kind: ImplWitnessAccess, arg0: inst50000040, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000044:    {kind: LookupImplWitness, arg0: inst50000034, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000045:    {kind: ImplWitnessAccess, arg0: inst50000044, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000046:    {kind: PatternType, arg0: inst50000043, type: type(TypeType)}
// CHECK:STDOUT:     inst50000047:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000000D)}
// CHECK:STDOUT:     inst50000048:    {kind: PatternType, arg0: inst50000045, type: type(TypeType)}
// CHECK:STDOUT:     inst50000049:    {kind: ValueBinding, arg0: entity_name50000008, arg1: inst5000005C, type: type(symbolic_constant5000000B)}
// CHECK:STDOUT:     inst5000004A:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000047, type: type(symbolic_constant5000000D)}
// CHECK:STDOUT:     inst5000004B:    {kind: NameRef, arg0: name5, arg1: inst50000032, type: type(inst50000027)}
// CHECK:STDOUT:     inst5000004C:    {kind: FacetAccessType, arg0: inst5000004B, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004D:    {kind: Converted, arg0: inst5000004B, arg1: inst5000004C, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004E:    {kind: ImportRefLoaded, arg0: import_ir_instB, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004F:    {kind: AssociatedEntity, arg0: element1, arg1: inst5000004E, type: type(inst5000003B)}
// CHECK:STDOUT:     inst50000050:    {kind: NameRef, arg0: name3, arg1: inst5000002B, type: type(inst5000003B)}
// CHECK:STDOUT:     inst50000051:    {kind: LookupImplWitness, arg0: inst50000033, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000052:    {kind: AssociatedConstantDecl, arg0: assoc_const50000003, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000053:    {kind: ImplWitnessAccess, arg0: inst50000040, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst50000054:    {kind: ImplWitnessAccess, arg0: inst50000040, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst50000055:    {kind: ImplWitnessAccess, arg0: inst50000044, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst50000056:    {kind: PatternType, arg0: inst50000054, type: type(TypeType)}
// CHECK:STDOUT:     inst50000057:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000011)}
// CHECK:STDOUT:     inst50000058:    {kind: PatternType, arg0: inst50000055, type: type(TypeType)}
// CHECK:STDOUT:     inst50000059:    {kind: ValueBinding, arg0: entity_name50000009, arg1: inst5000005E, type: type(symbolic_constant5000000F)}
// CHECK:STDOUT:     inst5000005A:    {kind: WrapperBindingPattern, arg0: entity_name50000009, arg1: inst50000057, type: type(symbolic_constant50000011)}
// CHECK:STDOUT:     inst5000005B:    {kind: SpliceBlock, arg0: inst_block5000000E, arg1: inst50000030, type: type(TypeType)}
// CHECK:STDOUT:     inst5000005C:    {kind: ValueParam, arg0: call_param0, arg1: name(Underscore), type: type(symbolic_constant5000000B)}
// CHECK:STDOUT:     inst5000005D:    {kind: SpliceBlock, arg0: inst_block50000013, arg1: inst50000042, type: type(TypeType)}
// CHECK:STDOUT:     inst5000005E:    {kind: ValueParam, arg0: call_param1, arg1: name(Underscore), type: type(symbolic_constant5000000F)}
// CHECK:STDOUT:     inst5000005F:    {kind: SpliceBlock, arg0: inst_block50000016, arg1: inst50000053, type: type(TypeType)}
// CHECK:STDOUT:     inst50000060:    {kind: FunctionDecl, arg0: function50000000, arg1: inst_block5000001B, type: type(inst50000061)}
// CHECK:STDOUT:     inst50000061:    {kind: FunctionType, arg0: function50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000062:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000063:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000061)}
// CHECK:STDOUT:     inst50000064:    {kind: RequireCompleteType, arg0: inst50000043, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000065:    {kind: RequireCompleteType, arg0: inst50000043, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000066:    {kind: RequireCompleteType, arg0: inst50000045, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000067:    {kind: RequireCompleteType, arg0: inst50000054, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000068:    {kind: RequireCompleteType, arg0: inst50000054, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000069:    {kind: RequireCompleteType, arg0: inst50000055, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000006A:    {kind: Return}
// CHECK:STDOUT:     inst5000006B:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000021)}
// CHECK:STDOUT:     inst5000006C:    {kind: NameRef, arg0: name1, arg1: inst50000012, type: type(TypeType)}
// CHECK:STDOUT:     inst5000006D:    {kind: PatternType, arg0: inst50000013, type: type(TypeType)}
// CHECK:STDOUT:     inst5000006E:    {kind: SymbolicBinding, arg0: entity_name5000000A, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst5000006F:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000070:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000071:    {kind: SymbolicBindingPattern, arg0: entity_name5000000A, type: type(inst5000006D)}
// CHECK:STDOUT:     inst50000072:    {kind: NameRef, arg0: name5, arg1: inst5000006E, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000073:    {kind: FacetAccessType, arg0: inst50000072, type: type(TypeType)}
// CHECK:STDOUT:     inst50000074:    {kind: FacetAccessType, arg0: inst5000006F, type: type(TypeType)}
// CHECK:STDOUT:     inst50000075:    {kind: FacetAccessType, arg0: inst50000070, type: type(TypeType)}
// CHECK:STDOUT:     inst50000076:    {kind: Converted, arg0: inst50000072, arg1: inst50000073, type: type(TypeType)}
// CHECK:STDOUT:     inst50000077:    {kind: NameRef, arg0: name2, arg1: inst5000001B, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000078:    {kind: LookupImplWitness, arg0: inst5000006F, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000079:    {kind: LookupImplWitness, arg0: inst5000006F, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000007A:    {kind: ImplWitnessAccess, arg0: inst50000079, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst5000007B:    {kind: ImplWitnessAccess, arg0: inst50000079, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst5000007C:    {kind: LookupImplWitness, arg0: inst50000070, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000007D:    {kind: ImplWitnessAccess, arg0: inst5000007C, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst5000007E:    {kind: PatternType, arg0: inst5000007B, type: type(TypeType)}
// CHECK:STDOUT:     inst5000007F:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000001F)}
// CHECK:STDOUT:     inst50000080:    {kind: PatternType, arg0: inst5000007D, type: type(TypeType)}
// CHECK:STDOUT:     inst50000081:    {kind: ValueBinding, arg0: entity_name5000000B, arg1: inst50000091, type: type(symbolic_constant5000001D)}
// CHECK:STDOUT:     inst50000082:    {kind: WrapperBindingPattern, arg0: entity_name5000000B, arg1: inst5000007F, type: type(symbolic_constant5000001F)}
// CHECK:STDOUT:     inst50000083:    {kind: NameRef, arg0: name5, arg1: inst5000006E, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000084:    {kind: FacetAccessType, arg0: inst50000083, type: type(TypeType)}
// CHECK:STDOUT:     inst50000085:    {kind: Converted, arg0: inst50000083, arg1: inst50000084, type: type(TypeType)}
// CHECK:STDOUT:     inst50000086:    {kind: NameRef, arg0: name3, arg1: inst5000001F, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000087:    {kind: LookupImplWitness, arg0: inst5000006F, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000088:    {kind: ImplWitnessAccess, arg0: inst50000079, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst50000089:    {kind: ImplWitnessAccess, arg0: inst50000079, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000008A:    {kind: ImplWitnessAccess, arg0: inst5000007C, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000008B:    {kind: PatternType, arg0: inst50000089, type: type(TypeType)}
// CHECK:STDOUT:     inst5000008C:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000023)}
// CHECK:STDOUT:     inst5000008D:    {kind: PatternType, arg0: inst5000008A, type: type(TypeType)}
// CHECK:STDOUT:     inst5000008E:    {kind: ValueBinding, arg0: entity_name5000000C, arg1: inst50000093, type: type(symbolic_constant50000021)}
// CHECK:STDOUT:     inst5000008F:    {kind: WrapperBindingPattern, arg0: entity_name5000000C, arg1: inst5000008C, type: type(symbolic_constant50000023)}
// CHECK:STDOUT:     inst50000090:    {kind: SpliceBlock, arg0: inst_block50000021, arg1: inst5000006C, type: type(TypeType)}
// CHECK:STDOUT:     inst50000091:    {kind: ValueParam, arg0: call_param0, arg1: name(Underscore), type: type(symbolic_constant5000001D)}
// CHECK:STDOUT:     inst50000092:    {kind: SpliceBlock, arg0: inst_block50000023, arg1: inst5000007A, type: type(TypeType)}
// CHECK:STDOUT:     inst50000093:    {kind: ValueParam, arg0: call_param1, arg1: name(Underscore), type: type(symbolic_constant50000021)}
// CHECK:STDOUT:     inst50000094:    {kind: SpliceBlock, arg0: inst_block50000026, arg1: inst50000088, type: type(TypeType)}
// CHECK:STDOUT:     inst50000095:    {kind: FunctionDecl, arg0: function50000001, arg1: inst_block5000002B, type: type(inst50000096)}
// CHECK:STDOUT:     inst50000096:    {kind: FunctionType, arg0: function50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000097:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000096)}
// CHECK:STDOUT:     inst50000098:    {kind: RequireCompleteType, arg0: inst5000007B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000099:    {kind: RequireCompleteType, arg0: inst5000007B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000009A:    {kind: RequireCompleteType, arg0: inst5000007D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000009B:    {kind: RequireCompleteType, arg0: inst50000089, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000009C:    {kind: RequireCompleteType, arg0: inst50000089, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000009D:    {kind: RequireCompleteType, arg0: inst5000008A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000009E:    {kind: Return}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst50000011:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000012:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst50000013:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst50000014:    symbolic_constant50000000
// CHECK:STDOUT:       inst50000015:    symbolic_constant50000000
// CHECK:STDOUT:       inst50000016:    symbolic_constant50000001
// CHECK:STDOUT:       inst50000017:    concrete_constant(inst50000017)
// CHECK:STDOUT:       inst50000018:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst50000019:    concrete_constant(inst50000019)
// CHECK:STDOUT:       inst5000001A:    concrete_constant(inst5000001A)
// CHECK:STDOUT:       inst5000001B:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst5000001C:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst5000001D:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst5000001E:    concrete_constant(inst5000001E)
// CHECK:STDOUT:       inst5000001F:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst50000020:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst50000021:    concrete_constant(inst50000021)
// CHECK:STDOUT:       inst50000022:    symbolic_constant50000002
// CHECK:STDOUT:       inst50000023:    symbolic_constant50000002
// CHECK:STDOUT:       inst50000024:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000025:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst50000026:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst50000027:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst50000028:    concrete_constant(inst50000028)
// CHECK:STDOUT:       inst50000029:    symbolic_constant50000003
// CHECK:STDOUT:       inst5000002A:    concrete_constant(inst5000003D)
// CHECK:STDOUT:       inst5000002B:    concrete_constant(inst5000004F)
// CHECK:STDOUT:       inst5000002C:    constant<none>
// CHECK:STDOUT:       inst5000002D:    constant<none>
// CHECK:STDOUT:       inst5000002E:    symbolic_constant50000003
// CHECK:STDOUT:       inst5000002F:    constant<none>
// CHECK:STDOUT:       inst50000030:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst50000031:    concrete_constant(inst50000031)
// CHECK:STDOUT:       inst50000032:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000033:    symbolic_constant50000004
// CHECK:STDOUT:       inst50000034:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000035:    concrete_constant(inst50000035)
// CHECK:STDOUT:       inst50000036:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000037:    symbolic_constant50000007
// CHECK:STDOUT:       inst50000038:    symbolic_constant50000006
// CHECK:STDOUT:       inst50000039:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000003A:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000003B:    concrete_constant(inst5000003B)
// CHECK:STDOUT:       inst5000003C:    concrete_constant(inst50000041)
// CHECK:STDOUT:       inst5000003D:    concrete_constant(inst5000003D)
// CHECK:STDOUT:       inst5000003E:    concrete_constant(inst5000003D)
// CHECK:STDOUT:       inst50000040:    symbolic_constant50000008
// CHECK:STDOUT:       inst50000041:    concrete_constant(inst50000041)
// CHECK:STDOUT:       inst50000042:    symbolic_constant5000000B
// CHECK:STDOUT:       inst50000043:    symbolic_constant50000009
// CHECK:STDOUT:       inst50000044:    symbolic_constant5000000A
// CHECK:STDOUT:       inst50000045:    symbolic_constant5000000B
// CHECK:STDOUT:       inst50000046:    symbolic_constant5000000C
// CHECK:STDOUT:       inst50000047:    concrete_constant(inst50000047)
// CHECK:STDOUT:       inst50000048:    symbolic_constant5000000D
// CHECK:STDOUT:       inst5000004A:    concrete_constant(inst5000004A)
// CHECK:STDOUT:       inst5000004B:    symbolic_constant50000005
// CHECK:STDOUT:       inst5000004C:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000004D:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000004E:    concrete_constant(inst50000052)
// CHECK:STDOUT:       inst5000004F:    concrete_constant(inst5000004F)
// CHECK:STDOUT:       inst50000050:    concrete_constant(inst5000004F)
// CHECK:STDOUT:       inst50000052:    concrete_constant(inst50000052)
// CHECK:STDOUT:       inst50000053:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000054:    symbolic_constant5000000E
// CHECK:STDOUT:       inst50000055:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000056:    symbolic_constant50000010
// CHECK:STDOUT:       inst50000057:    concrete_constant(inst50000057)
// CHECK:STDOUT:       inst50000058:    symbolic_constant50000011
// CHECK:STDOUT:       inst5000005A:    concrete_constant(inst5000005A)
// CHECK:STDOUT:       inst5000005B:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst5000005D:    symbolic_constant5000000B
// CHECK:STDOUT:       inst5000005F:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000060:    concrete_constant(inst50000063)
// CHECK:STDOUT:       inst50000061:    concrete_constant(inst50000061)
// CHECK:STDOUT:       inst50000062:    concrete_constant(inst50000062)
// CHECK:STDOUT:       inst50000063:    concrete_constant(inst50000063)
// CHECK:STDOUT:       inst50000064:    symbolic_constant50000013
// CHECK:STDOUT:       inst50000065:    symbolic_constant50000012
// CHECK:STDOUT:       inst50000066:    symbolic_constant50000013
// CHECK:STDOUT:       inst50000067:    symbolic_constant50000015
// CHECK:STDOUT:       inst50000068:    symbolic_constant50000014
// CHECK:STDOUT:       inst50000069:    symbolic_constant50000015
// CHECK:STDOUT:       inst5000006B:    symbolic_constant50000002
// CHECK:STDOUT:       inst5000006C:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst5000006D:    concrete_constant(inst5000006D)
// CHECK:STDOUT:       inst5000006E:    symbolic_constant50000017
// CHECK:STDOUT:       inst5000006F:    symbolic_constant50000016
// CHECK:STDOUT:       inst50000070:    symbolic_constant50000017
// CHECK:STDOUT:       inst50000071:    concrete_constant(inst50000071)
// CHECK:STDOUT:       inst50000072:    symbolic_constant50000017
// CHECK:STDOUT:       inst50000073:    symbolic_constant50000019
// CHECK:STDOUT:       inst50000074:    symbolic_constant50000018
// CHECK:STDOUT:       inst50000075:    symbolic_constant50000019
// CHECK:STDOUT:       inst50000076:    symbolic_constant50000019
// CHECK:STDOUT:       inst50000077:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst50000079:    symbolic_constant5000001A
// CHECK:STDOUT:       inst5000007A:    symbolic_constant5000001D
// CHECK:STDOUT:       inst5000007B:    symbolic_constant5000001B
// CHECK:STDOUT:       inst5000007C:    symbolic_constant5000001C
// CHECK:STDOUT:       inst5000007D:    symbolic_constant5000001D
// CHECK:STDOUT:       inst5000007E:    symbolic_constant5000001E
// CHECK:STDOUT:       inst5000007F:    concrete_constant(inst5000007F)
// CHECK:STDOUT:       inst50000080:    symbolic_constant5000001F
// CHECK:STDOUT:       inst50000082:    concrete_constant(inst50000082)
// CHECK:STDOUT:       inst50000083:    symbolic_constant50000017
// CHECK:STDOUT:       inst50000084:    symbolic_constant50000019
// CHECK:STDOUT:       inst50000085:    symbolic_constant50000019
// CHECK:STDOUT:       inst50000086:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst50000088:    symbolic_constant50000021
// CHECK:STDOUT:       inst50000089:    symbolic_constant50000020
// CHECK:STDOUT:       inst5000008A:    symbolic_constant50000021
// CHECK:STDOUT:       inst5000008B:    symbolic_constant50000022
// CHECK:STDOUT:       inst5000008C:    concrete_constant(inst5000008C)
// CHECK:STDOUT:       inst5000008D:    symbolic_constant50000023
// CHECK:STDOUT:       inst5000008F:    concrete_constant(inst5000008F)
// CHECK:STDOUT:       inst50000090:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst50000092:    symbolic_constant5000001D
// CHECK:STDOUT:       inst50000094:    symbolic_constant50000021
// CHECK:STDOUT:       inst50000095:    concrete_constant(inst50000097)
// CHECK:STDOUT:       inst50000096:    concrete_constant(inst50000096)
// CHECK:STDOUT:       inst50000097:    concrete_constant(inst50000097)
// CHECK:STDOUT:       inst50000098:    symbolic_constant50000025
// CHECK:STDOUT:       inst50000099:    symbolic_constant50000024
// CHECK:STDOUT:       inst5000009A:    symbolic_constant50000025
// CHECK:STDOUT:       inst5000009B:    symbolic_constant50000027
// CHECK:STDOUT:       inst5000009C:    symbolic_constant50000026
// CHECK:STDOUT:       inst5000009D:    symbolic_constant50000027
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant50000000: {inst: inst50000015, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000001: {inst: inst50000015, kind: checked, attached: {generic: generic50000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant50000002: {inst: inst50000023, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000003: {inst: inst50000029, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000004: {inst: inst50000033, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000005: {inst: inst50000033, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant50000006: {inst: inst50000038, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000007: {inst: inst50000038, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant50000008: {inst: inst50000040, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000009: {inst: inst50000043, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000A: {inst: inst50000040, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant5000000B: {inst: inst50000043, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant5000000C: {inst: inst50000046, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000D: {inst: inst50000046, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant5000000E: {inst: inst50000054, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000F: {inst: inst50000054, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant50000010: {inst: inst50000056, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000011: {inst: inst50000056, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant50000012: {inst: inst50000065, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000013: {inst: inst50000065, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant50000014: {inst: inst50000068, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000015: {inst: inst50000068, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant50000016: {inst: inst5000006F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000017: {inst: inst5000006F, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant50000018: {inst: inst50000074, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000019: {inst: inst50000074, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant5000001A: {inst: inst50000079, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000001B: {inst: inst5000007B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000001C: {inst: inst50000079, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant5000001D: {inst: inst5000007B, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant5000001E: {inst: inst5000007E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000001F: {inst: inst5000007E, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant50000020: {inst: inst50000089, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000021: {inst: inst50000089, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant50000022: {inst: inst5000008B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000023: {inst: inst5000008B, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant50000024: {inst: inst50000099, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000025: {inst: inst50000099, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant50000026: {inst: inst5000009C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000027: {inst: inst5000009C, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_def1}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst50000012
// CHECK:STDOUT:       1:               inst50000060
// CHECK:STDOUT:       2:               inst50000095
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst50000011
// CHECK:STDOUT:       1:               inst50000025
// CHECK:STDOUT:       2:               inst50000026
// CHECK:STDOUT:       3:               inst50000028
// CHECK:STDOUT:       4:               inst5000002A
// CHECK:STDOUT:       5:               inst5000002B
// CHECK:STDOUT:       6:               inst5000002C
// CHECK:STDOUT:       7:               inst5000002D
// CHECK:STDOUT:       8:               inst5000002E
// CHECK:STDOUT:       9:               inst5000002F
// CHECK:STDOUT:       10:              inst5000003C
// CHECK:STDOUT:       11:              inst50000041
// CHECK:STDOUT:       12:              inst5000004E
// CHECK:STDOUT:       13:              inst50000052
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block50000005:
// CHECK:STDOUT:       0:               inst50000014
// CHECK:STDOUT:     inst_block50000006:
// CHECK:STDOUT:       0:               inst50000015
// CHECK:STDOUT:     inst_block50000007:
// CHECK:STDOUT:       0:               inst50000014
// CHECK:STDOUT:       1:               inst50000017
// CHECK:STDOUT:     inst_block50000008:
// CHECK:STDOUT:       0:               inst50000019
// CHECK:STDOUT:       1:               inst5000001E
// CHECK:STDOUT:     inst_block50000009:
// CHECK:STDOUT:       0:               inst50000018
// CHECK:STDOUT:     inst_block5000000A:
// CHECK:STDOUT:       0:               inst5000001B
// CHECK:STDOUT:     inst_block5000000B:
// CHECK:STDOUT:       0:               inst5000001D
// CHECK:STDOUT:     inst_block5000000C:
// CHECK:STDOUT:       0:               inst5000001F
// CHECK:STDOUT:     inst_block5000000D:
// CHECK:STDOUT:       0:               inst50000019
// CHECK:STDOUT:       1:               inst5000001E
// CHECK:STDOUT:     inst_block5000000E:
// CHECK:STDOUT:       0:               inst50000022
// CHECK:STDOUT:       1:               inst50000024
// CHECK:STDOUT:       2:               inst50000030
// CHECK:STDOUT:     inst_block5000000F:
// CHECK:STDOUT:       0:               inst5000002C
// CHECK:STDOUT:       1:               inst5000002D
// CHECK:STDOUT:     inst_block50000010:
// CHECK:STDOUT:       0:               inst5000002E
// CHECK:STDOUT:     inst_block50000011:
// CHECK:STDOUT:       0:               inst50000029
// CHECK:STDOUT:     inst_block50000012:
// CHECK:STDOUT:       0:               inst50000035
// CHECK:STDOUT:     inst_block50000013:
// CHECK:STDOUT:       0:               inst50000036
// CHECK:STDOUT:       1:               inst50000037
// CHECK:STDOUT:       2:               inst5000003A
// CHECK:STDOUT:       3:               inst5000003E
// CHECK:STDOUT:       4:               inst50000042
// CHECK:STDOUT:     inst_block50000014:
// CHECK:STDOUT:       0:               inst50000033
// CHECK:STDOUT:     inst_block50000015:
// CHECK:STDOUT:       0:               inst50000040
// CHECK:STDOUT:     inst_block50000016:
// CHECK:STDOUT:       0:               inst5000004B
// CHECK:STDOUT:       1:               inst5000004C
// CHECK:STDOUT:       2:               inst5000004D
// CHECK:STDOUT:       3:               inst50000050
// CHECK:STDOUT:       4:               inst50000053
// CHECK:STDOUT:     inst_block50000017:
// CHECK:STDOUT:       0:               inst5000004A
// CHECK:STDOUT:       1:               inst5000005A
// CHECK:STDOUT:     inst_block50000018:
// CHECK:STDOUT:       0:               inst50000047
// CHECK:STDOUT:       1:               inst50000057
// CHECK:STDOUT:     inst_block50000019:
// CHECK:STDOUT:       0:               inst5000005C
// CHECK:STDOUT:       1:               inst5000005E
// CHECK:STDOUT:     inst_block5000001A:
// CHECK:STDOUT:       0:               inst50000035
// CHECK:STDOUT:       1:               inst50000047
// CHECK:STDOUT:       2:               inst5000004A
// CHECK:STDOUT:       3:               inst50000057
// CHECK:STDOUT:       4:               inst5000005A
// CHECK:STDOUT:     inst_block5000001B:
// CHECK:STDOUT:       0:               inst5000005B
// CHECK:STDOUT:       1:               inst50000032
// CHECK:STDOUT:       2:               inst5000005C
// CHECK:STDOUT:       3:               inst5000005D
// CHECK:STDOUT:       4:               inst50000049
// CHECK:STDOUT:       5:               inst5000005E
// CHECK:STDOUT:       6:               inst5000005F
// CHECK:STDOUT:       7:               inst50000059
// CHECK:STDOUT:     inst_block5000001C:
// CHECK:STDOUT:       0:               inst50000032
// CHECK:STDOUT:     inst_block5000001D:
// CHECK:STDOUT:       0:               inst50000034
// CHECK:STDOUT:       1:               inst50000039
// CHECK:STDOUT:       2:               inst50000044
// CHECK:STDOUT:       3:               inst50000045
// CHECK:STDOUT:       4:               inst50000048
// CHECK:STDOUT:       5:               inst50000055
// CHECK:STDOUT:       6:               inst50000058
// CHECK:STDOUT:     inst_block5000001E:
// CHECK:STDOUT:       0:               inst50000033
// CHECK:STDOUT:       1:               inst50000038
// CHECK:STDOUT:       2:               inst50000040
// CHECK:STDOUT:       3:               inst50000043
// CHECK:STDOUT:       4:               inst50000046
// CHECK:STDOUT:       5:               inst50000054
// CHECK:STDOUT:       6:               inst50000056
// CHECK:STDOUT:     inst_block5000001F:
// CHECK:STDOUT:       0:               inst5000006A
// CHECK:STDOUT:     inst_block50000020:
// CHECK:STDOUT:       0:               inst50000066
// CHECK:STDOUT:       1:               inst50000069
// CHECK:STDOUT:     inst_block50000021:
// CHECK:STDOUT:       0:               inst5000006B
// CHECK:STDOUT:       1:               inst5000006C
// CHECK:STDOUT:     inst_block50000022:
// CHECK:STDOUT:       0:               inst50000071
// CHECK:STDOUT:     inst_block50000023:
// CHECK:STDOUT:       0:               inst50000072
// CHECK:STDOUT:       1:               inst50000073
// CHECK:STDOUT:       2:               inst50000076
// CHECK:STDOUT:       3:               inst50000077
// CHECK:STDOUT:       4:               inst5000007A
// CHECK:STDOUT:     inst_block50000024:
// CHECK:STDOUT:       0:               inst5000006F
// CHECK:STDOUT:     inst_block50000025:
// CHECK:STDOUT:       0:               inst50000079
// CHECK:STDOUT:     inst_block50000026:
// CHECK:STDOUT:       0:               inst50000083
// CHECK:STDOUT:       1:               inst50000084
// CHECK:STDOUT:       2:               inst50000085
// CHECK:STDOUT:       3:               inst50000086
// CHECK:STDOUT:       4:               inst50000088
// CHECK:STDOUT:     inst_block50000027:
// CHECK:STDOUT:       0:               inst50000082
// CHECK:STDOUT:       1:               inst5000008F
// CHECK:STDOUT:     inst_block50000028:
// CHECK:STDOUT:       0:               inst5000007F
// CHECK:STDOUT:       1:               inst5000008C
// CHECK:STDOUT:     inst_block50000029:
// CHECK:STDOUT:       0:               inst50000091
// CHECK:STDOUT:       1:               inst50000093
// CHECK:STDOUT:     inst_block5000002A:
// CHECK:STDOUT:       0:               inst50000071
// CHECK:STDOUT:       1:               inst5000007F
// CHECK:STDOUT:       2:               inst50000082
// CHECK:STDOUT:       3:               inst5000008C
// CHECK:STDOUT:       4:               inst5000008F
// CHECK:STDOUT:     inst_block5000002B:
// CHECK:STDOUT:       0:               inst50000090
// CHECK:STDOUT:       1:               inst5000006E
// CHECK:STDOUT:       2:               inst50000091
// CHECK:STDOUT:       3:               inst50000092
// CHECK:STDOUT:       4:               inst50000081
// CHECK:STDOUT:       5:               inst50000093
// CHECK:STDOUT:       6:               inst50000094
// CHECK:STDOUT:       7:               inst5000008E
// CHECK:STDOUT:     inst_block5000002C:
// CHECK:STDOUT:       0:               inst5000006E
// CHECK:STDOUT:     inst_block5000002D:
// CHECK:STDOUT:       0:               inst50000070
// CHECK:STDOUT:       1:               inst50000075
// CHECK:STDOUT:       2:               inst5000007C
// CHECK:STDOUT:       3:               inst5000007D
// CHECK:STDOUT:       4:               inst50000080
// CHECK:STDOUT:       5:               inst5000008A
// CHECK:STDOUT:       6:               inst5000008D
// CHECK:STDOUT:     inst_block5000002E:
// CHECK:STDOUT:       0:               inst5000006F
// CHECK:STDOUT:       1:               inst50000074
// CHECK:STDOUT:       2:               inst50000079
// CHECK:STDOUT:       3:               inst5000007B
// CHECK:STDOUT:       4:               inst5000007E
// CHECK:STDOUT:       5:               inst50000089
// CHECK:STDOUT:       6:               inst5000008B
// CHECK:STDOUT:     inst_block5000002F:
// CHECK:STDOUT:       0:               inst5000009E
// CHECK:STDOUT:     inst_block50000030:
// CHECK:STDOUT:       0:               inst5000009A
// CHECK:STDOUT:       1:               inst5000009D
// CHECK:STDOUT:     inst_block50000031:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst50000010
// CHECK:STDOUT:       2:               inst50000012
// CHECK:STDOUT:       3:               inst50000060
// CHECK:STDOUT:       4:               inst50000095
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     NonCore
// CHECK:STDOUT:         identifier1:     Copy
// CHECK:STDOUT:         identifier2:     T1
// CHECK:STDOUT:         identifier3:     T2
// CHECK:STDOUT:         identifier4:     UseNonCoreDestroy
// CHECK:STDOUT:         identifier5:     T
// CHECK:STDOUT:         identifier6:     Destroy
// CHECK:STDOUT:         identifier7:     UseLocalCopy
// CHECK:STDOUT:       strings:         {}
// CHECK:STDOUT: ...

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
  let T1: type;
  let T2: type;
}

// --- main.carbon
import NonCore;

interface Copy {
  let T1: type;
  let T2: type;
}

//@dump-sem-ir-begin
fn UseNonCoreDestroy[T: NonCore.Destroy](_: T.T1, _: T.T2) {}
fn UseLocalCopy[T: Copy](_: T.T1, _: T.T2) {}
fn HasRewrites[T: Copy where .Self impls NonCore.Destroy and .T1 = ()](_: T.T1, _: T.T2) {}
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
// CHECK:STDOUT:     name_scope0:     {inst: inst(Package), parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst60000010}}
// CHECK:STDOUT:     name_scope60000001: {inst: inst60000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name(SelfType): inst60000012}}
// CHECK:STDOUT:     name_scope60000002: {inst: inst60000015, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {name2: inst60000019, name3: inst6000001D}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name(SelfType), parent_scope: name_scope60000001, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
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
// CHECK:STDOUT:     generic60000001: {decl: inst60000015, bindings: inst_block60000006, self_specific_id: specific60000000, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific60000000: {generic: generic60000001, args: inst_block60000007, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
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
// CHECK:STDOUT:   declared_facet_types:
// CHECK:STDOUT:     'declared_facet_type(Empty)': {}
// CHECK:STDOUT:     declared_facet_type60000001: {extends interface: [interface60000000]}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: FacetType, arg0: declared_facet_type(Empty), type: type(TypeType)}
// CHECK:STDOUT:     'inst(Package)':   {kind: Namespace, arg0: name_scope0, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000010:    {kind: InterfaceDecl, arg0: interface60000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000011:    {kind: FacetType, arg0: declared_facet_type60000001, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000013:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000014:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000015:    {kind: InterfaceWithSelfDecl, arg0: interface60000000}
// CHECK:STDOUT:     inst60000016:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst60000017:    {kind: AssociatedConstantDecl, arg0: assoc_const60000000, arg1: inst_block6000000B, type: type(TypeType)}
// CHECK:STDOUT:     inst60000018:    {kind: AssociatedEntityType, arg0: interface60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000019:    {kind: AssociatedEntity, arg0: element0, arg1: inst60000017, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001A:    {kind: AssociatedEntity, arg0: element0, arg1: inst60000017, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001B:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst6000001C:    {kind: AssociatedConstantDecl, arg0: assoc_const60000001, arg1: inst_block6000000D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: AssociatedEntity, arg0: element1, arg1: inst6000001C, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001E:    {kind: AssociatedEntity, arg0: element1, arg1: inst6000001C, type: type(inst60000018)}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(Package)':   concrete_constant(inst(Package))
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
// CHECK:STDOUT:     extra_global_variables: {}
// CHECK:STDOUT:     inst_block60000006:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:     inst_block60000007:
// CHECK:STDOUT:       0:               inst60000013
// CHECK:STDOUT:     inst_block60000008:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:       1:               inst60000015
// CHECK:STDOUT:     inst_block60000009:
// CHECK:STDOUT:       0:               inst60000017
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:     inst_block6000000A:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:     inst_block6000000B:
// CHECK:STDOUT:       0:               inst60000019
// CHECK:STDOUT:     inst_block6000000C:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:     inst_block6000000D:
// CHECK:STDOUT:       0:               inst6000001D
// CHECK:STDOUT:     inst_block6000000E:
// CHECK:STDOUT:       0:               inst60000017
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:     inst_block6000000F:
// CHECK:STDOUT:       0:               inst(Package)
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
// CHECK:STDOUT:     name8:           HasRewrites
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir50000002: {decl_id: inst50000010, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir50000002, inst_id: inst60000010}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir50000002, inst_id: inst60000015}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir50000002, inst_id: inst60000019}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir50000002, inst_id: inst6000001D}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir50000002, inst_id: inst60000017}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir50000002, inst_id: inst6000001C}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir50000002, inst_id: inst60000012}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   clang_decl_signatures: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst(Package), parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst50000011, name1: inst50000012, name4: inst5000006D, name7: inst500000B0, name8: inst5000011B}}
// CHECK:STDOUT:     name_scope50000001: {inst: inst50000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name6: inst50000024}}
// CHECK:STDOUT:     name_scope50000002: {inst: inst50000012, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name(SelfType): inst50000014}}
// CHECK:STDOUT:     name_scope50000003: {inst: inst50000017, parent_scope: name_scope50000002, has_error: false, extended_scopes: [], names: {name2: inst5000001B, name3: inst5000001F}}
// CHECK:STDOUT:     name_scope50000004: {inst: inst50000025, parent_scope: name_scope50000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst5000002E}}
// CHECK:STDOUT:     name_scope50000005: {inst: inst50000027, parent_scope: name_scope50000004, has_error: false, extended_scopes: [], names: {name2: inst50000029, name3: inst5000002A}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name50000000: {name: name(SelfType), parent_scope: name_scope50000002, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000001: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, is_frozen_period_self: 1, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000002: {name: name6, parent_scope: name_scope50000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000003: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000004: {name: name2, parent_scope: name_scope50000005, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000005: {name: name3, parent_scope: name_scope50000005, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name50000007: {name: name5, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst5000002F}
// CHECK:STDOUT:     entity_name50000008: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>, type: inst50000047}
// CHECK:STDOUT:     entity_name50000009: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>, type: inst5000005C}
// CHECK:STDOUT:     entity_name5000000A: {name: name5, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst50000079}
// CHECK:STDOUT:     entity_name5000000B: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>, type: inst5000008D}
// CHECK:STDOUT:     entity_name5000000C: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>, type: inst5000009F}
// CHECK:STDOUT:     entity_name5000000D: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, is_frozen_period_self: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name5000000E: {name: name5, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst500000E0}
// CHECK:STDOUT:     entity_name5000000F: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>, type: inst500000FD}
// CHECK:STDOUT:     entity_name50000010: {name: name(Underscore), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>, type: inst5000010A}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function50000000: {name: name4, parent_scope: name_scope0, call_param_patterns_id: inst_block50000019, call_params_id: inst_block5000001A, body: [inst_block50000020]}
// CHECK:STDOUT:     function50000001: {name: name7, parent_scope: name_scope0, call_param_patterns_id: inst_block50000029, call_params_id: inst_block5000002A, body: [inst_block50000030]}
// CHECK:STDOUT:     function50000002: {name: name8, parent_scope: name_scope0, call_param_patterns_id: inst_block5000003F, call_params_id: inst_block50000040, body: [inst_block50000046]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface50000000: {name: name1, parent_scope: name_scope0, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:     interface50000001: {name: name6, parent_scope: name_scope50000001, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:   associated_constants:
// CHECK:STDOUT:     assoc_const50000000: {name: name2, parent_scope: name_scope50000003, decl_id: inst50000019, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const50000001: {name: name3, parent_scope: name_scope50000003, decl_id: inst5000001E, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const50000002: {name: name2, parent_scope: name_scope50000005, decl_id: inst50000046, default_value_id: inst<none>}
// CHECK:STDOUT:     assoc_const50000003: {name: name3, parent_scope: name_scope50000005, decl_id: inst5000005B, default_value_id: inst<none>}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic50000000: {decl: inst<none>, bindings: inst_block<none>, self_specific_id: specific<none>, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic50000001: {decl: inst50000017, bindings: inst_block50000006, self_specific_id: specific50000000, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic50000002: {decl: inst50000027, bindings: inst_block50000011, self_specific_id: specific50000001, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic50000003: {decl: inst5000006D, bindings: inst_block5000001D, self_specific_id: specific50000003, decl_block_id: inst_block5000001E, definition_block_id: inst_block50000021}
// CHECK:STDOUT:     generic50000004: {decl: inst500000B0, bindings: inst_block5000002D, self_specific_id: specific50000005, decl_block_id: inst_block5000002E, definition_block_id: inst_block50000031}
// CHECK:STDOUT:     generic50000005: {decl: inst5000011B, bindings: inst_block50000043, self_specific_id: specific50000009, decl_block_id: inst_block50000044, definition_block_id: inst_block50000047}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific50000000: {generic: generic50000001, args: inst_block50000007, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000001: {generic: generic50000002, args: inst_block50000012, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000002: {generic: generic50000002, args: inst_block50000015, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000003: {generic: generic50000003, args: inst_block50000015, identified: 0, decl_block_id: inst_block5000001F, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific50000004: {generic: generic50000001, args: inst_block50000025, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000005: {generic: generic50000004, args: inst_block50000025, identified: 0, decl_block_id: inst_block5000002F, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific50000006: {generic: generic50000001, args: inst_block50000033, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000007: {generic: generic50000001, args: inst_block50000039, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000008: {generic: generic50000001, args: inst_block5000003C, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000009: {generic: generic50000005, args: inst_block50000039, identified: 0, decl_block_id: inst_block50000045, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
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
// CHECK:STDOUT:     'type(inst50000026)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000026)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(RequireImplWitnessSpecificDefinitionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000006E)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000006F)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000006F)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000006F)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant5000000E)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant5000000E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(symbolic_constant50000016)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant50000016)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(inst50000013)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000013)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst500000B1)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000006F)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant5000002C)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant5000002C)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(symbolic_constant50000034)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant50000034)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(inst500000E1)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst500000E1)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000011C)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000006F)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant50000058)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant50000058)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:   declared_facet_types:
// CHECK:STDOUT:     'declared_facet_type(Empty)': {}
// CHECK:STDOUT:     declared_facet_type50000001: {extends interface: [interface50000000]}
// CHECK:STDOUT:     declared_facet_type50000002: {extends interface: [interface50000001]}
// CHECK:STDOUT:     declared_facet_type50000003: {extends interface: [interface50000000], self impls interface: [interface50000001], rewrites: {inst500000DE: inst5000006F}}
// CHECK:STDOUT:     declared_facet_type50000004: {extends interface: [interface50000000], self impls interface: [interface50000001], rewrites: {inst500000F4: inst5000006F}}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: FacetType, arg0: declared_facet_type(Empty), type: type(TypeType)}
// CHECK:STDOUT:     'inst(Package)':   {kind: Namespace, arg0: name_scope0, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000010:    {kind: ImportDecl, arg0: name0}
// CHECK:STDOUT:     inst50000011:    {kind: Namespace, arg0: name_scope50000001, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000012:    {kind: InterfaceDecl, arg0: interface50000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000013:    {kind: FacetType, arg0: declared_facet_type50000001, type: type(TypeType)}
// CHECK:STDOUT:     inst50000014:    {kind: SymbolicBinding, arg0: entity_name50000000, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000015:    {kind: SymbolicBinding, arg0: entity_name50000000, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000016:    {kind: SymbolicBinding, arg0: entity_name50000000, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000017:    {kind: InterfaceWithSelfDecl, arg0: interface50000000}
// CHECK:STDOUT:     inst50000018:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst50000019:    {kind: AssociatedConstantDecl, arg0: assoc_const50000000, arg1: inst_block5000000B, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001A:    {kind: AssociatedEntityType, arg0: interface50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001B:    {kind: AssociatedEntity, arg0: element0, arg1: inst50000019, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000001C:    {kind: AssociatedEntity, arg0: element0, arg1: inst50000019, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000001D:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst5000001E:    {kind: AssociatedConstantDecl, arg0: assoc_const50000001, arg1: inst_block5000000D, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001F:    {kind: AssociatedEntity, arg0: element1, arg1: inst5000001E, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000020:    {kind: AssociatedEntity, arg0: element1, arg1: inst5000001E, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000021:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000022:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000023:    {kind: NameRef, arg0: name0, arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000024:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name50000002, type: type(TypeType)}
// CHECK:STDOUT:     inst50000025:    {kind: InterfaceDecl, arg0: interface50000001, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000026:    {kind: FacetType, arg0: declared_facet_type50000002, type: type(TypeType)}
// CHECK:STDOUT:     inst50000027:    {kind: InterfaceWithSelfDecl, arg0: interface50000001}
// CHECK:STDOUT:     inst50000028:    {kind: SymbolicBinding, arg0: entity_name50000003, arg1: inst<none>, type: type(inst50000026)}
// CHECK:STDOUT:     inst50000029:    {kind: ImportRefLoaded, arg0: import_ir_inst2, arg1: entity_name<none>, type: type(inst5000003C)}
// CHECK:STDOUT:     inst5000002A:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst5000003C)}
// CHECK:STDOUT:     inst5000002B:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name50000004}
// CHECK:STDOUT:     inst5000002C:    {kind: ImportRefUnloaded, arg0: import_ir_inst5, arg1: entity_name50000005}
// CHECK:STDOUT:     inst5000002D:    {kind: ImportRefLoaded, arg0: import_ir_inst6, arg1: entity_name<none>, type: type(inst50000026)}
// CHECK:STDOUT:     inst5000002E:    {kind: ImportRefUnloaded, arg0: import_ir_inst6, arg1: entity_name<none>}
// CHECK:STDOUT:     inst5000002F:    {kind: NameRef, arg0: name6, arg1: inst50000024, type: type(TypeType)}
// CHECK:STDOUT:     inst50000030:    {kind: PatternType, arg0: inst50000026, type: type(TypeType)}
// CHECK:STDOUT:     inst50000031:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst50000030)}
// CHECK:STDOUT:     inst50000032:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst50000030)}
// CHECK:STDOUT:     inst50000033:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst50000030)}
// CHECK:STDOUT:     inst50000034:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000026)}
// CHECK:STDOUT:     inst50000035:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000026)}
// CHECK:STDOUT:     inst50000036:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000026)}
// CHECK:STDOUT:     inst50000037:    {kind: NameRef, arg0: name5, arg1: inst50000034, type: type(inst50000026)}
// CHECK:STDOUT:     inst50000038:    {kind: FacetAccessType, arg0: inst50000037, type: type(TypeType)}
// CHECK:STDOUT:     inst50000039:    {kind: FacetAccessType, arg0: inst50000035, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003A:    {kind: FacetAccessType, arg0: inst50000036, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003B:    {kind: Converted, arg0: inst50000037, arg1: inst50000038, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003C:    {kind: AssociatedEntityType, arg0: interface50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003D:    {kind: ImportRefLoaded, arg0: import_ir_inst4, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003E:    {kind: AssociatedEntity, arg0: element0, arg1: inst5000003D, type: type(inst5000003C)}
// CHECK:STDOUT:     inst5000003F:    {kind: NameRef, arg0: name2, arg1: inst50000029, type: type(inst5000003C)}
// CHECK:STDOUT:     inst50000040:    {kind: LookupImplWitness, arg0: inst50000035, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000041:    {kind: LookupImplWitness, arg0: inst50000035, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000042:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000041, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst50000043:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000041, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst50000044:    {kind: LookupImplWitness, arg0: inst50000036, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000045:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000044, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst50000046:    {kind: AssociatedConstantDecl, arg0: assoc_const50000002, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000047:    {kind: ImplWitnessAccess, arg0: inst50000041, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000048:    {kind: ImplWitnessAccess, arg0: inst50000041, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000049:    {kind: ImplWitnessAccess, arg0: inst50000044, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004A:    {kind: PatternType, arg0: inst50000048, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004B:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000012)}
// CHECK:STDOUT:     inst5000004C:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000010)}
// CHECK:STDOUT:     inst5000004D:    {kind: PatternType, arg0: inst50000049, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004E:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000012)}
// CHECK:STDOUT:     inst5000004F:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst5000004B, type: type(symbolic_constant50000012)}
// CHECK:STDOUT:     inst50000050:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst5000004C, type: type(symbolic_constant50000010)}
// CHECK:STDOUT:     inst50000051:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst5000004E, type: type(symbolic_constant50000012)}
// CHECK:STDOUT:     inst50000052:    {kind: WrapperBinding, arg0: entity_name50000008, arg1: inst50000069, type: type(symbolic_constant5000000F)}
// CHECK:STDOUT:     inst50000053:    {kind: NameRef, arg0: name5, arg1: inst50000034, type: type(inst50000026)}
// CHECK:STDOUT:     inst50000054:    {kind: FacetAccessType, arg0: inst50000053, type: type(TypeType)}
// CHECK:STDOUT:     inst50000055:    {kind: Converted, arg0: inst50000053, arg1: inst50000054, type: type(TypeType)}
// CHECK:STDOUT:     inst50000056:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000057:    {kind: AssociatedEntity, arg0: element1, arg1: inst50000056, type: type(inst5000003C)}
// CHECK:STDOUT:     inst50000058:    {kind: NameRef, arg0: name3, arg1: inst5000002A, type: type(inst5000003C)}
// CHECK:STDOUT:     inst50000059:    {kind: LookupImplWitness, arg0: inst50000035, arg1: specific_interface50000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000005A:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000041, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst5000005B:    {kind: AssociatedConstantDecl, arg0: assoc_const50000003, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst5000005C:    {kind: ImplWitnessAccess, arg0: inst50000041, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000005D:    {kind: ImplWitnessAccess, arg0: inst50000041, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000005E:    {kind: ImplWitnessAccess, arg0: inst50000044, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000005F:    {kind: PatternType, arg0: inst5000005D, type: type(TypeType)}
// CHECK:STDOUT:     inst50000060:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000001A)}
// CHECK:STDOUT:     inst50000061:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000018)}
// CHECK:STDOUT:     inst50000062:    {kind: PatternType, arg0: inst5000005E, type: type(TypeType)}
// CHECK:STDOUT:     inst50000063:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000001A)}
// CHECK:STDOUT:     inst50000064:    {kind: WrapperBindingPattern, arg0: entity_name50000009, arg1: inst50000060, type: type(symbolic_constant5000001A)}
// CHECK:STDOUT:     inst50000065:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000061, type: type(symbolic_constant50000018)}
// CHECK:STDOUT:     inst50000066:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000063, type: type(symbolic_constant5000001A)}
// CHECK:STDOUT:     inst50000067:    {kind: WrapperBinding, arg0: entity_name50000009, arg1: inst5000006B, type: type(symbolic_constant50000017)}
// CHECK:STDOUT:     inst50000068:    {kind: SpliceBlock, arg0: inst_block5000000F, arg1: inst5000002F, type: type(TypeType)}
// CHECK:STDOUT:     inst50000069:    {kind: ValueParam, arg0: call_param0, arg1: name(Underscore), type: type(symbolic_constant5000000F)}
// CHECK:STDOUT:     inst5000006A:    {kind: SpliceBlock, arg0: inst_block50000014, arg1: inst50000047, type: type(TypeType)}
// CHECK:STDOUT:     inst5000006B:    {kind: ValueParam, arg0: call_param1, arg1: name(Underscore), type: type(symbolic_constant50000017)}
// CHECK:STDOUT:     inst5000006C:    {kind: SpliceBlock, arg0: inst_block50000017, arg1: inst5000005C, type: type(TypeType)}
// CHECK:STDOUT:     inst5000006D:    {kind: FunctionDecl, arg0: function50000000, arg1: inst_block5000001C, type: type(inst5000006E)}
// CHECK:STDOUT:     inst5000006E:    {kind: FunctionType, arg0: function50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000006F:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000070:    {kind: StructValue, arg0: inst_block_empty, type: type(inst5000006E)}
// CHECK:STDOUT:     inst50000071:    {kind: RequireCompleteType, arg0: inst50000048, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000072:    {kind: RequireCompleteType, arg0: inst50000048, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000073:    {kind: RequireCompleteType, arg0: inst50000049, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000074:    {kind: RequireCompleteType, arg0: inst5000005D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000075:    {kind: RequireCompleteType, arg0: inst5000005D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000076:    {kind: RequireCompleteType, arg0: inst5000005E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000077:    {kind: Return}
// CHECK:STDOUT:     inst50000078:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000079:    {kind: NameRef, arg0: name1, arg1: inst50000012, type: type(TypeType)}
// CHECK:STDOUT:     inst5000007A:    {kind: PatternType, arg0: inst50000013, type: type(TypeType)}
// CHECK:STDOUT:     inst5000007B:    {kind: SymbolicBindingPattern, arg0: entity_name5000000A, type: type(inst5000007A)}
// CHECK:STDOUT:     inst5000007C:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst5000007A)}
// CHECK:STDOUT:     inst5000007D:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst5000007A)}
// CHECK:STDOUT:     inst5000007E:    {kind: SymbolicBinding, arg0: entity_name5000000A, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst5000007F:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000080:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000081:    {kind: NameRef, arg0: name5, arg1: inst5000007E, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000082:    {kind: FacetAccessType, arg0: inst50000081, type: type(TypeType)}
// CHECK:STDOUT:     inst50000083:    {kind: FacetAccessType, arg0: inst5000007F, type: type(TypeType)}
// CHECK:STDOUT:     inst50000084:    {kind: FacetAccessType, arg0: inst50000080, type: type(TypeType)}
// CHECK:STDOUT:     inst50000085:    {kind: Converted, arg0: inst50000081, arg1: inst50000082, type: type(TypeType)}
// CHECK:STDOUT:     inst50000086:    {kind: NameRef, arg0: name2, arg1: inst5000001B, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000087:    {kind: LookupImplWitness, arg0: inst5000007F, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000088:    {kind: LookupImplWitness, arg0: inst5000007F, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000089:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000088, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst5000008A:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000088, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst5000008B:    {kind: LookupImplWitness, arg0: inst50000080, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000008C:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst5000008B, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst5000008D:    {kind: ImplWitnessAccess, arg0: inst50000088, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst5000008E:    {kind: ImplWitnessAccess, arg0: inst50000088, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst5000008F:    {kind: ImplWitnessAccess, arg0: inst5000008B, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000090:    {kind: PatternType, arg0: inst5000008E, type: type(TypeType)}
// CHECK:STDOUT:     inst50000091:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000030)}
// CHECK:STDOUT:     inst50000092:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000002E)}
// CHECK:STDOUT:     inst50000093:    {kind: PatternType, arg0: inst5000008F, type: type(TypeType)}
// CHECK:STDOUT:     inst50000094:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000030)}
// CHECK:STDOUT:     inst50000095:    {kind: WrapperBindingPattern, arg0: entity_name5000000B, arg1: inst50000091, type: type(symbolic_constant50000030)}
// CHECK:STDOUT:     inst50000096:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000092, type: type(symbolic_constant5000002E)}
// CHECK:STDOUT:     inst50000097:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000094, type: type(symbolic_constant50000030)}
// CHECK:STDOUT:     inst50000098:    {kind: WrapperBinding, arg0: entity_name5000000B, arg1: inst500000AC, type: type(symbolic_constant5000002D)}
// CHECK:STDOUT:     inst50000099:    {kind: NameRef, arg0: name5, arg1: inst5000007E, type: type(inst50000013)}
// CHECK:STDOUT:     inst5000009A:    {kind: FacetAccessType, arg0: inst50000099, type: type(TypeType)}
// CHECK:STDOUT:     inst5000009B:    {kind: Converted, arg0: inst50000099, arg1: inst5000009A, type: type(TypeType)}
// CHECK:STDOUT:     inst5000009C:    {kind: NameRef, arg0: name3, arg1: inst5000001F, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000009D:    {kind: LookupImplWitness, arg0: inst5000007F, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000009E:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst50000088, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst5000009F:    {kind: ImplWitnessAccess, arg0: inst50000088, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst500000A0:    {kind: ImplWitnessAccess, arg0: inst50000088, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst500000A1:    {kind: ImplWitnessAccess, arg0: inst5000008B, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst500000A2:    {kind: PatternType, arg0: inst500000A0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000A3:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000038)}
// CHECK:STDOUT:     inst500000A4:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000036)}
// CHECK:STDOUT:     inst500000A5:    {kind: PatternType, arg0: inst500000A1, type: type(TypeType)}
// CHECK:STDOUT:     inst500000A6:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant50000038)}
// CHECK:STDOUT:     inst500000A7:    {kind: WrapperBindingPattern, arg0: entity_name5000000C, arg1: inst500000A3, type: type(symbolic_constant50000038)}
// CHECK:STDOUT:     inst500000A8:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst500000A4, type: type(symbolic_constant50000036)}
// CHECK:STDOUT:     inst500000A9:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst500000A6, type: type(symbolic_constant50000038)}
// CHECK:STDOUT:     inst500000AA:    {kind: WrapperBinding, arg0: entity_name5000000C, arg1: inst500000AE, type: type(symbolic_constant50000035)}
// CHECK:STDOUT:     inst500000AB:    {kind: SpliceBlock, arg0: inst_block50000022, arg1: inst50000079, type: type(TypeType)}
// CHECK:STDOUT:     inst500000AC:    {kind: ValueParam, arg0: call_param0, arg1: name(Underscore), type: type(symbolic_constant5000002D)}
// CHECK:STDOUT:     inst500000AD:    {kind: SpliceBlock, arg0: inst_block50000024, arg1: inst5000008D, type: type(TypeType)}
// CHECK:STDOUT:     inst500000AE:    {kind: ValueParam, arg0: call_param1, arg1: name(Underscore), type: type(symbolic_constant50000035)}
// CHECK:STDOUT:     inst500000AF:    {kind: SpliceBlock, arg0: inst_block50000027, arg1: inst5000009F, type: type(TypeType)}
// CHECK:STDOUT:     inst500000B0:    {kind: FunctionDecl, arg0: function50000001, arg1: inst_block5000002C, type: type(inst500000B1)}
// CHECK:STDOUT:     inst500000B1:    {kind: FunctionType, arg0: function50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst500000B2:    {kind: StructValue, arg0: inst_block_empty, type: type(inst500000B1)}
// CHECK:STDOUT:     inst500000B3:    {kind: RequireCompleteType, arg0: inst5000008E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000B4:    {kind: RequireCompleteType, arg0: inst5000008E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000B5:    {kind: RequireCompleteType, arg0: inst5000008F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000B6:    {kind: RequireCompleteType, arg0: inst500000A0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000B7:    {kind: RequireCompleteType, arg0: inst500000A0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000B8:    {kind: RequireCompleteType, arg0: inst500000A1, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000B9:    {kind: Return}
// CHECK:STDOUT:     inst500000BA:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst500000BB:    {kind: NameRef, arg0: name1, arg1: inst50000012, type: type(TypeType)}
// CHECK:STDOUT:     inst500000BC:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000BD:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000BE:    {kind: RequirementBaseFacetType, arg0: inst500000BB}
// CHECK:STDOUT:     inst500000BF:    {kind: NameRef, arg0: name(PeriodSelf), arg1: inst500000BC, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000C0:    {kind: NameRef, arg0: name0, arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst500000C1:    {kind: NameRef, arg0: name6, arg1: inst50000024, type: type(TypeType)}
// CHECK:STDOUT:     inst500000C2:    {kind: FacetAccessType, arg0: inst500000BF, type: type(TypeType)}
// CHECK:STDOUT:     inst500000C3:    {kind: FacetAccessType, arg0: inst500000BD, type: type(TypeType)}
// CHECK:STDOUT:     inst500000C4:    {kind: Converted, arg0: inst500000BF, arg1: inst500000C2, type: type(TypeType)}
// CHECK:STDOUT:     inst500000C5:    {kind: RequirementImpls, arg0: inst500000C4, arg1: inst500000C1}
// CHECK:STDOUT:     inst500000C6:    {kind: NameRef, arg0: name(PeriodSelf), arg1: inst500000BC, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000C7:    {kind: FacetAccessType, arg0: inst500000C6, type: type(TypeType)}
// CHECK:STDOUT:     inst500000C8:    {kind: Converted, arg0: inst500000C6, arg1: inst500000C7, type: type(TypeType)}
// CHECK:STDOUT:     inst500000C9:    {kind: NameRef, arg0: name2, arg1: inst5000001B, type: type(inst5000001A)}
// CHECK:STDOUT:     inst500000CA:    {kind: SymbolicBinding, arg0: entity_name5000000D, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000CB:    {kind: LookupImplWitness, arg0: inst500000BD, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000CC:    {kind: LookupImplWitness, arg0: inst500000BD, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000CD:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst500000CC, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst500000CE:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst500000CC, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst500000CF:    {kind: ImplWitnessAccess, arg0: inst500000CC, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000D0:    {kind: ImplWitnessAccess, arg0: inst500000CC, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000D1:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst5000006F)}
// CHECK:STDOUT:     inst500000D2:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst5000006F)}
// CHECK:STDOUT:     inst500000D3:    {kind: Converted, arg0: inst500000D1, arg1: inst5000006F, type: type(TypeType)}
// CHECK:STDOUT:     inst500000D4:    {kind: RequirementRewrite, arg0: inst500000CF, arg1: inst500000D3}
// CHECK:STDOUT:     inst500000D5:    {kind: SymbolicBinding, arg0: entity_name5000000D, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000D6:    {kind: NameRef, arg0: name(PeriodSelf), arg1: inst500000D5, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000D7:    {kind: FacetAccessType, arg0: inst500000D6, type: type(TypeType)}
// CHECK:STDOUT:     inst500000D8:    {kind: FacetAccessType, arg0: inst500000CA, type: type(TypeType)}
// CHECK:STDOUT:     inst500000D9:    {kind: Converted, arg0: inst500000BF, arg1: inst500000D7, type: type(TypeType)}
// CHECK:STDOUT:     inst500000DA:    {kind: RequirementImpls, arg0: inst500000D9, arg1: inst500000C1}
// CHECK:STDOUT:     inst500000DB:    {kind: LookupImplWitness, arg0: inst500000CA, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000DC:    {kind: LookupImplWitness, arg0: inst500000CA, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000DD:    {kind: ImplWitnessAccess, arg0: inst500000DC, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000DE:    {kind: ImplWitnessAccess, arg0: inst500000DC, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000DF:    {kind: RequirementRewrite, arg0: inst500000DD, arg1: inst500000D3}
// CHECK:STDOUT:     inst500000E0:    {kind: WhereExpr, arg0: inst_block50000036, type: type(TypeType)}
// CHECK:STDOUT:     inst500000E1:    {kind: FacetType, arg0: declared_facet_type50000003, type: type(TypeType)}
// CHECK:STDOUT:     inst500000E2:    {kind: PatternType, arg0: inst500000E1, type: type(TypeType)}
// CHECK:STDOUT:     inst500000E3:    {kind: SymbolicBindingPattern, arg0: entity_name5000000E, type: type(inst500000E2)}
// CHECK:STDOUT:     inst500000E4:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst500000E2)}
// CHECK:STDOUT:     inst500000E5:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst500000E2)}
// CHECK:STDOUT:     inst500000E6:    {kind: SymbolicBinding, arg0: entity_name5000000E, arg1: inst<none>, type: type(inst500000E1)}
// CHECK:STDOUT:     inst500000E7:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst500000E1)}
// CHECK:STDOUT:     inst500000E8:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst500000E1)}
// CHECK:STDOUT:     inst500000E9:    {kind: NameRef, arg0: name5, arg1: inst500000E6, type: type(inst500000E1)}
// CHECK:STDOUT:     inst500000EA:    {kind: FacetAccessType, arg0: inst500000E9, type: type(TypeType)}
// CHECK:STDOUT:     inst500000EB:    {kind: FacetAccessType, arg0: inst500000E7, type: type(TypeType)}
// CHECK:STDOUT:     inst500000EC:    {kind: FacetAccessType, arg0: inst500000E8, type: type(TypeType)}
// CHECK:STDOUT:     inst500000ED:    {kind: Converted, arg0: inst500000E9, arg1: inst500000EA, type: type(TypeType)}
// CHECK:STDOUT:     inst500000EE:    {kind: NameRef, arg0: name2, arg1: inst5000001B, type: type(inst5000001A)}
// CHECK:STDOUT:     inst500000EF:    {kind: LookupImplWitness, arg0: inst500000E7, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000F0:    {kind: IdentifiedWitness, arg0: inst500000E7, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000F1:    {kind: FacetValue, arg0: inst500000EB, arg1: inst_block5000003A, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000F2:    {kind: LookupImplWitness, arg0: inst500000F1, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000F3:    {kind: ImplWitnessAccess, arg0: inst500000F0, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000F4:    {kind: ImplWitnessAccess, arg0: inst500000F0, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000F5:    {kind: FacetType, arg0: declared_facet_type50000004, type: type(TypeType)}
// CHECK:STDOUT:     inst500000F6:    {kind: LookupImplWitness, arg0: inst500000E7, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000F7:    {kind: LookupImplWitness, arg0: inst500000E7, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000F8:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst500000F7, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst500000F9:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst500000F7, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst500000FA:    {kind: LookupImplWitness, arg0: inst500000E8, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst500000FB:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst500000FA, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst500000FC:    {kind: FacetValue, arg0: inst500000EB, arg1: inst_block5000003B, type: type(inst50000013)}
// CHECK:STDOUT:     inst500000FD:    {kind: ImplWitnessAccess, arg0: inst500000F7, arg1: element0, type: type(TypeType)}
// CHECK:STDOUT:     inst500000FE:    {kind: PatternType, arg0: inst5000006F, type: type(TypeType)}
// CHECK:STDOUT:     inst500000FF:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(inst500000FE)}
// CHECK:STDOUT:     inst50000100:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(inst500000FE)}
// CHECK:STDOUT:     inst50000101:    {kind: WrapperBindingPattern, arg0: entity_name5000000F, arg1: inst500000FF, type: type(inst500000FE)}
// CHECK:STDOUT:     inst50000102:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000100, type: type(inst500000FE)}
// CHECK:STDOUT:     inst50000103:    {kind: WrapperBinding, arg0: entity_name5000000F, arg1: inst50000117, type: type(inst5000006F)}
// CHECK:STDOUT:     inst50000104:    {kind: NameRef, arg0: name5, arg1: inst500000E6, type: type(inst500000E1)}
// CHECK:STDOUT:     inst50000105:    {kind: FacetAccessType, arg0: inst50000104, type: type(TypeType)}
// CHECK:STDOUT:     inst50000106:    {kind: Converted, arg0: inst50000104, arg1: inst50000105, type: type(TypeType)}
// CHECK:STDOUT:     inst50000107:    {kind: NameRef, arg0: name3, arg1: inst5000001F, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000108:    {kind: LookupImplWitness, arg0: inst500000E7, arg1: specific_interface50000001, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000109:    {kind: RequireImplWitnessSpecificDefinition, arg0: inst500000F7, type: type(inst(RequireImplWitnessSpecificDefinitionType))}
// CHECK:STDOUT:     inst5000010A:    {kind: ImplWitnessAccess, arg0: inst500000F7, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000010B:    {kind: ImplWitnessAccess, arg0: inst500000F7, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000010C:    {kind: ImplWitnessAccess, arg0: inst500000FA, arg1: element1, type: type(TypeType)}
// CHECK:STDOUT:     inst5000010D:    {kind: PatternType, arg0: inst5000010B, type: type(TypeType)}
// CHECK:STDOUT:     inst5000010E:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000005C)}
// CHECK:STDOUT:     inst5000010F:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000005A)}
// CHECK:STDOUT:     inst50000110:    {kind: PatternType, arg0: inst5000010C, type: type(TypeType)}
// CHECK:STDOUT:     inst50000111:    {kind: ValueParamPattern, arg0: name(Underscore), type: type(symbolic_constant5000005C)}
// CHECK:STDOUT:     inst50000112:    {kind: WrapperBindingPattern, arg0: entity_name50000010, arg1: inst5000010E, type: type(symbolic_constant5000005C)}
// CHECK:STDOUT:     inst50000113:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst5000010F, type: type(symbolic_constant5000005A)}
// CHECK:STDOUT:     inst50000114:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst50000111, type: type(symbolic_constant5000005C)}
// CHECK:STDOUT:     inst50000115:    {kind: WrapperBinding, arg0: entity_name50000010, arg1: inst50000119, type: type(symbolic_constant50000059)}
// CHECK:STDOUT:     inst50000116:    {kind: SpliceBlock, arg0: inst_block50000032, arg1: inst500000E0, type: type(TypeType)}
// CHECK:STDOUT:     inst50000117:    {kind: ValueParam, arg0: call_param0, arg1: name(Underscore), type: type(inst5000006F)}
// CHECK:STDOUT:     inst50000118:    {kind: SpliceBlock, arg0: inst_block50000038, arg1: inst500000FD, type: type(TypeType)}
// CHECK:STDOUT:     inst50000119:    {kind: ValueParam, arg0: call_param1, arg1: name(Underscore), type: type(symbolic_constant50000059)}
// CHECK:STDOUT:     inst5000011A:    {kind: SpliceBlock, arg0: inst_block5000003D, arg1: inst5000010A, type: type(TypeType)}
// CHECK:STDOUT:     inst5000011B:    {kind: FunctionDecl, arg0: function50000002, arg1: inst_block50000042, type: type(inst5000011C)}
// CHECK:STDOUT:     inst5000011C:    {kind: FunctionType, arg0: function50000002, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000011D:    {kind: StructValue, arg0: inst_block_empty, type: type(inst5000011C)}
// CHECK:STDOUT:     inst5000011E:    {kind: RequireCompleteType, arg0: inst5000010B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000011F:    {kind: RequireCompleteType, arg0: inst5000010B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000120:    {kind: RequireCompleteType, arg0: inst5000010C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000121:    {kind: Return}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(Package)':   concrete_constant(inst(Package))
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
// CHECK:STDOUT:       inst50000021:    symbolic_constant50000002
// CHECK:STDOUT:       inst50000022:    symbolic_constant50000002
// CHECK:STDOUT:       inst50000023:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000024:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst50000025:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst50000026:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst50000027:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst50000028:    symbolic_constant50000003
// CHECK:STDOUT:       inst50000029:    concrete_constant(inst5000003E)
// CHECK:STDOUT:       inst5000002A:    concrete_constant(inst50000057)
// CHECK:STDOUT:       inst5000002B:    constant<none>
// CHECK:STDOUT:       inst5000002C:    constant<none>
// CHECK:STDOUT:       inst5000002D:    symbolic_constant50000003
// CHECK:STDOUT:       inst5000002E:    constant<none>
// CHECK:STDOUT:       inst5000002F:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst50000030:    concrete_constant(inst50000030)
// CHECK:STDOUT:       inst50000031:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000032:    symbolic_constant50000004
// CHECK:STDOUT:       inst50000033:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000034:    symbolic_constant50000007
// CHECK:STDOUT:       inst50000035:    symbolic_constant50000006
// CHECK:STDOUT:       inst50000036:    symbolic_constant50000007
// CHECK:STDOUT:       inst50000037:    symbolic_constant50000007
// CHECK:STDOUT:       inst50000038:    symbolic_constant50000009
// CHECK:STDOUT:       inst50000039:    symbolic_constant50000008
// CHECK:STDOUT:       inst5000003A:    symbolic_constant50000009
// CHECK:STDOUT:       inst5000003B:    symbolic_constant50000009
// CHECK:STDOUT:       inst5000003C:    concrete_constant(inst5000003C)
// CHECK:STDOUT:       inst5000003D:    concrete_constant(inst50000046)
// CHECK:STDOUT:       inst5000003E:    concrete_constant(inst5000003E)
// CHECK:STDOUT:       inst5000003F:    concrete_constant(inst5000003E)
// CHECK:STDOUT:       inst50000041:    symbolic_constant5000000A
// CHECK:STDOUT:       inst50000042:    symbolic_constant5000000D
// CHECK:STDOUT:       inst50000043:    symbolic_constant5000000B
// CHECK:STDOUT:       inst50000044:    symbolic_constant5000000C
// CHECK:STDOUT:       inst50000045:    symbolic_constant5000000D
// CHECK:STDOUT:       inst50000046:    concrete_constant(inst50000046)
// CHECK:STDOUT:       inst50000047:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000048:    symbolic_constant5000000E
// CHECK:STDOUT:       inst50000049:    symbolic_constant5000000F
// CHECK:STDOUT:       inst5000004A:    symbolic_constant50000010
// CHECK:STDOUT:       inst5000004B:    symbolic_constant50000013
// CHECK:STDOUT:       inst5000004C:    symbolic_constant50000011
// CHECK:STDOUT:       inst5000004D:    symbolic_constant50000012
// CHECK:STDOUT:       inst5000004E:    symbolic_constant50000013
// CHECK:STDOUT:       inst5000004F:    symbolic_constant50000015
// CHECK:STDOUT:       inst50000050:    symbolic_constant50000014
// CHECK:STDOUT:       inst50000051:    symbolic_constant50000015
// CHECK:STDOUT:       inst50000053:    symbolic_constant50000007
// CHECK:STDOUT:       inst50000054:    symbolic_constant50000009
// CHECK:STDOUT:       inst50000055:    symbolic_constant50000009
// CHECK:STDOUT:       inst50000056:    concrete_constant(inst5000005B)
// CHECK:STDOUT:       inst50000057:    concrete_constant(inst50000057)
// CHECK:STDOUT:       inst50000058:    concrete_constant(inst50000057)
// CHECK:STDOUT:       inst5000005A:    symbolic_constant5000000D
// CHECK:STDOUT:       inst5000005B:    concrete_constant(inst5000005B)
// CHECK:STDOUT:       inst5000005C:    symbolic_constant50000017
// CHECK:STDOUT:       inst5000005D:    symbolic_constant50000016
// CHECK:STDOUT:       inst5000005E:    symbolic_constant50000017
// CHECK:STDOUT:       inst5000005F:    symbolic_constant50000018
// CHECK:STDOUT:       inst50000060:    symbolic_constant5000001B
// CHECK:STDOUT:       inst50000061:    symbolic_constant50000019
// CHECK:STDOUT:       inst50000062:    symbolic_constant5000001A
// CHECK:STDOUT:       inst50000063:    symbolic_constant5000001B
// CHECK:STDOUT:       inst50000064:    symbolic_constant5000001D
// CHECK:STDOUT:       inst50000065:    symbolic_constant5000001C
// CHECK:STDOUT:       inst50000066:    symbolic_constant5000001D
// CHECK:STDOUT:       inst50000068:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst5000006A:    symbolic_constant5000000F
// CHECK:STDOUT:       inst5000006C:    symbolic_constant50000017
// CHECK:STDOUT:       inst5000006D:    concrete_constant(inst50000070)
// CHECK:STDOUT:       inst5000006E:    concrete_constant(inst5000006E)
// CHECK:STDOUT:       inst5000006F:    concrete_constant(inst5000006F)
// CHECK:STDOUT:       inst50000070:    concrete_constant(inst50000070)
// CHECK:STDOUT:       inst50000071:    symbolic_constant5000001F
// CHECK:STDOUT:       inst50000072:    symbolic_constant5000001E
// CHECK:STDOUT:       inst50000073:    symbolic_constant5000001F
// CHECK:STDOUT:       inst50000074:    symbolic_constant50000021
// CHECK:STDOUT:       inst50000075:    symbolic_constant50000020
// CHECK:STDOUT:       inst50000076:    symbolic_constant50000021
// CHECK:STDOUT:       inst50000078:    symbolic_constant50000002
// CHECK:STDOUT:       inst50000079:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst5000007A:    concrete_constant(inst5000007A)
// CHECK:STDOUT:       inst5000007B:    symbolic_constant50000023
// CHECK:STDOUT:       inst5000007C:    symbolic_constant50000022
// CHECK:STDOUT:       inst5000007D:    symbolic_constant50000023
// CHECK:STDOUT:       inst5000007E:    symbolic_constant50000025
// CHECK:STDOUT:       inst5000007F:    symbolic_constant50000024
// CHECK:STDOUT:       inst50000080:    symbolic_constant50000025
// CHECK:STDOUT:       inst50000081:    symbolic_constant50000025
// CHECK:STDOUT:       inst50000082:    symbolic_constant50000027
// CHECK:STDOUT:       inst50000083:    symbolic_constant50000026
// CHECK:STDOUT:       inst50000084:    symbolic_constant50000027
// CHECK:STDOUT:       inst50000085:    symbolic_constant50000027
// CHECK:STDOUT:       inst50000086:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst50000088:    symbolic_constant50000028
// CHECK:STDOUT:       inst50000089:    symbolic_constant5000002B
// CHECK:STDOUT:       inst5000008A:    symbolic_constant50000029
// CHECK:STDOUT:       inst5000008B:    symbolic_constant5000002A
// CHECK:STDOUT:       inst5000008C:    symbolic_constant5000002B
// CHECK:STDOUT:       inst5000008D:    symbolic_constant5000002D
// CHECK:STDOUT:       inst5000008E:    symbolic_constant5000002C
// CHECK:STDOUT:       inst5000008F:    symbolic_constant5000002D
// CHECK:STDOUT:       inst50000090:    symbolic_constant5000002E
// CHECK:STDOUT:       inst50000091:    symbolic_constant50000031
// CHECK:STDOUT:       inst50000092:    symbolic_constant5000002F
// CHECK:STDOUT:       inst50000093:    symbolic_constant50000030
// CHECK:STDOUT:       inst50000094:    symbolic_constant50000031
// CHECK:STDOUT:       inst50000095:    symbolic_constant50000033
// CHECK:STDOUT:       inst50000096:    symbolic_constant50000032
// CHECK:STDOUT:       inst50000097:    symbolic_constant50000033
// CHECK:STDOUT:       inst50000099:    symbolic_constant50000025
// CHECK:STDOUT:       inst5000009A:    symbolic_constant50000027
// CHECK:STDOUT:       inst5000009B:    symbolic_constant50000027
// CHECK:STDOUT:       inst5000009C:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst5000009E:    symbolic_constant5000002B
// CHECK:STDOUT:       inst5000009F:    symbolic_constant50000035
// CHECK:STDOUT:       inst500000A0:    symbolic_constant50000034
// CHECK:STDOUT:       inst500000A1:    symbolic_constant50000035
// CHECK:STDOUT:       inst500000A2:    symbolic_constant50000036
// CHECK:STDOUT:       inst500000A3:    symbolic_constant50000039
// CHECK:STDOUT:       inst500000A4:    symbolic_constant50000037
// CHECK:STDOUT:       inst500000A5:    symbolic_constant50000038
// CHECK:STDOUT:       inst500000A6:    symbolic_constant50000039
// CHECK:STDOUT:       inst500000A7:    symbolic_constant5000003B
// CHECK:STDOUT:       inst500000A8:    symbolic_constant5000003A
// CHECK:STDOUT:       inst500000A9:    symbolic_constant5000003B
// CHECK:STDOUT:       inst500000AB:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst500000AD:    symbolic_constant5000002D
// CHECK:STDOUT:       inst500000AF:    symbolic_constant50000035
// CHECK:STDOUT:       inst500000B0:    concrete_constant(inst500000B2)
// CHECK:STDOUT:       inst500000B1:    concrete_constant(inst500000B1)
// CHECK:STDOUT:       inst500000B2:    concrete_constant(inst500000B2)
// CHECK:STDOUT:       inst500000B3:    symbolic_constant5000003D
// CHECK:STDOUT:       inst500000B4:    symbolic_constant5000003C
// CHECK:STDOUT:       inst500000B5:    symbolic_constant5000003D
// CHECK:STDOUT:       inst500000B6:    symbolic_constant5000003F
// CHECK:STDOUT:       inst500000B7:    symbolic_constant5000003E
// CHECK:STDOUT:       inst500000B8:    symbolic_constant5000003F
// CHECK:STDOUT:       inst500000BA:    symbolic_constant50000002
// CHECK:STDOUT:       inst500000BB:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst500000BC:    symbolic_constant50000040
// CHECK:STDOUT:       inst500000BD:    symbolic_constant50000040
// CHECK:STDOUT:       inst500000BE:    concrete_constant(inst500000BE)
// CHECK:STDOUT:       inst500000BF:    symbolic_constant50000040
// CHECK:STDOUT:       inst500000C0:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst500000C1:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst500000C2:    symbolic_constant50000041
// CHECK:STDOUT:       inst500000C3:    symbolic_constant50000041
// CHECK:STDOUT:       inst500000C4:    symbolic_constant50000041
// CHECK:STDOUT:       inst500000C5:    concrete_constant(inst500000C5)
// CHECK:STDOUT:       inst500000C6:    symbolic_constant50000040
// CHECK:STDOUT:       inst500000C7:    symbolic_constant50000041
// CHECK:STDOUT:       inst500000C8:    symbolic_constant50000041
// CHECK:STDOUT:       inst500000C9:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst500000CA:    symbolic_constant50000042
// CHECK:STDOUT:       inst500000CC:    symbolic_constant50000043
// CHECK:STDOUT:       inst500000CD:    symbolic_constant50000044
// CHECK:STDOUT:       inst500000CE:    symbolic_constant50000044
// CHECK:STDOUT:       inst500000CF:    symbolic_constant50000045
// CHECK:STDOUT:       inst500000D0:    symbolic_constant50000045
// CHECK:STDOUT:       inst500000D1:    concrete_constant(inst500000D2)
// CHECK:STDOUT:       inst500000D2:    concrete_constant(inst500000D2)
// CHECK:STDOUT:       inst500000D3:    concrete_constant(inst5000006F)
// CHECK:STDOUT:       inst500000D4:    concrete_constant(inst500000D4)
// CHECK:STDOUT:       inst500000D5:    symbolic_constant50000042
// CHECK:STDOUT:       inst500000D6:    symbolic_constant50000042
// CHECK:STDOUT:       inst500000D7:    symbolic_constant50000046
// CHECK:STDOUT:       inst500000D8:    symbolic_constant50000046
// CHECK:STDOUT:       inst500000D9:    symbolic_constant50000046
// CHECK:STDOUT:       inst500000DA:    concrete_constant(inst500000DA)
// CHECK:STDOUT:       inst500000DC:    symbolic_constant50000047
// CHECK:STDOUT:       inst500000DD:    symbolic_constant50000048
// CHECK:STDOUT:       inst500000DE:    symbolic_constant50000048
// CHECK:STDOUT:       inst500000DF:    concrete_constant(inst500000DF)
// CHECK:STDOUT:       inst500000E0:    concrete_constant(inst500000E1)
// CHECK:STDOUT:       inst500000E1:    concrete_constant(inst500000E1)
// CHECK:STDOUT:       inst500000E2:    concrete_constant(inst500000E2)
// CHECK:STDOUT:       inst500000E3:    symbolic_constant5000004A
// CHECK:STDOUT:       inst500000E4:    symbolic_constant50000049
// CHECK:STDOUT:       inst500000E5:    symbolic_constant5000004A
// CHECK:STDOUT:       inst500000E6:    symbolic_constant5000004C
// CHECK:STDOUT:       inst500000E7:    symbolic_constant5000004B
// CHECK:STDOUT:       inst500000E8:    symbolic_constant5000004C
// CHECK:STDOUT:       inst500000E9:    symbolic_constant5000004C
// CHECK:STDOUT:       inst500000EA:    symbolic_constant5000004E
// CHECK:STDOUT:       inst500000EB:    symbolic_constant5000004D
// CHECK:STDOUT:       inst500000EC:    symbolic_constant5000004E
// CHECK:STDOUT:       inst500000ED:    symbolic_constant5000004E
// CHECK:STDOUT:       inst500000EE:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst500000F0:    symbolic_constant5000004F
// CHECK:STDOUT:       inst500000F1:    symbolic_constant50000050
// CHECK:STDOUT:       inst500000F4:    symbolic_constant50000051
// CHECK:STDOUT:       inst500000F5:    symbolic_constant50000052
// CHECK:STDOUT:       inst500000F7:    symbolic_constant50000053
// CHECK:STDOUT:       inst500000F8:    symbolic_constant50000056
// CHECK:STDOUT:       inst500000F9:    symbolic_constant50000054
// CHECK:STDOUT:       inst500000FA:    symbolic_constant50000055
// CHECK:STDOUT:       inst500000FB:    symbolic_constant50000056
// CHECK:STDOUT:       inst500000FC:    symbolic_constant50000057
// CHECK:STDOUT:       inst500000FD:    concrete_constant(inst5000006F)
// CHECK:STDOUT:       inst500000FE:    concrete_constant(inst500000FE)
// CHECK:STDOUT:       inst500000FF:    concrete_constant(inst50000100)
// CHECK:STDOUT:       inst50000100:    concrete_constant(inst50000100)
// CHECK:STDOUT:       inst50000101:    concrete_constant(inst50000102)
// CHECK:STDOUT:       inst50000102:    concrete_constant(inst50000102)
// CHECK:STDOUT:       inst50000104:    symbolic_constant5000004C
// CHECK:STDOUT:       inst50000105:    symbolic_constant5000004E
// CHECK:STDOUT:       inst50000106:    symbolic_constant5000004E
// CHECK:STDOUT:       inst50000107:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst50000109:    symbolic_constant50000056
// CHECK:STDOUT:       inst5000010A:    symbolic_constant50000059
// CHECK:STDOUT:       inst5000010B:    symbolic_constant50000058
// CHECK:STDOUT:       inst5000010C:    symbolic_constant50000059
// CHECK:STDOUT:       inst5000010D:    symbolic_constant5000005A
// CHECK:STDOUT:       inst5000010E:    symbolic_constant5000005D
// CHECK:STDOUT:       inst5000010F:    symbolic_constant5000005B
// CHECK:STDOUT:       inst50000110:    symbolic_constant5000005C
// CHECK:STDOUT:       inst50000111:    symbolic_constant5000005D
// CHECK:STDOUT:       inst50000112:    symbolic_constant5000005F
// CHECK:STDOUT:       inst50000113:    symbolic_constant5000005E
// CHECK:STDOUT:       inst50000114:    symbolic_constant5000005F
// CHECK:STDOUT:       inst50000116:    concrete_constant(inst500000E1)
// CHECK:STDOUT:       inst50000118:    concrete_constant(inst5000006F)
// CHECK:STDOUT:       inst5000011A:    symbolic_constant50000059
// CHECK:STDOUT:       inst5000011B:    concrete_constant(inst5000011D)
// CHECK:STDOUT:       inst5000011C:    concrete_constant(inst5000011C)
// CHECK:STDOUT:       inst5000011D:    concrete_constant(inst5000011D)
// CHECK:STDOUT:       inst5000011E:    symbolic_constant50000061
// CHECK:STDOUT:       inst5000011F:    symbolic_constant50000060
// CHECK:STDOUT:       inst50000120:    symbolic_constant50000061
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant50000000: {inst: inst50000015, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000001: {inst: inst50000015, kind: checked, attached: {generic: generic50000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant50000002: {inst: inst50000022, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000003: {inst: inst50000028, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000004: {inst: inst50000032, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000005: {inst: inst50000032, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant50000006: {inst: inst50000035, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000007: {inst: inst50000035, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant50000008: {inst: inst50000039, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000009: {inst: inst50000039, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant5000000A: {inst: inst50000041, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000B: {inst: inst50000043, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000C: {inst: inst50000041, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant5000000D: {inst: inst50000043, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant5000000E: {inst: inst50000048, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000F: {inst: inst50000048, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant50000010: {inst: inst5000004A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000011: {inst: inst5000004C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000012: {inst: inst5000004A, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant50000013: {inst: inst5000004C, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant50000014: {inst: inst50000050, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000015: {inst: inst50000050, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant50000016: {inst: inst5000005D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000017: {inst: inst5000005D, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant50000018: {inst: inst5000005F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000019: {inst: inst50000061, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000001A: {inst: inst5000005F, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant5000001B: {inst: inst50000061, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant5000001C: {inst: inst50000065, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000001D: {inst: inst50000065, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant5000001E: {inst: inst50000072, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000001F: {inst: inst50000072, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant50000020: {inst: inst50000075, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000021: {inst: inst50000075, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant50000022: {inst: inst5000007C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000023: {inst: inst5000007C, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant50000024: {inst: inst5000007F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000025: {inst: inst5000007F, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant50000026: {inst: inst50000083, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000027: {inst: inst50000083, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant50000028: {inst: inst50000088, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000029: {inst: inst5000008A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000002A: {inst: inst50000088, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant5000002B: {inst: inst5000008A, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant5000002C: {inst: inst5000008E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000002D: {inst: inst5000008E, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant5000002E: {inst: inst50000090, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000002F: {inst: inst50000092, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000030: {inst: inst50000090, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant50000031: {inst: inst50000092, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant50000032: {inst: inst50000096, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000033: {inst: inst50000096, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant50000034: {inst: inst500000A0, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000035: {inst: inst500000A0, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant50000036: {inst: inst500000A2, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000037: {inst: inst500000A4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000038: {inst: inst500000A2, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant50000039: {inst: inst500000A4, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant5000003A: {inst: inst500000A8, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000003B: {inst: inst500000A8, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant5000003C: {inst: inst500000B4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000003D: {inst: inst500000B4, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant5000003E: {inst: inst500000B7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000003F: {inst: inst500000B7, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant50000040: {inst: inst500000BD, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000041: {inst: inst500000C3, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000042: {inst: inst500000CA, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000043: {inst: inst500000CC, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000044: {inst: inst500000CE, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000045: {inst: inst500000D0, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000046: {inst: inst500000D8, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000047: {inst: inst500000DC, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000048: {inst: inst500000DE, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000049: {inst: inst500000E4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000004A: {inst: inst500000E4, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant5000004B: {inst: inst500000E7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000004C: {inst: inst500000E7, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant5000004D: {inst: inst500000EB, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000004E: {inst: inst500000EB, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant5000004F: {inst: inst500000F0, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000050: {inst: inst500000F1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000051: {inst: inst500000F4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000052: {inst: inst500000F5, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000053: {inst: inst500000F7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000054: {inst: inst500000F9, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000055: {inst: inst500000F7, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant50000056: {inst: inst500000F9, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant50000057: {inst: inst500000FC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000058: {inst: inst5000010B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000059: {inst: inst5000010B, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant5000005A: {inst: inst5000010D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000005B: {inst: inst5000010F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000005C: {inst: inst5000010D, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant5000005D: {inst: inst5000010F, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant5000005E: {inst: inst50000113, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000005F: {inst: inst50000113, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant50000060: {inst: inst5000011F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000061: {inst: inst5000011F, kind: checked, attached: {generic: generic50000005, index: generic_inst_in_def0}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst50000012
// CHECK:STDOUT:       1:               inst5000006D
// CHECK:STDOUT:       2:               inst500000B0
// CHECK:STDOUT:       3:               inst5000011B
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst50000011
// CHECK:STDOUT:       1:               inst50000024
// CHECK:STDOUT:       2:               inst50000025
// CHECK:STDOUT:       3:               inst50000027
// CHECK:STDOUT:       4:               inst50000029
// CHECK:STDOUT:       5:               inst5000002A
// CHECK:STDOUT:       6:               inst5000002B
// CHECK:STDOUT:       7:               inst5000002C
// CHECK:STDOUT:       8:               inst5000002D
// CHECK:STDOUT:       9:               inst5000002E
// CHECK:STDOUT:       10:              inst5000003D
// CHECK:STDOUT:       11:              inst50000046
// CHECK:STDOUT:       12:              inst50000056
// CHECK:STDOUT:       13:              inst5000005B
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     extra_global_variables: {}
// CHECK:STDOUT:     inst_block50000006:
// CHECK:STDOUT:       0:               inst50000014
// CHECK:STDOUT:     inst_block50000007:
// CHECK:STDOUT:       0:               inst50000015
// CHECK:STDOUT:     inst_block50000008:
// CHECK:STDOUT:       0:               inst50000014
// CHECK:STDOUT:       1:               inst50000017
// CHECK:STDOUT:     inst_block50000009:
// CHECK:STDOUT:       0:               inst50000019
// CHECK:STDOUT:       1:               inst5000001E
// CHECK:STDOUT:     inst_block5000000A:
// CHECK:STDOUT:       0:               inst50000018
// CHECK:STDOUT:     inst_block5000000B:
// CHECK:STDOUT:       0:               inst5000001B
// CHECK:STDOUT:     inst_block5000000C:
// CHECK:STDOUT:       0:               inst5000001D
// CHECK:STDOUT:     inst_block5000000D:
// CHECK:STDOUT:       0:               inst5000001F
// CHECK:STDOUT:     inst_block5000000E:
// CHECK:STDOUT:       0:               inst50000019
// CHECK:STDOUT:       1:               inst5000001E
// CHECK:STDOUT:     inst_block5000000F:
// CHECK:STDOUT:       0:               inst50000021
// CHECK:STDOUT:       1:               inst50000023
// CHECK:STDOUT:       2:               inst5000002F
// CHECK:STDOUT:     inst_block50000010:
// CHECK:STDOUT:       0:               inst5000002B
// CHECK:STDOUT:       1:               inst5000002C
// CHECK:STDOUT:     inst_block50000011:
// CHECK:STDOUT:       0:               inst5000002D
// CHECK:STDOUT:     inst_block50000012:
// CHECK:STDOUT:       0:               inst50000028
// CHECK:STDOUT:     inst_block50000013:
// CHECK:STDOUT:       0:               inst50000031
// CHECK:STDOUT:     inst_block50000014:
// CHECK:STDOUT:       0:               inst50000037
// CHECK:STDOUT:       1:               inst50000038
// CHECK:STDOUT:       2:               inst5000003B
// CHECK:STDOUT:       3:               inst5000003F
// CHECK:STDOUT:       4:               inst50000047
// CHECK:STDOUT:     inst_block50000015:
// CHECK:STDOUT:       0:               inst50000035
// CHECK:STDOUT:     inst_block50000016:
// CHECK:STDOUT:       0:               inst50000041
// CHECK:STDOUT:     inst_block50000017:
// CHECK:STDOUT:       0:               inst50000053
// CHECK:STDOUT:       1:               inst50000054
// CHECK:STDOUT:       2:               inst50000055
// CHECK:STDOUT:       3:               inst50000058
// CHECK:STDOUT:       4:               inst5000005C
// CHECK:STDOUT:     inst_block50000018:
// CHECK:STDOUT:       0:               inst5000004F
// CHECK:STDOUT:       1:               inst50000064
// CHECK:STDOUT:     inst_block50000019:
// CHECK:STDOUT:       0:               inst5000004B
// CHECK:STDOUT:       1:               inst50000060
// CHECK:STDOUT:     inst_block5000001A:
// CHECK:STDOUT:       0:               inst50000069
// CHECK:STDOUT:       1:               inst5000006B
// CHECK:STDOUT:     inst_block5000001B:
// CHECK:STDOUT:       0:               inst50000031
// CHECK:STDOUT:       1:               inst5000004B
// CHECK:STDOUT:       2:               inst5000004F
// CHECK:STDOUT:       3:               inst50000060
// CHECK:STDOUT:       4:               inst50000064
// CHECK:STDOUT:     inst_block5000001C:
// CHECK:STDOUT:       0:               inst50000068
// CHECK:STDOUT:       1:               inst50000034
// CHECK:STDOUT:       2:               inst50000069
// CHECK:STDOUT:       3:               inst5000006A
// CHECK:STDOUT:       4:               inst50000052
// CHECK:STDOUT:       5:               inst5000006B
// CHECK:STDOUT:       6:               inst5000006C
// CHECK:STDOUT:       7:               inst50000067
// CHECK:STDOUT:     inst_block5000001D:
// CHECK:STDOUT:       0:               inst50000034
// CHECK:STDOUT:     inst_block5000001E:
// CHECK:STDOUT:       0:               inst50000033
// CHECK:STDOUT:       1:               inst50000036
// CHECK:STDOUT:       2:               inst5000003A
// CHECK:STDOUT:       3:               inst50000044
// CHECK:STDOUT:       4:               inst50000045
// CHECK:STDOUT:       5:               inst50000049
// CHECK:STDOUT:       6:               inst5000004D
// CHECK:STDOUT:       7:               inst5000004E
// CHECK:STDOUT:       8:               inst50000051
// CHECK:STDOUT:       9:               inst5000005E
// CHECK:STDOUT:       10:              inst50000062
// CHECK:STDOUT:       11:              inst50000063
// CHECK:STDOUT:       12:              inst50000066
// CHECK:STDOUT:     inst_block5000001F:
// CHECK:STDOUT:       0:               inst50000032
// CHECK:STDOUT:       1:               inst50000035
// CHECK:STDOUT:       2:               inst50000039
// CHECK:STDOUT:       3:               inst50000041
// CHECK:STDOUT:       4:               inst50000043
// CHECK:STDOUT:       5:               inst50000048
// CHECK:STDOUT:       6:               inst5000004A
// CHECK:STDOUT:       7:               inst5000004C
// CHECK:STDOUT:       8:               inst50000050
// CHECK:STDOUT:       9:               inst5000005D
// CHECK:STDOUT:       10:              inst5000005F
// CHECK:STDOUT:       11:              inst50000061
// CHECK:STDOUT:       12:              inst50000065
// CHECK:STDOUT:     inst_block50000020:
// CHECK:STDOUT:       0:               inst50000077
// CHECK:STDOUT:     inst_block50000021:
// CHECK:STDOUT:       0:               inst50000073
// CHECK:STDOUT:       1:               inst50000076
// CHECK:STDOUT:     inst_block50000022:
// CHECK:STDOUT:       0:               inst50000078
// CHECK:STDOUT:       1:               inst50000079
// CHECK:STDOUT:     inst_block50000023:
// CHECK:STDOUT:       0:               inst5000007B
// CHECK:STDOUT:     inst_block50000024:
// CHECK:STDOUT:       0:               inst50000081
// CHECK:STDOUT:       1:               inst50000082
// CHECK:STDOUT:       2:               inst50000085
// CHECK:STDOUT:       3:               inst50000086
// CHECK:STDOUT:       4:               inst5000008D
// CHECK:STDOUT:     inst_block50000025:
// CHECK:STDOUT:       0:               inst5000007F
// CHECK:STDOUT:     inst_block50000026:
// CHECK:STDOUT:       0:               inst50000088
// CHECK:STDOUT:     inst_block50000027:
// CHECK:STDOUT:       0:               inst50000099
// CHECK:STDOUT:       1:               inst5000009A
// CHECK:STDOUT:       2:               inst5000009B
// CHECK:STDOUT:       3:               inst5000009C
// CHECK:STDOUT:       4:               inst5000009F
// CHECK:STDOUT:     inst_block50000028:
// CHECK:STDOUT:       0:               inst50000095
// CHECK:STDOUT:       1:               inst500000A7
// CHECK:STDOUT:     inst_block50000029:
// CHECK:STDOUT:       0:               inst50000091
// CHECK:STDOUT:       1:               inst500000A3
// CHECK:STDOUT:     inst_block5000002A:
// CHECK:STDOUT:       0:               inst500000AC
// CHECK:STDOUT:       1:               inst500000AE
// CHECK:STDOUT:     inst_block5000002B:
// CHECK:STDOUT:       0:               inst5000007B
// CHECK:STDOUT:       1:               inst50000091
// CHECK:STDOUT:       2:               inst50000095
// CHECK:STDOUT:       3:               inst500000A3
// CHECK:STDOUT:       4:               inst500000A7
// CHECK:STDOUT:     inst_block5000002C:
// CHECK:STDOUT:       0:               inst500000AB
// CHECK:STDOUT:       1:               inst5000007E
// CHECK:STDOUT:       2:               inst500000AC
// CHECK:STDOUT:       3:               inst500000AD
// CHECK:STDOUT:       4:               inst50000098
// CHECK:STDOUT:       5:               inst500000AE
// CHECK:STDOUT:       6:               inst500000AF
// CHECK:STDOUT:       7:               inst500000AA
// CHECK:STDOUT:     inst_block5000002D:
// CHECK:STDOUT:       0:               inst5000007E
// CHECK:STDOUT:     inst_block5000002E:
// CHECK:STDOUT:       0:               inst5000007D
// CHECK:STDOUT:       1:               inst50000080
// CHECK:STDOUT:       2:               inst50000084
// CHECK:STDOUT:       3:               inst5000008B
// CHECK:STDOUT:       4:               inst5000008C
// CHECK:STDOUT:       5:               inst5000008F
// CHECK:STDOUT:       6:               inst50000093
// CHECK:STDOUT:       7:               inst50000094
// CHECK:STDOUT:       8:               inst50000097
// CHECK:STDOUT:       9:               inst500000A1
// CHECK:STDOUT:       10:              inst500000A5
// CHECK:STDOUT:       11:              inst500000A6
// CHECK:STDOUT:       12:              inst500000A9
// CHECK:STDOUT:     inst_block5000002F:
// CHECK:STDOUT:       0:               inst5000007C
// CHECK:STDOUT:       1:               inst5000007F
// CHECK:STDOUT:       2:               inst50000083
// CHECK:STDOUT:       3:               inst50000088
// CHECK:STDOUT:       4:               inst5000008A
// CHECK:STDOUT:       5:               inst5000008E
// CHECK:STDOUT:       6:               inst50000090
// CHECK:STDOUT:       7:               inst50000092
// CHECK:STDOUT:       8:               inst50000096
// CHECK:STDOUT:       9:               inst500000A0
// CHECK:STDOUT:       10:              inst500000A2
// CHECK:STDOUT:       11:              inst500000A4
// CHECK:STDOUT:       12:              inst500000A8
// CHECK:STDOUT:     inst_block50000030:
// CHECK:STDOUT:       0:               inst500000B9
// CHECK:STDOUT:     inst_block50000031:
// CHECK:STDOUT:       0:               inst500000B5
// CHECK:STDOUT:       1:               inst500000B8
// CHECK:STDOUT:     inst_block50000032:
// CHECK:STDOUT:       0:               inst500000BA
// CHECK:STDOUT:       1:               inst500000BB
// CHECK:STDOUT:       2:               inst500000BC
// CHECK:STDOUT:       3:               inst500000BE
// CHECK:STDOUT:       4:               inst500000BF
// CHECK:STDOUT:       5:               inst500000C0
// CHECK:STDOUT:       6:               inst500000C1
// CHECK:STDOUT:       7:               inst500000C2
// CHECK:STDOUT:       8:               inst500000C4
// CHECK:STDOUT:       9:               inst500000C5
// CHECK:STDOUT:       10:              inst500000C6
// CHECK:STDOUT:       11:              inst500000C7
// CHECK:STDOUT:       12:              inst500000C8
// CHECK:STDOUT:       13:              inst500000C9
// CHECK:STDOUT:       14:              inst500000CF
// CHECK:STDOUT:       15:              inst500000D1
// CHECK:STDOUT:       16:              inst500000D3
// CHECK:STDOUT:       17:              inst500000D4
// CHECK:STDOUT:       18:              inst500000D5
// CHECK:STDOUT:       19:              inst500000D6
// CHECK:STDOUT:       20:              inst500000D7
// CHECK:STDOUT:       21:              inst500000D9
// CHECK:STDOUT:       22:              inst500000DA
// CHECK:STDOUT:       23:              inst500000DD
// CHECK:STDOUT:       24:              inst500000DF
// CHECK:STDOUT:       25:              inst500000E0
// CHECK:STDOUT:     inst_block50000033:
// CHECK:STDOUT:       0:               inst500000BD
// CHECK:STDOUT:     inst_block50000034:
// CHECK:STDOUT:       0:               inst500000CC
// CHECK:STDOUT:     inst_block50000035:
// CHECK:STDOUT:       0:               inst500000BE
// CHECK:STDOUT:       1:               inst500000C5
// CHECK:STDOUT:       2:               inst500000D4
// CHECK:STDOUT:     inst_block50000036:
// CHECK:STDOUT:       0:               inst500000BE
// CHECK:STDOUT:       1:               inst500000DA
// CHECK:STDOUT:       2:               inst500000DF
// CHECK:STDOUT:     inst_block50000037:
// CHECK:STDOUT:       0:               inst500000E3
// CHECK:STDOUT:     inst_block50000038:
// CHECK:STDOUT:       0:               inst500000E9
// CHECK:STDOUT:       1:               inst500000EA
// CHECK:STDOUT:       2:               inst500000ED
// CHECK:STDOUT:       3:               inst500000EE
// CHECK:STDOUT:       4:               inst500000FD
// CHECK:STDOUT:     inst_block50000039:
// CHECK:STDOUT:       0:               inst500000E7
// CHECK:STDOUT:     inst_block5000003A:
// CHECK:STDOUT:       0:               inst500000F0
// CHECK:STDOUT:     inst_block5000003B:
// CHECK:STDOUT:       0:               inst500000F7
// CHECK:STDOUT:     inst_block5000003C:
// CHECK:STDOUT:       0:               inst500000FC
// CHECK:STDOUT:     inst_block5000003D:
// CHECK:STDOUT:       0:               inst50000104
// CHECK:STDOUT:       1:               inst50000105
// CHECK:STDOUT:       2:               inst50000106
// CHECK:STDOUT:       3:               inst50000107
// CHECK:STDOUT:       4:               inst5000010A
// CHECK:STDOUT:     inst_block5000003E:
// CHECK:STDOUT:       0:               inst50000101
// CHECK:STDOUT:       1:               inst50000112
// CHECK:STDOUT:     inst_block5000003F:
// CHECK:STDOUT:       0:               inst500000FF
// CHECK:STDOUT:       1:               inst5000010E
// CHECK:STDOUT:     inst_block50000040:
// CHECK:STDOUT:       0:               inst50000117
// CHECK:STDOUT:       1:               inst50000119
// CHECK:STDOUT:     inst_block50000041:
// CHECK:STDOUT:       0:               inst500000E3
// CHECK:STDOUT:       1:               inst500000FF
// CHECK:STDOUT:       2:               inst50000101
// CHECK:STDOUT:       3:               inst5000010E
// CHECK:STDOUT:       4:               inst50000112
// CHECK:STDOUT:     inst_block50000042:
// CHECK:STDOUT:       0:               inst50000116
// CHECK:STDOUT:       1:               inst500000E6
// CHECK:STDOUT:       2:               inst50000117
// CHECK:STDOUT:       3:               inst50000118
// CHECK:STDOUT:       4:               inst50000103
// CHECK:STDOUT:       5:               inst50000119
// CHECK:STDOUT:       6:               inst5000011A
// CHECK:STDOUT:       7:               inst50000115
// CHECK:STDOUT:     inst_block50000043:
// CHECK:STDOUT:       0:               inst500000E6
// CHECK:STDOUT:     inst_block50000044:
// CHECK:STDOUT:       0:               inst500000E5
// CHECK:STDOUT:       1:               inst500000E8
// CHECK:STDOUT:       2:               inst500000EC
// CHECK:STDOUT:       3:               inst500000FA
// CHECK:STDOUT:       4:               inst500000FB
// CHECK:STDOUT:       5:               inst5000010C
// CHECK:STDOUT:       6:               inst50000110
// CHECK:STDOUT:       7:               inst50000111
// CHECK:STDOUT:       8:               inst50000114
// CHECK:STDOUT:     inst_block50000045:
// CHECK:STDOUT:       0:               inst500000E4
// CHECK:STDOUT:       1:               inst500000E7
// CHECK:STDOUT:       2:               inst500000EB
// CHECK:STDOUT:       3:               inst500000F7
// CHECK:STDOUT:       4:               inst500000F9
// CHECK:STDOUT:       5:               inst5000010B
// CHECK:STDOUT:       6:               inst5000010D
// CHECK:STDOUT:       7:               inst5000010F
// CHECK:STDOUT:       8:               inst50000113
// CHECK:STDOUT:     inst_block50000046:
// CHECK:STDOUT:       0:               inst50000121
// CHECK:STDOUT:     inst_block50000047:
// CHECK:STDOUT:       0:               inst50000120
// CHECK:STDOUT:     inst_block50000048:
// CHECK:STDOUT:       0:               inst(Package)
// CHECK:STDOUT:       1:               inst50000010
// CHECK:STDOUT:       2:               inst50000012
// CHECK:STDOUT:       3:               inst5000006D
// CHECK:STDOUT:       4:               inst500000B0
// CHECK:STDOUT:       5:               inst5000011B
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
// CHECK:STDOUT:         identifier8:     HasRewrites
// CHECK:STDOUT:       strings:         {}
// CHECK:STDOUT: ...

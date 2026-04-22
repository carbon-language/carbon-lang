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
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name1: inst60000010}}
// CHECK:STDOUT:     name_scope60000001: {inst: inst60000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name(SelfType): inst60000012}}
// CHECK:STDOUT:     name_scope60000002: {inst: inst60000015, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name(SelfType), parent_scope: name_scope60000001, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface60000000: {name: name1, parent_scope: name_scope0, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:   associated_constants: {}
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
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst60000010:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000012:    symbolic_constant60000000
// CHECK:STDOUT:       inst60000013:    symbolic_constant60000000
// CHECK:STDOUT:       inst60000014:    symbolic_constant60000001
// CHECK:STDOUT:       inst60000015:    concrete_constant(inst60000015)
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
// CHECK:STDOUT:     inst_block60000008: {}
// CHECK:STDOUT:     inst_block60000009:
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
// CHECK:STDOUT:       strings:         {}
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        main.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           NonCore
// CHECK:STDOUT:     name1:           Copy
// CHECK:STDOUT:     name2:           UseNonCoreCopy
// CHECK:STDOUT:     name3:           T
// CHECK:STDOUT:     name4:           Destroy
// CHECK:STDOUT:     name5:           x
// CHECK:STDOUT:     name6:           UseLocalCopy
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir50000002: {decl_id: inst50000010, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir50000002, inst_id: inst60000010}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir50000002, inst_id: inst60000010}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir50000002, inst_id: inst60000015}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir50000002, inst_id: inst60000012}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir50000002, inst_id: inst60000012}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst50000011, name1: inst50000012, name2: inst50000037, name6: inst50000052}}
// CHECK:STDOUT:     name_scope50000001: {inst: inst50000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name4: inst5000001D}}
// CHECK:STDOUT:     name_scope50000002: {inst: inst50000012, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name(SelfType): inst50000014}}
// CHECK:STDOUT:     name_scope50000003: {inst: inst50000017, parent_scope: name_scope50000002, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope50000004: {inst: inst5000001E, parent_scope: name_scope50000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst50000023}}
// CHECK:STDOUT:     name_scope50000005: {inst: inst50000020, parent_scope: name_scope50000004, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name50000000: {name: name(SelfType), parent_scope: name_scope50000002, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000001: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000002: {name: name4, parent_scope: name_scope50000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000003: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000004: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000005: {name: name3, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000006: {name: name5, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>}
// CHECK:STDOUT:     entity_name50000007: {name: name3, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000008: {name: name5, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 1, form: inst<none>}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function50000000: {name: name2, parent_scope: name_scope0, call_param_patterns_id: inst_block5000000F, call_params_id: inst_block50000010, body: [inst_block50000017]}
// CHECK:STDOUT:     function50000001: {name: name6, parent_scope: name_scope0, call_param_patterns_id: inst_block5000001D, call_params_id: inst_block5000001E, body: [inst_block50000025]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface50000000: {name: name1, parent_scope: name_scope0, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:     interface50000001: {name: name4, parent_scope: name_scope50000001, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic50000000: {decl: inst<none>, bindings: inst_block<none>, self_specific_id: specific<none>, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic50000001: {decl: inst50000017, bindings: inst_block50000005, self_specific_id: specific50000000, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic50000002: {decl: inst50000020, bindings: inst_block5000000A, self_specific_id: specific50000001, decl_block_id: inst_block_empty, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic50000003: {decl: inst50000037, bindings: inst_block50000013, self_specific_id: specific50000002, decl_block_id: inst_block50000015, definition_block_id: inst_block50000018}
// CHECK:STDOUT:     generic50000004: {decl: inst50000052, bindings: inst_block50000021, self_specific_id: specific50000003, decl_block_id: inst_block50000023, definition_block_id: inst_block50000026}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific50000000: {generic: generic50000001, args: inst_block50000006, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000001: {generic: generic50000002, args: inst_block5000000B, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block_empty, definition_has_error: 0}
// CHECK:STDOUT:     specific50000002: {generic: generic50000003, args: inst_block50000014, decl_block_id: inst_block50000016, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific50000003: {generic: generic50000004, args: inst_block50000022, decl_block_id: inst_block50000024, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
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
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001C)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000001C)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001C)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000001F)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst5000001F)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000038)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001C)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant50000006)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant50000006)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       0
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000013)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000013)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000053)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001C)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant5000000E)':
// CHECK:STDOUT:       value_repr:      {kind: dependent, type: type(symbolic_constant5000000E)}
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
// CHECK:STDOUT:     inst50000018:    {kind: FacetType, arg0: facet_type50000001, type: type(TypeType)}
// CHECK:STDOUT:     inst50000019:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000018)}
// CHECK:STDOUT:     inst5000001A:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000018)}
// CHECK:STDOUT:     inst5000001B:    {kind: NameRef, arg0: name0, arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst5000001C:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001D:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name50000002, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001E:    {kind: InterfaceDecl, arg0: interface50000001, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001F:    {kind: FacetType, arg0: facet_type50000002, type: type(TypeType)}
// CHECK:STDOUT:     inst50000020:    {kind: InterfaceWithSelfDecl, arg0: interface50000001}
// CHECK:STDOUT:     inst50000021:    {kind: SymbolicBinding, arg0: entity_name50000003, arg1: inst<none>, type: type(inst5000001F)}
// CHECK:STDOUT:     inst50000022:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst5000001F)}
// CHECK:STDOUT:     inst50000023:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name<none>}
// CHECK:STDOUT:     inst50000024:    {kind: NameRef, arg0: name4, arg1: inst5000001D, type: type(TypeType)}
// CHECK:STDOUT:     inst50000025:    {kind: PatternType, arg0: inst5000001F, type: type(TypeType)}
// CHECK:STDOUT:     inst50000026:    {kind: SymbolicBinding, arg0: entity_name50000005, arg1: inst<none>, type: type(inst5000001F)}
// CHECK:STDOUT:     inst50000027:    {kind: SymbolicBinding, arg0: entity_name50000005, arg1: inst<none>, type: type(inst5000001F)}
// CHECK:STDOUT:     inst50000028:    {kind: SymbolicBinding, arg0: entity_name50000005, arg1: inst<none>, type: type(inst5000001F)}
// CHECK:STDOUT:     inst50000029:    {kind: SymbolicBindingPattern, arg0: entity_name50000005, type: type(inst50000025)}
// CHECK:STDOUT:     inst5000002A:    {kind: NameRef, arg0: name3, arg1: inst50000026, type: type(inst5000001F)}
// CHECK:STDOUT:     inst5000002B:    {kind: FacetAccessType, arg0: inst5000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002C:    {kind: SymbolicBindingType, arg0: entity_name50000005, arg1: inst50000027, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002D:    {kind: SymbolicBindingType, arg0: entity_name50000005, arg1: inst50000028, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002E:    {kind: Converted, arg0: inst5000002A, arg1: inst5000002B, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002F:    {kind: PatternType, arg0: inst5000002C, type: type(TypeType)}
// CHECK:STDOUT:     inst50000030:    {kind: ValueParamPattern, arg0: name5, type: type(symbolic_constant50000009)}
// CHECK:STDOUT:     inst50000031:    {kind: PatternType, arg0: inst5000002D, type: type(TypeType)}
// CHECK:STDOUT:     inst50000032:    {kind: ValueBinding, arg0: entity_name50000006, arg1: inst50000035, type: type(symbolic_constant50000007)}
// CHECK:STDOUT:     inst50000033:    {kind: WrapperBindingPattern, arg0: entity_name50000006, arg1: inst50000030, type: type(symbolic_constant50000009)}
// CHECK:STDOUT:     inst50000034:    {kind: SpliceBlock, arg0: inst_block50000009, arg1: inst50000024, type: type(TypeType)}
// CHECK:STDOUT:     inst50000035:    {kind: ValueParam, arg0: call_param0, arg1: name5, type: type(symbolic_constant50000007)}
// CHECK:STDOUT:     inst50000036:    {kind: SpliceBlock, arg0: inst_block5000000D, arg1: inst5000002E, type: type(TypeType)}
// CHECK:STDOUT:     inst50000037:    {kind: FunctionDecl, arg0: function50000000, arg1: inst_block50000012, type: type(inst50000038)}
// CHECK:STDOUT:     inst50000038:    {kind: FunctionType, arg0: function50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000039:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000038)}
// CHECK:STDOUT:     inst5000003A:    {kind: RequireCompleteType, arg0: inst5000002C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000003B:    {kind: RequireCompleteType, arg0: inst5000002C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000003C:    {kind: RequireCompleteType, arg0: inst5000002D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst5000003D:    {kind: Return}
// CHECK:STDOUT:     inst5000003E:    {kind: SymbolicBinding, arg0: entity_name50000001, arg1: inst<none>, type: type(inst50000018)}
// CHECK:STDOUT:     inst5000003F:    {kind: NameRef, arg0: name1, arg1: inst50000012, type: type(TypeType)}
// CHECK:STDOUT:     inst50000040:    {kind: PatternType, arg0: inst50000013, type: type(TypeType)}
// CHECK:STDOUT:     inst50000041:    {kind: SymbolicBinding, arg0: entity_name50000007, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000042:    {kind: SymbolicBinding, arg0: entity_name50000005, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000043:    {kind: SymbolicBinding, arg0: entity_name50000005, arg1: inst<none>, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000044:    {kind: SymbolicBindingPattern, arg0: entity_name50000007, type: type(inst50000040)}
// CHECK:STDOUT:     inst50000045:    {kind: NameRef, arg0: name3, arg1: inst50000041, type: type(inst50000013)}
// CHECK:STDOUT:     inst50000046:    {kind: FacetAccessType, arg0: inst50000045, type: type(TypeType)}
// CHECK:STDOUT:     inst50000047:    {kind: SymbolicBindingType, arg0: entity_name50000005, arg1: inst50000042, type: type(TypeType)}
// CHECK:STDOUT:     inst50000048:    {kind: SymbolicBindingType, arg0: entity_name50000005, arg1: inst50000043, type: type(TypeType)}
// CHECK:STDOUT:     inst50000049:    {kind: Converted, arg0: inst50000045, arg1: inst50000046, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004A:    {kind: PatternType, arg0: inst50000047, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004B:    {kind: ValueParamPattern, arg0: name5, type: type(symbolic_constant50000011)}
// CHECK:STDOUT:     inst5000004C:    {kind: PatternType, arg0: inst50000048, type: type(TypeType)}
// CHECK:STDOUT:     inst5000004D:    {kind: ValueBinding, arg0: entity_name50000008, arg1: inst50000050, type: type(symbolic_constant5000000F)}
// CHECK:STDOUT:     inst5000004E:    {kind: WrapperBindingPattern, arg0: entity_name50000008, arg1: inst5000004B, type: type(symbolic_constant50000011)}
// CHECK:STDOUT:     inst5000004F:    {kind: SpliceBlock, arg0: inst_block50000019, arg1: inst5000003F, type: type(TypeType)}
// CHECK:STDOUT:     inst50000050:    {kind: ValueParam, arg0: call_param0, arg1: name5, type: type(symbolic_constant5000000F)}
// CHECK:STDOUT:     inst50000051:    {kind: SpliceBlock, arg0: inst_block5000001B, arg1: inst50000049, type: type(TypeType)}
// CHECK:STDOUT:     inst50000052:    {kind: FunctionDecl, arg0: function50000001, arg1: inst_block50000020, type: type(inst50000053)}
// CHECK:STDOUT:     inst50000053:    {kind: FunctionType, arg0: function50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000054:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000053)}
// CHECK:STDOUT:     inst50000055:    {kind: RequireCompleteType, arg0: inst50000047, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000056:    {kind: RequireCompleteType, arg0: inst50000047, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000057:    {kind: RequireCompleteType, arg0: inst50000048, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000058:    {kind: Return}
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
// CHECK:STDOUT:       inst50000018:    concrete_constant(inst50000018)
// CHECK:STDOUT:       inst50000019:    symbolic_constant50000002
// CHECK:STDOUT:       inst5000001A:    symbolic_constant50000002
// CHECK:STDOUT:       inst5000001B:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst5000001C:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst5000001D:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst5000001E:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst5000001F:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst50000020:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst50000021:    symbolic_constant50000003
// CHECK:STDOUT:       inst50000022:    symbolic_constant50000003
// CHECK:STDOUT:       inst50000023:    constant<none>
// CHECK:STDOUT:       inst50000024:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst50000025:    concrete_constant(inst50000025)
// CHECK:STDOUT:       inst50000026:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000027:    symbolic_constant50000004
// CHECK:STDOUT:       inst50000028:    symbolic_constant50000005
// CHECK:STDOUT:       inst50000029:    concrete_constant(inst50000029)
// CHECK:STDOUT:       inst5000002A:    symbolic_constant50000005
// CHECK:STDOUT:       inst5000002B:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000002C:    symbolic_constant50000006
// CHECK:STDOUT:       inst5000002D:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000002E:    symbolic_constant50000007
// CHECK:STDOUT:       inst5000002F:    symbolic_constant50000008
// CHECK:STDOUT:       inst50000030:    concrete_constant(inst50000030)
// CHECK:STDOUT:       inst50000031:    symbolic_constant50000009
// CHECK:STDOUT:       inst50000033:    concrete_constant(inst50000033)
// CHECK:STDOUT:       inst50000034:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst50000036:    symbolic_constant50000007
// CHECK:STDOUT:       inst50000037:    concrete_constant(inst50000039)
// CHECK:STDOUT:       inst50000038:    concrete_constant(inst50000038)
// CHECK:STDOUT:       inst50000039:    concrete_constant(inst50000039)
// CHECK:STDOUT:       inst5000003A:    symbolic_constant5000000B
// CHECK:STDOUT:       inst5000003B:    symbolic_constant5000000A
// CHECK:STDOUT:       inst5000003C:    symbolic_constant5000000B
// CHECK:STDOUT:       inst5000003E:    symbolic_constant50000002
// CHECK:STDOUT:       inst5000003F:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst50000040:    concrete_constant(inst50000040)
// CHECK:STDOUT:       inst50000041:    symbolic_constant5000000D
// CHECK:STDOUT:       inst50000042:    symbolic_constant5000000C
// CHECK:STDOUT:       inst50000043:    symbolic_constant5000000D
// CHECK:STDOUT:       inst50000044:    concrete_constant(inst50000044)
// CHECK:STDOUT:       inst50000045:    symbolic_constant5000000D
// CHECK:STDOUT:       inst50000046:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000047:    symbolic_constant5000000E
// CHECK:STDOUT:       inst50000048:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000049:    symbolic_constant5000000F
// CHECK:STDOUT:       inst5000004A:    symbolic_constant50000010
// CHECK:STDOUT:       inst5000004B:    concrete_constant(inst5000004B)
// CHECK:STDOUT:       inst5000004C:    symbolic_constant50000011
// CHECK:STDOUT:       inst5000004E:    concrete_constant(inst5000004E)
// CHECK:STDOUT:       inst5000004F:    concrete_constant(inst50000013)
// CHECK:STDOUT:       inst50000051:    symbolic_constant5000000F
// CHECK:STDOUT:       inst50000052:    concrete_constant(inst50000054)
// CHECK:STDOUT:       inst50000053:    concrete_constant(inst50000053)
// CHECK:STDOUT:       inst50000054:    concrete_constant(inst50000054)
// CHECK:STDOUT:       inst50000055:    symbolic_constant50000013
// CHECK:STDOUT:       inst50000056:    symbolic_constant50000012
// CHECK:STDOUT:       inst50000057:    symbolic_constant50000013
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant50000000: {inst: inst50000015, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000001: {inst: inst50000015, kind: checked, attached: {generic: generic50000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant50000002: {inst: inst5000001A, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant50000003: {inst: inst50000021, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000004: {inst: inst50000027, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000005: {inst: inst50000027, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant50000006: {inst: inst5000002C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000007: {inst: inst5000002C, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant50000008: {inst: inst5000002F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000009: {inst: inst5000002F, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant5000000A: {inst: inst5000003B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000B: {inst: inst5000003B, kind: checked, attached: {generic: generic50000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant5000000C: {inst: inst50000042, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000D: {inst: inst50000042, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant5000000E: {inst: inst50000047, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant5000000F: {inst: inst50000047, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant50000010: {inst: inst5000004A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000011: {inst: inst5000004A, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant50000012: {inst: inst50000056, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant50000013: {inst: inst50000056, kind: checked, attached: {generic: generic50000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst50000012
// CHECK:STDOUT:       1:               inst50000037
// CHECK:STDOUT:       2:               inst50000052
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst50000011
// CHECK:STDOUT:       1:               inst5000001D
// CHECK:STDOUT:       2:               inst5000001E
// CHECK:STDOUT:       3:               inst50000020
// CHECK:STDOUT:       4:               inst50000022
// CHECK:STDOUT:       5:               inst50000023
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block50000005:
// CHECK:STDOUT:       0:               inst50000014
// CHECK:STDOUT:     inst_block50000006:
// CHECK:STDOUT:       0:               inst50000015
// CHECK:STDOUT:     inst_block50000007:
// CHECK:STDOUT:       0:               inst50000014
// CHECK:STDOUT:       1:               inst50000017
// CHECK:STDOUT:     inst_block50000008: {}
// CHECK:STDOUT:     inst_block50000009:
// CHECK:STDOUT:       0:               inst50000019
// CHECK:STDOUT:       1:               inst5000001B
// CHECK:STDOUT:       2:               inst50000024
// CHECK:STDOUT:     inst_block5000000A:
// CHECK:STDOUT:       0:               inst50000022
// CHECK:STDOUT:     inst_block5000000B:
// CHECK:STDOUT:       0:               inst50000021
// CHECK:STDOUT:     inst_block5000000C:
// CHECK:STDOUT:       0:               inst50000029
// CHECK:STDOUT:     inst_block5000000D:
// CHECK:STDOUT:       0:               inst5000002A
// CHECK:STDOUT:       1:               inst5000002B
// CHECK:STDOUT:       2:               inst5000002E
// CHECK:STDOUT:     inst_block5000000E:
// CHECK:STDOUT:       0:               inst50000033
// CHECK:STDOUT:     inst_block5000000F:
// CHECK:STDOUT:       0:               inst50000030
// CHECK:STDOUT:     inst_block50000010:
// CHECK:STDOUT:       0:               inst50000035
// CHECK:STDOUT:     inst_block50000011:
// CHECK:STDOUT:       0:               inst50000029
// CHECK:STDOUT:       1:               inst50000030
// CHECK:STDOUT:       2:               inst50000033
// CHECK:STDOUT:     inst_block50000012:
// CHECK:STDOUT:       0:               inst50000034
// CHECK:STDOUT:       1:               inst50000026
// CHECK:STDOUT:       2:               inst50000035
// CHECK:STDOUT:       3:               inst50000036
// CHECK:STDOUT:       4:               inst50000032
// CHECK:STDOUT:     inst_block50000013:
// CHECK:STDOUT:       0:               inst50000026
// CHECK:STDOUT:     inst_block50000014:
// CHECK:STDOUT:       0:               inst50000027
// CHECK:STDOUT:     inst_block50000015:
// CHECK:STDOUT:       0:               inst50000028
// CHECK:STDOUT:       1:               inst5000002D
// CHECK:STDOUT:       2:               inst50000031
// CHECK:STDOUT:     inst_block50000016:
// CHECK:STDOUT:       0:               inst50000027
// CHECK:STDOUT:       1:               inst5000002C
// CHECK:STDOUT:       2:               inst5000002F
// CHECK:STDOUT:     inst_block50000017:
// CHECK:STDOUT:       0:               inst5000003D
// CHECK:STDOUT:     inst_block50000018:
// CHECK:STDOUT:       0:               inst5000003C
// CHECK:STDOUT:     inst_block50000019:
// CHECK:STDOUT:       0:               inst5000003E
// CHECK:STDOUT:       1:               inst5000003F
// CHECK:STDOUT:     inst_block5000001A:
// CHECK:STDOUT:       0:               inst50000044
// CHECK:STDOUT:     inst_block5000001B:
// CHECK:STDOUT:       0:               inst50000045
// CHECK:STDOUT:       1:               inst50000046
// CHECK:STDOUT:       2:               inst50000049
// CHECK:STDOUT:     inst_block5000001C:
// CHECK:STDOUT:       0:               inst5000004E
// CHECK:STDOUT:     inst_block5000001D:
// CHECK:STDOUT:       0:               inst5000004B
// CHECK:STDOUT:     inst_block5000001E:
// CHECK:STDOUT:       0:               inst50000050
// CHECK:STDOUT:     inst_block5000001F:
// CHECK:STDOUT:       0:               inst50000044
// CHECK:STDOUT:       1:               inst5000004B
// CHECK:STDOUT:       2:               inst5000004E
// CHECK:STDOUT:     inst_block50000020:
// CHECK:STDOUT:       0:               inst5000004F
// CHECK:STDOUT:       1:               inst50000041
// CHECK:STDOUT:       2:               inst50000050
// CHECK:STDOUT:       3:               inst50000051
// CHECK:STDOUT:       4:               inst5000004D
// CHECK:STDOUT:     inst_block50000021:
// CHECK:STDOUT:       0:               inst50000041
// CHECK:STDOUT:     inst_block50000022:
// CHECK:STDOUT:       0:               inst50000042
// CHECK:STDOUT:     inst_block50000023:
// CHECK:STDOUT:       0:               inst50000043
// CHECK:STDOUT:       1:               inst50000048
// CHECK:STDOUT:       2:               inst5000004C
// CHECK:STDOUT:     inst_block50000024:
// CHECK:STDOUT:       0:               inst50000042
// CHECK:STDOUT:       1:               inst50000047
// CHECK:STDOUT:       2:               inst5000004A
// CHECK:STDOUT:     inst_block50000025:
// CHECK:STDOUT:       0:               inst50000058
// CHECK:STDOUT:     inst_block50000026:
// CHECK:STDOUT:       0:               inst50000057
// CHECK:STDOUT:     inst_block50000027:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst50000010
// CHECK:STDOUT:       2:               inst50000012
// CHECK:STDOUT:       3:               inst50000037
// CHECK:STDOUT:       4:               inst50000052
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     NonCore
// CHECK:STDOUT:         identifier1:     Copy
// CHECK:STDOUT:         identifier2:     UseNonCoreCopy
// CHECK:STDOUT:         identifier3:     T
// CHECK:STDOUT:         identifier4:     Destroy
// CHECK:STDOUT:         identifier5:     x
// CHECK:STDOUT:         identifier6:     UseLocalCopy
// CHECK:STDOUT:       strings:         {}
// CHECK:STDOUT: ...

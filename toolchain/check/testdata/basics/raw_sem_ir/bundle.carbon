// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/form.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --no-dump-sem-ir
//
// Check that raw bundle dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/bundle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/bundle.carbon

fn F(Form:! Core.Form) ->? Form;

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        bundle.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           F
// CHECK:STDOUT:     name1:           Form
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir70000002: {decl_id: inst70000010, is_export: false}
// CHECK:STDOUT:     import_ir70000003: {decl_id: inst70000010, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir70000003, inst_id: inst50000022}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   clang_decl_signatures: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst70000011, name0: inst7000002F}}
// CHECK:STDOUT:     name_scope70000001: {inst: inst70000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name1: inst70000016}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name70000000: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name70000001: {name: name1, parent_scope: name_scope70000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name70000002: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function70000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block70000007, call_params_id: inst_block70000008, return_type_inst_id: inst70000020, return_form_inst_id: inst7000001F, return_pattern_id: inst70000028}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:      {}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic70000000: {decl: inst7000002F, bindings: inst_block7000000B, self_specific_id: specific70000000, decl_block_id: inst_block7000000D, definition_block_id: inst_block<none>}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific70000000: {generic: generic70000000, args: inst_block7000000C, decl_block_id: inst_block7000000E, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
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
// CHECK:STDOUT:     'type(inst70000034)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst70000035)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst70000035)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst70000035)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:   facet_types:
// CHECK:STDOUT:     facet_type70000000: {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst70000010:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst70000011:    {kind: Namespace, arg0: name_scope70000001, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst70000012:    {kind: FacetType, arg0: facet_type70000000, type: type(TypeType)}
// CHECK:STDOUT:     inst70000013:    {kind: SymbolicBinding, arg0: entity_name70000000, arg1: inst<none>, type: type(inst70000012)}
// CHECK:STDOUT:     inst70000014:    {kind: SymbolicBinding, arg0: entity_name70000000, arg1: inst<none>, type: type(inst70000012)}
// CHECK:STDOUT:     inst70000015:    {kind: NameRef, arg0: name(Core), arg1: inst70000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst70000016:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name70000001, type: type(TypeType)}
// CHECK:STDOUT:     inst70000017:    {kind: NameRef, arg0: name1, arg1: inst70000016, type: type(TypeType)}
// CHECK:STDOUT:     inst70000018:    {kind: PatternType, arg0: inst(FormType), type: type(TypeType)}
// CHECK:STDOUT:     inst70000019:    {kind: SymbolicBindingPattern, arg0: entity_name70000002, type: type(inst70000018)}
// CHECK:STDOUT:     inst7000001A:    {kind: SymbolicBindingPattern, arg0: entity_name70000002, type: type(inst70000018)}
// CHECK:STDOUT:     inst7000001B:    {kind: SymbolicBindingPattern, arg0: entity_name70000002, type: type(inst70000018)}
// CHECK:STDOUT:     inst7000001C:    {kind: SymbolicBinding, arg0: entity_name70000002, arg1: inst<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7000001D:    {kind: SymbolicBinding, arg0: entity_name70000002, arg1: inst<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7000001E:    {kind: SymbolicBinding, arg0: entity_name70000002, arg1: inst<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7000001F:    {kind: NameRef, arg0: name1, arg1: inst7000001C, type: type(inst(FormType))}
// CHECK:STDOUT:     inst70000020:    {kind: TypeComponentOf, arg0: inst7000001D, type: type(TypeType)}
// CHECK:STDOUT:     inst70000021:    {kind: PatternType, arg0: inst70000020, type: type(TypeType)}
// CHECK:STDOUT:     inst70000022:    {kind: OutFormParamPatternAction, arg0: inst7000001F, type: type(inst(InstType))}
// CHECK:STDOUT:     inst70000023:    {kind: SpliceInst, arg0: inst70000022, type: type(symbolic_constant7000000A)}
// CHECK:STDOUT:     inst70000024:    {kind: SpliceInst, arg0: inst70000022, type: type(symbolic_constant70000006)}
// CHECK:STDOUT:     inst70000025:    {kind: TypeComponentOf, arg0: inst7000001E, type: type(TypeType)}
// CHECK:STDOUT:     inst70000026:    {kind: PatternType, arg0: inst70000025, type: type(TypeType)}
// CHECK:STDOUT:     inst70000027:    {kind: SpliceInst, arg0: inst70000022, type: type(symbolic_constant7000000A)}
// CHECK:STDOUT:     inst70000028:    {kind: ReturnSlotPattern, arg0: inst70000023, arg1: inst70000020, type: type(symbolic_constant7000000A)}
// CHECK:STDOUT:     inst70000029:    {kind: ReturnSlotPattern, arg0: inst70000024, arg1: inst70000020, type: type(symbolic_constant70000006)}
// CHECK:STDOUT:     inst7000002A:    {kind: ReturnSlotPattern, arg0: inst70000027, arg1: inst70000025, type: type(symbolic_constant7000000A)}
// CHECK:STDOUT:     inst7000002B:    {kind: SpliceBlock, arg0: inst_block70000005, arg1: inst70000017, type: type(TypeType)}
// CHECK:STDOUT:     inst7000002C:    {kind: CalleePatternMatchAction, arg0: bundle70000000, type: type(inst(InstType))}
// CHECK:STDOUT:     inst7000002D:    {kind: SpliceInst, arg0: inst7000002C, type: type(symbolic_constant7000000A)}
// CHECK:STDOUT:     inst7000002E:    {kind: ReturnSlot, arg0: inst70000020, arg1: inst7000002D, type: type(symbolic_constant70000009)}
// CHECK:STDOUT:     inst7000002F:    {kind: FunctionDecl, arg0: function70000000, arg1: inst_block7000000A, type: type(inst70000034)}
// CHECK:STDOUT:     inst70000030:    {kind: OutFormParamPatternAction, arg0: inst7000001D, type: type(inst(InstType))}
// CHECK:STDOUT:     inst70000031:    {kind: SpliceInst, arg0: inst70000030, type: type(symbolic_constant70000006)}
// CHECK:STDOUT:     inst70000032:    {kind: ReturnSlotPattern, arg0: inst70000031, arg1: inst70000020, type: type(symbolic_constant70000006)}
// CHECK:STDOUT:     inst70000033:    {kind: CalleePatternMatchAction, arg0: bundle70000001, type: type(inst(InstType))}
// CHECK:STDOUT:     inst70000034:    {kind: FunctionType, arg0: function70000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst70000035:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst70000036:    {kind: StructValue, arg0: inst_block_empty, type: type(inst70000034)}
// CHECK:STDOUT:   bundles:
// CHECK:STDOUT:     bundle70000000:  {arg0: inst70000023, arg1: call_param0}
// CHECK:STDOUT:     bundle70000001:  {arg0: inst70000031, arg1: call_param0}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst70000011:    concrete_constant(inst70000011)
// CHECK:STDOUT:       inst70000012:    concrete_constant(inst70000012)
// CHECK:STDOUT:       inst70000013:    symbolic_constant70000000
// CHECK:STDOUT:       inst70000014:    symbolic_constant70000000
// CHECK:STDOUT:       inst70000015:    concrete_constant(inst70000011)
// CHECK:STDOUT:       inst70000016:    concrete_constant(inst(FormType))
// CHECK:STDOUT:       inst70000017:    concrete_constant(inst(FormType))
// CHECK:STDOUT:       inst70000018:    concrete_constant(inst70000018)
// CHECK:STDOUT:       inst70000019:    symbolic_constant70000002
// CHECK:STDOUT:       inst7000001A:    symbolic_constant70000001
// CHECK:STDOUT:       inst7000001B:    symbolic_constant70000002
// CHECK:STDOUT:       inst7000001C:    symbolic_constant70000004
// CHECK:STDOUT:       inst7000001D:    symbolic_constant70000003
// CHECK:STDOUT:       inst7000001E:    symbolic_constant70000004
// CHECK:STDOUT:       inst7000001F:    symbolic_constant70000004
// CHECK:STDOUT:       inst70000020:    symbolic_constant70000005
// CHECK:STDOUT:       inst70000021:    symbolic_constant70000006
// CHECK:STDOUT:       inst70000022:    symbolic_constant70000007
// CHECK:STDOUT:       inst70000023:    symbolic_constant7000000B
// CHECK:STDOUT:       inst70000024:    symbolic_constant70000008
// CHECK:STDOUT:       inst70000025:    symbolic_constant70000009
// CHECK:STDOUT:       inst70000026:    symbolic_constant7000000A
// CHECK:STDOUT:       inst70000027:    symbolic_constant7000000B
// CHECK:STDOUT:       inst70000028:    symbolic_constant7000000D
// CHECK:STDOUT:       inst70000029:    symbolic_constant7000000C
// CHECK:STDOUT:       inst7000002A:    symbolic_constant7000000D
// CHECK:STDOUT:       inst7000002B:    concrete_constant(inst(FormType))
// CHECK:STDOUT:       inst7000002C:    symbolic_constant7000000E
// CHECK:STDOUT:       inst7000002F:    concrete_constant(inst70000036)
// CHECK:STDOUT:       inst70000030:    symbolic_constant7000000F
// CHECK:STDOUT:       inst70000031:    symbolic_constant70000010
// CHECK:STDOUT:       inst70000032:    symbolic_constant70000011
// CHECK:STDOUT:       inst70000033:    symbolic_constant70000012
// CHECK:STDOUT:       inst70000034:    concrete_constant(inst70000034)
// CHECK:STDOUT:       inst70000035:    concrete_constant(inst70000035)
// CHECK:STDOUT:       inst70000036:    concrete_constant(inst70000036)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant70000000: {inst: inst70000014, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant70000001: {inst: inst7000001A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant70000002: {inst: inst7000001A, kind: checked, attached: {generic: generic70000000, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant70000003: {inst: inst7000001D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant70000004: {inst: inst7000001D, kind: checked, attached: {generic: generic70000000, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant70000005: {inst: inst70000020, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant70000006: {inst: inst70000021, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant70000007: {inst: inst70000022, kind: template, attached: {generic: generic70000000, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant70000008: {inst: inst70000024, kind: template, attached: null}
// CHECK:STDOUT:       symbolic_constant70000009: {inst: inst70000020, kind: checked, attached: {generic: generic70000000, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7000000A: {inst: inst70000021, kind: checked, attached: {generic: generic70000000, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7000000B: {inst: inst70000024, kind: template, attached: {generic: generic70000000, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7000000C: {inst: inst70000029, kind: template, attached: null}
// CHECK:STDOUT:       symbolic_constant7000000D: {inst: inst70000029, kind: template, attached: {generic: generic70000000, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7000000E: {inst: inst7000002C, kind: template, attached: {generic: generic70000000, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7000000F: {inst: inst70000030, kind: template, attached: null}
// CHECK:STDOUT:       symbolic_constant70000010: {inst: inst70000031, kind: template, attached: null}
// CHECK:STDOUT:       symbolic_constant70000011: {inst: inst70000032, kind: template, attached: null}
// CHECK:STDOUT:       symbolic_constant70000012: {inst: inst70000033, kind: template, attached: null}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst7000002F
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst70000011
// CHECK:STDOUT:       1:               inst70000016
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block70000005:
// CHECK:STDOUT:       0:               inst70000013
// CHECK:STDOUT:       1:               inst70000015
// CHECK:STDOUT:       2:               inst70000017
// CHECK:STDOUT:     inst_block70000006:
// CHECK:STDOUT:       0:               inst70000019
// CHECK:STDOUT:     inst_block70000007:
// CHECK:STDOUT:       0:               inst70000023
// CHECK:STDOUT:     inst_block70000008:
// CHECK:STDOUT:       0:               inst7000002D
// CHECK:STDOUT:     inst_block70000009:
// CHECK:STDOUT:       0:               inst70000019
// CHECK:STDOUT:       1:               inst70000028
// CHECK:STDOUT:     inst_block7000000A:
// CHECK:STDOUT:       0:               inst7000001F
// CHECK:STDOUT:       1:               inst70000023
// CHECK:STDOUT:       2:               inst7000002B
// CHECK:STDOUT:       3:               inst7000001C
// CHECK:STDOUT:       4:               inst7000002D
// CHECK:STDOUT:       5:               inst7000002E
// CHECK:STDOUT:     inst_block7000000B:
// CHECK:STDOUT:       0:               inst7000001C
// CHECK:STDOUT:     inst_block7000000C:
// CHECK:STDOUT:       0:               inst7000001D
// CHECK:STDOUT:     inst_block7000000D:
// CHECK:STDOUT:       0:               inst7000001B
// CHECK:STDOUT:       1:               inst7000001E
// CHECK:STDOUT:       2:               inst70000022
// CHECK:STDOUT:       3:               inst70000025
// CHECK:STDOUT:       4:               inst70000026
// CHECK:STDOUT:       5:               inst70000027
// CHECK:STDOUT:       6:               inst7000002A
// CHECK:STDOUT:       7:               inst7000002C
// CHECK:STDOUT:     inst_block7000000E:
// CHECK:STDOUT:       0:               inst7000001A
// CHECK:STDOUT:       1:               inst7000001D
// CHECK:STDOUT:       2:               inst70000030
// CHECK:STDOUT:       3:               inst70000020
// CHECK:STDOUT:       4:               inst70000021
// CHECK:STDOUT:       5:               inst70000031
// CHECK:STDOUT:       6:               inst70000032
// CHECK:STDOUT:       7:               inst70000033
// CHECK:STDOUT:     inst_block7000000F:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst70000010
// CHECK:STDOUT:       2:               inst7000002F
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     F
// CHECK:STDOUT:         identifier1:     Form
// CHECK:STDOUT:       strings:
// CHECK:STDOUT:         string0:         prelude
// CHECK:STDOUT: ...

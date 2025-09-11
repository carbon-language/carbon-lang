// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --no-dump-sem-ir
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file.carbon

fn Foo[T:! type](p: T*) -> (T*, ()) {
  return (p, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        one_file.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir2:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:     ir3:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:     ir4:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:     ir5:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir4, inst_id: inst15}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: ir4, inst_id: inst15}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: ir4, inst_id: inst17}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: ir4, inst_id: inst45}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: ir4, inst_id: inst40}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: ir4, inst_id: inst40}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: ir4, inst_id: inst40}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: ir4, inst_id: inst34}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: ir4, inst_id: inst35}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: ir4, inst_id: inst28}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: ir4, inst_id: inst30}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: ir4, inst_id: inst17}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: ir4, inst_id: inst21}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: ir4, inst_id: inst24}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: ir4, inst_id: inst29}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: ir4, inst_id: inst83}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: ir4, inst_id: inst81}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: ir4, inst_id: inst79}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: ir4, inst_id: inst80}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: ir4, inst_id: inst107}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: ir4, inst_id: inst105}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: ir4, inst_id: inst103}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: ir4, inst_id: inst104}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: ir4, inst_id: inst131}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: ir4, inst_id: inst129}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: ir4, inst_id: inst127}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: ir4, inst_id: inst128}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: ir4, inst_id: inst155}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: ir4, inst_id: inst153}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: ir4, inst_id: inst151}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: ir4, inst_id: inst152}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: ir4, inst_id: inst186}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: ir4, inst_id: inst184}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: ir4, inst_id: inst178}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: ir4, inst_id: inst175}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: ir4, inst_id: inst180}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: ir4, inst_id: inst183}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: ir4, inst_id: inst203}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: ir4, inst_id: inst185}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: ir4, inst_id: inst177}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: ir4, inst_id: inst182}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: ir4, inst_id: inst188}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: ir4, inst_id: inst203}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: ir4, inst_id: inst198}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: ir4, inst_id: inst199}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: ir4, inst_id: inst194}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: ir4, inst_id: inst196}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: ir4, inst_id: inst175}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: ir4, inst_id: inst206}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: ir4, inst_id: inst207}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: ir4, inst_id: inst210}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: ir4, inst_id: inst190}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: ir4, inst_id: inst191}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: ir4, inst_id: inst195}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: ir4, inst_id: inst216}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: ir4, inst_id: inst214}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: ir4, inst_id: inst(TypeType)}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: ir4, inst_id: inst213}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: ir4, inst_id: inst237}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: ir4, inst_id: inst235}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: ir4, inst_id: inst233}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: ir4, inst_id: inst234}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: ir4, inst_id: inst288}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: ir4, inst_id: inst286}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: ir4, inst_id: inst266}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: ir4, inst_id: inst260}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: ir4, inst_id: inst256}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: ir4, inst_id: inst263}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: ir4, inst_id: inst281}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: ir4, inst_id: inst283}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: ir4, inst_id: inst310}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: ir4, inst_id: inst287}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: ir4, inst_id: inst258}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: ir4, inst_id: inst265}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: ir4, inst_id: inst274}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: ir4, inst_id: inst278}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: ir4, inst_id: inst282}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: ir4, inst_id: inst290}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: ir4, inst_id: inst310}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: ir4, inst_id: inst305}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: ir4, inst_id: inst306}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: ir4, inst_id: inst301}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: ir4, inst_id: inst303}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: ir4, inst_id: inst256}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: ir4, inst_id: inst263}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: ir4, inst_id: inst313}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: ir4, inst_id: inst314}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: ir4, inst_id: inst294}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: ir4, inst_id: inst295}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: ir4, inst_id: inst296}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: ir4, inst_id: inst297}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: ir4, inst_id: inst298}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: ir4, inst_id: inst302}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: ir4, inst_id: inst319}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: ir4, inst_id: inst328}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: ir4, inst_id: inst335}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: ir4, inst_id: inst340}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: ir4, inst_id: inst341}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: ir4, inst_id: inst342}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: ir4, inst_id: inst347}
// CHECK:STDOUT:     import_ir_inst100: {ir_id: ir4, inst_id: inst358}
// CHECK:STDOUT:     import_ir_inst101: {ir_id: ir4, inst_id: inst365}
// CHECK:STDOUT:     import_ir_inst102: {ir_id: ir4, inst_id: inst370}
// CHECK:STDOUT:     import_ir_inst103: {ir_id: ir4, inst_id: inst371}
// CHECK:STDOUT:     import_ir_inst104: {ir_id: ir4, inst_id: inst372}
// CHECK:STDOUT:     import_ir_inst105: {ir_id: ir4, inst_id: inst377}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst16, name0: inst53}}
// CHECK:STDOUT:     name_scope1:     {inst: inst16, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst69}}
// CHECK:STDOUT:     name_scope2:     {inst: inst70, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {name(SelfType): inst73, name4: inst74}}
// CHECK:STDOUT:     name_scope3:     {inst: inst94, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope4:     {inst: inst98, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope5:     {inst: inst102, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6:     {inst: inst106, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7:     {inst: inst110, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope8:     {inst: inst136, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope9:     {inst: inst140, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope10:    {inst: inst144, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name3:    {name: name3, parent_scope: name_scope1, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name4:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name5:    {name: name4, parent_scope: name_scope2, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name6:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name7:    {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name8:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name9:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name10:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name11:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name12:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name13:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name14:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name15:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name16:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name17:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name18:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name19:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name20:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name21:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name22:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name23:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name24:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name25:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: inst48, body: [inst_block20]}
// CHECK:STDOUT:     function1:       {name: name4, parent_scope: name_scope2, return_slot_pattern: inst85}
// CHECK:STDOUT:     function2:       {name: name4, parent_scope: name_scope7, return_slot_pattern: inst125}
// CHECK:STDOUT:     function3:       {name: name4, parent_scope: name_scope10, return_slot_pattern: inst171}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst53, bindings: inst_block16}
// CHECK:STDOUT:     generic1:        {decl: inst79, bindings: inst_block24}
// CHECK:STDOUT:     generic2:        {decl: inst110, bindings: inst_block29}
// CHECK:STDOUT:     generic3:        {decl: inst121, bindings: inst_block35}
// CHECK:STDOUT:     generic4:        {decl: inst144, bindings: inst_block42}
// CHECK:STDOUT:     generic5:        {decl: inst166, bindings: inst_block50}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block17}
// CHECK:STDOUT:     specific1:       {generic: generic1, args: inst_block26}
// CHECK:STDOUT:     specific2:       {generic: generic2, args: inst_block17}
// CHECK:STDOUT:     specific3:       {generic: generic2, args: inst_block31}
// CHECK:STDOUT:     specific4:       {generic: generic3, args: inst_block17}
// CHECK:STDOUT:     specific5:       {generic: generic4, args: inst_block44}
// CHECK:STDOUT:     specific6:       {generic: generic4, args: inst_block46}
// CHECK:STDOUT:     specific7:       {generic: generic5, args: inst_block44}
// CHECK:STDOUT:     specific8:       {generic: generic1, args: inst_block57}
// CHECK:STDOUT:     specific9:       {generic: generic1, args: inst_block59}
// CHECK:STDOUT:     specific10:      {generic: generic1, args: inst_block61}
// CHECK:STDOUT:     specific11:      {generic: generic1, args: inst_block63}
// CHECK:STDOUT:     specific12:      {generic: generic1, args: inst_block71}
// CHECK:STDOUT:     specific13:      {generic: generic1, args: inst_block73}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst36)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(inst38)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst40)}
// CHECK:STDOUT:     'type(inst40)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst40)}
// CHECK:STDOUT:     'type(inst54)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(symbolic_constant3)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'type(symbolic_constant7)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant11)}
// CHECK:STDOUT:     'type(symbolic_constant11)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant11)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant4)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant4)}
// CHECK:STDOUT:     'type(symbolic_constant8)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant11)}
// CHECK:STDOUT:     'type(inst71)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst71)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(symbolic_constant133)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(symbolic_constant136)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst15:          {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst16:          {kind: Namespace, arg0: name_scope1, arg1: inst15, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst17:          {kind: FacetType, arg0: facet_type0, type: type(TypeType)}
// CHECK:STDOUT:     inst18:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(inst17)}
// CHECK:STDOUT:     inst19:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(inst17)}
// CHECK:STDOUT:     inst20:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst21:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst22:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst23:          {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst24:          {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(inst23)}
// CHECK:STDOUT:     inst25:          {kind: NameRef, arg0: name1, arg1: inst20, type: type(TypeType)}
// CHECK:STDOUT:     inst26:          {kind: PointerType, arg0: inst25, type: type(TypeType)}
// CHECK:STDOUT:     inst27:          {kind: PointerType, arg0: inst21, type: type(TypeType)}
// CHECK:STDOUT:     inst28:          {kind: PointerType, arg0: inst22, type: type(TypeType)}
// CHECK:STDOUT:     inst29:          {kind: BindName, arg0: entity_name2, arg1: inst49, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst30:          {kind: PatternType, arg0: inst27, type: type(TypeType)}
// CHECK:STDOUT:     inst31:          {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst32:          {kind: PatternType, arg0: inst28, type: type(TypeType)}
// CHECK:STDOUT:     inst33:          {kind: ValueParamPattern, arg0: inst31, arg1: call_param0, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst34:          {kind: NameRef, arg0: name1, arg1: inst20, type: type(TypeType)}
// CHECK:STDOUT:     inst35:          {kind: PointerType, arg0: inst34, type: type(TypeType)}
// CHECK:STDOUT:     inst36:          {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst37:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst36)}
// CHECK:STDOUT:     inst38:          {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     inst39:          {kind: TupleLiteral, arg0: inst_block8, type: type(inst38)}
// CHECK:STDOUT:     inst40:          {kind: PointerType, arg0: inst38, type: type(TypeType)}
// CHECK:STDOUT:     inst41:          {kind: Converted, arg0: inst37, arg1: inst36, type: type(TypeType)}
// CHECK:STDOUT:     inst42:          {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     inst43:          {kind: Converted, arg0: inst39, arg1: inst42, type: type(TypeType)}
// CHECK:STDOUT:     inst44:          {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     inst45:          {kind: PatternType, arg0: inst42, type: type(TypeType)}
// CHECK:STDOUT:     inst46:          {kind: ReturnSlotPattern, arg0: inst43, type: type(symbolic_constant10)}
// CHECK:STDOUT:     inst47:          {kind: PatternType, arg0: inst44, type: type(TypeType)}
// CHECK:STDOUT:     inst48:          {kind: OutParamPattern, arg0: inst46, arg1: call_param1, type: type(symbolic_constant10)}
// CHECK:STDOUT:     inst49:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst50:          {kind: SpliceBlock, arg0: inst_block6, arg1: inst26, type: type(TypeType)}
// CHECK:STDOUT:     inst51:          {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst52:          {kind: ReturnSlot, arg0: inst42, arg1: inst51, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst53:          {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(inst54)}
// CHECK:STDOUT:     inst54:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst55:          {kind: StructValue, arg0: inst_block_empty, type: type(inst54)}
// CHECK:STDOUT:     inst56:          {kind: PointerType, arg0: inst42, type: type(TypeType)}
// CHECK:STDOUT:     inst57:          {kind: RequireCompleteType, arg0: inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst58:          {kind: RequireCompleteType, arg0: inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst59:          {kind: RequireCompleteType, arg0: inst44, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60:          {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst61:          {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst62:          {kind: RequireCompleteType, arg0: inst28, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst63:          {kind: NameRef, arg0: name2, arg1: inst29, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst64:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst36)}
// CHECK:STDOUT:     inst65:          {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst66:          {kind: RequireCompleteType, arg0: inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst67:          {kind: TupleAccess, arg0: inst52, arg1: element0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst68:          {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst69:          {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name3, type: type(TypeType)}
// CHECK:STDOUT:     inst70:          {kind: InterfaceDecl, arg0: interface0, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst71:          {kind: FacetType, arg0: facet_type1, type: type(TypeType)}
// CHECK:STDOUT:     inst72:          {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst73:          {kind: ImportRefUnloaded, arg0: import_ir_inst2, arg1: entity_name<none>}
// CHECK:STDOUT:     inst74:          {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst76)}
// CHECK:STDOUT:     inst75:          {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name5}
// CHECK:STDOUT:     inst76:          {kind: AssociatedEntityType, arg0: interface0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst77:          {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(inst80)}
// CHECK:STDOUT:     inst78:          {kind: AssociatedEntity, arg0: element0, arg1: inst77, type: type(inst76)}
// CHECK:STDOUT:     inst79:          {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(inst80)}
// CHECK:STDOUT:     inst80:          {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst81:          {kind: StructValue, arg0: inst_block_empty, type: type(inst80)}
// CHECK:STDOUT:     inst82:          {kind: FacetAccessType, arg0: inst72, type: type(TypeType)}
// CHECK:STDOUT:     inst83:          {kind: PatternType, arg0: inst82, type: type(TypeType)}
// CHECK:STDOUT:     inst84:          {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst85:          {kind: OutParamPattern, arg0: inst84, arg1: call_param1, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst86:          {kind: BindingPattern, arg0: entity_name7, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst87:          {kind: ValueParamPattern, arg0: inst86, arg1: call_param0, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst88:          {kind: ImportRefLoaded, arg0: import_ir_inst11, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst89:          {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst90:          {kind: FacetAccessType, arg0: inst89, type: type(TypeType)}
// CHECK:STDOUT:     inst91:          {kind: PatternType, arg0: inst90, type: type(TypeType)}
// CHECK:STDOUT:     inst92:          {kind: LookupImplWitness, arg0: inst27, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst93:          {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     inst94:          {kind: ImplDecl, arg0: impl0, arg1: inst_block_empty}
// CHECK:STDOUT:     inst95:          {kind: ImportRefLoaded, arg0: import_ir_inst17, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst96:          {kind: ImportRefLoaded, arg0: import_ir_inst18, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst97:          {kind: ImportRefUnloaded, arg0: import_ir_inst19, arg1: entity_name<none>}
// CHECK:STDOUT:     inst98:          {kind: ImplDecl, arg0: impl1, arg1: inst_block_empty}
// CHECK:STDOUT:     inst99:          {kind: ImportRefLoaded, arg0: import_ir_inst21, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst100:         {kind: ImportRefLoaded, arg0: import_ir_inst22, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst101:         {kind: ImportRefUnloaded, arg0: import_ir_inst23, arg1: entity_name<none>}
// CHECK:STDOUT:     inst102:         {kind: ImplDecl, arg0: impl2, arg1: inst_block_empty}
// CHECK:STDOUT:     inst103:         {kind: ImportRefLoaded, arg0: import_ir_inst25, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst104:         {kind: ImportRefLoaded, arg0: import_ir_inst26, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst105:         {kind: ImportRefUnloaded, arg0: import_ir_inst27, arg1: entity_name<none>}
// CHECK:STDOUT:     inst106:         {kind: ImplDecl, arg0: impl3, arg1: inst_block_empty}
// CHECK:STDOUT:     inst107:         {kind: ImportRefLoaded, arg0: import_ir_inst29, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst108:         {kind: ImportRefLoaded, arg0: import_ir_inst30, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst109:         {kind: ImportRefLoaded, arg0: import_ir_inst31, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst110:         {kind: ImplDecl, arg0: impl4, arg1: inst_block_empty}
// CHECK:STDOUT:     inst111:         {kind: SymbolicBindingPattern, arg0: entity_name10, type: type(inst23)}
// CHECK:STDOUT:     inst112:         {kind: ImportRefLoaded, arg0: import_ir_inst34, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst113:         {kind: ImportRefLoaded, arg0: import_ir_inst35, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst114:         {kind: ImportRefLoaded, arg0: import_ir_inst36, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst115:         {kind: ImportRefUnloaded, arg0: import_ir_inst37, arg1: entity_name<none>}
// CHECK:STDOUT:     inst116:         {kind: ImplWitnessTable, arg0: inst_block30, arg1: impl4}
// CHECK:STDOUT:     inst117:         {kind: ImplWitness, arg0: inst116, arg1: specific2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst118:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst119:         {kind: PointerType, arg0: inst118, type: type(TypeType)}
// CHECK:STDOUT:     inst120:         {kind: ImplWitness, arg0: inst116, arg1: specific3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst121:         {kind: FunctionDecl, arg0: function2, arg1: inst_block_empty, type: type(symbolic_constant33)}
// CHECK:STDOUT:     inst122:         {kind: FunctionType, arg0: function2, arg1: specific2, type: type(TypeType)}
// CHECK:STDOUT:     inst123:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant33)}
// CHECK:STDOUT:     inst124:         {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant37)}
// CHECK:STDOUT:     inst125:         {kind: OutParamPattern, arg0: inst124, arg1: call_param1, type: type(symbolic_constant37)}
// CHECK:STDOUT:     inst126:         {kind: BindingPattern, arg0: entity_name13, type: type(symbolic_constant37)}
// CHECK:STDOUT:     inst127:         {kind: ValueParamPattern, arg0: inst126, arg1: call_param0, type: type(symbolic_constant37)}
// CHECK:STDOUT:     inst128:         {kind: ImportRefLoaded, arg0: import_ir_inst47, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst129:         {kind: FunctionType, arg0: function2, arg1: specific3, type: type(TypeType)}
// CHECK:STDOUT:     inst130:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant40)}
// CHECK:STDOUT:     inst131:         {kind: RequireCompleteType, arg0: inst119, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst132:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst133:         {kind: PointerType, arg0: inst132, type: type(TypeType)}
// CHECK:STDOUT:     inst134:         {kind: PatternType, arg0: inst133, type: type(TypeType)}
// CHECK:STDOUT:     inst135:         {kind: ImportRefUnloaded, arg0: import_ir_inst54, arg1: entity_name<none>}
// CHECK:STDOUT:     inst136:         {kind: ImplDecl, arg0: impl5, arg1: inst_block_empty}
// CHECK:STDOUT:     inst137:         {kind: ImportRefLoaded, arg0: import_ir_inst56, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst138:         {kind: ImportRefLoaded, arg0: import_ir_inst57, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst139:         {kind: ImportRefUnloaded, arg0: import_ir_inst58, arg1: entity_name<none>}
// CHECK:STDOUT:     inst140:         {kind: ImplDecl, arg0: impl6, arg1: inst_block_empty}
// CHECK:STDOUT:     inst141:         {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst142:         {kind: ImportRefLoaded, arg0: import_ir_inst61, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst143:         {kind: ImportRefUnloaded, arg0: import_ir_inst62, arg1: entity_name<none>}
// CHECK:STDOUT:     inst144:         {kind: ImplDecl, arg0: impl7, arg1: inst_block_empty}
// CHECK:STDOUT:     inst145:         {kind: BindSymbolicName, arg0: entity_name15, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst146:         {kind: FacetAccessType, arg0: inst145, type: type(TypeType)}
// CHECK:STDOUT:     inst147:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst148:         {kind: FacetAccessType, arg0: inst147, type: type(TypeType)}
// CHECK:STDOUT:     inst149:         {kind: TupleType, arg0: inst_block40, type: type(TypeType)}
// CHECK:STDOUT:     inst150:         {kind: PatternType, arg0: inst71, type: type(TypeType)}
// CHECK:STDOUT:     inst151:         {kind: SymbolicBindingPattern, arg0: entity_name17, type: type(inst150)}
// CHECK:STDOUT:     inst152:         {kind: SymbolicBindingPattern, arg0: entity_name18, type: type(inst150)}
// CHECK:STDOUT:     inst153:         {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst154:         {kind: ImportRefLoaded, arg0: import_ir_inst67, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst155:         {kind: ImportRefLoaded, arg0: import_ir_inst68, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst156:         {kind: ImportRefLoaded, arg0: import_ir_inst69, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst157:         {kind: ImportRefUnloaded, arg0: import_ir_inst70, arg1: entity_name<none>}
// CHECK:STDOUT:     inst158:         {kind: ImplWitnessTable, arg0: inst_block43, arg1: impl7}
// CHECK:STDOUT:     inst159:         {kind: ImplWitness, arg0: inst158, arg1: specific5, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst160:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst161:         {kind: BindSymbolicName, arg0: entity_name15, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst162:         {kind: FacetAccessType, arg0: inst160, type: type(TypeType)}
// CHECK:STDOUT:     inst163:         {kind: FacetAccessType, arg0: inst161, type: type(TypeType)}
// CHECK:STDOUT:     inst164:         {kind: TupleType, arg0: inst_block45, type: type(TypeType)}
// CHECK:STDOUT:     inst165:         {kind: ImplWitness, arg0: inst158, arg1: specific6, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst166:         {kind: FunctionDecl, arg0: function3, arg1: inst_block_empty, type: type(symbolic_constant68)}
// CHECK:STDOUT:     inst167:         {kind: FunctionType, arg0: function3, arg1: specific5, type: type(TypeType)}
// CHECK:STDOUT:     inst168:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant68)}
// CHECK:STDOUT:     inst169:         {kind: PatternType, arg0: inst149, type: type(TypeType)}
// CHECK:STDOUT:     inst170:         {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant73)}
// CHECK:STDOUT:     inst171:         {kind: OutParamPattern, arg0: inst170, arg1: call_param1, type: type(symbolic_constant73)}
// CHECK:STDOUT:     inst172:         {kind: BindingPattern, arg0: entity_name23, type: type(symbolic_constant73)}
// CHECK:STDOUT:     inst173:         {kind: ValueParamPattern, arg0: inst172, arg1: call_param0, type: type(symbolic_constant73)}
// CHECK:STDOUT:     inst174:         {kind: ImportRefLoaded, arg0: import_ir_inst83, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst175:         {kind: ImportRefLoaded, arg0: import_ir_inst84, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst176:         {kind: FunctionType, arg0: function3, arg1: specific6, type: type(TypeType)}
// CHECK:STDOUT:     inst177:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant75)}
// CHECK:STDOUT:     inst178:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst179:         {kind: FacetAccessType, arg0: inst178, type: type(TypeType)}
// CHECK:STDOUT:     inst180:         {kind: BindSymbolicName, arg0: entity_name15, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst181:         {kind: FacetAccessType, arg0: inst180, type: type(TypeType)}
// CHECK:STDOUT:     inst182:         {kind: TupleType, arg0: inst_block53, type: type(TypeType)}
// CHECK:STDOUT:     inst183:         {kind: PatternType, arg0: inst182, type: type(TypeType)}
// CHECK:STDOUT:     inst184:         {kind: RequireCompleteType, arg0: inst149, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst185:         {kind: RequireCompleteType, arg0: inst148, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst186:         {kind: LookupImplWitness, arg0: inst147, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst187:         {kind: FacetValue, arg0: inst148, arg1: inst_block56, type: type(inst71)}
// CHECK:STDOUT:     inst188:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst187, type: type(TypeType)}
// CHECK:STDOUT:     inst189:         {kind: ImplWitnessAccess, arg0: inst186, arg1: element0, type: type(symbolic_constant96)}
// CHECK:STDOUT:     inst190:         {kind: SpecificImplFunction, arg0: inst189, arg1: specific8, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst191:         {kind: RequireCompleteType, arg0: inst146, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst192:         {kind: LookupImplWitness, arg0: inst145, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst193:         {kind: FacetValue, arg0: inst146, arg1: inst_block58, type: type(inst71)}
// CHECK:STDOUT:     inst194:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst193, type: type(TypeType)}
// CHECK:STDOUT:     inst195:         {kind: ImplWitnessAccess, arg0: inst192, arg1: element0, type: type(symbolic_constant108)}
// CHECK:STDOUT:     inst196:         {kind: SpecificImplFunction, arg0: inst195, arg1: specific9, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst197:         {kind: RequireCompleteType, arg0: inst182, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst198:         {kind: RequireCompleteType, arg0: inst179, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst199:         {kind: LookupImplWitness, arg0: inst178, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst200:         {kind: FacetValue, arg0: inst179, arg1: inst_block60, type: type(inst71)}
// CHECK:STDOUT:     inst201:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst200, type: type(TypeType)}
// CHECK:STDOUT:     inst202:         {kind: ImplWitnessAccess, arg0: inst199, arg1: element0, type: type(symbolic_constant118)}
// CHECK:STDOUT:     inst203:         {kind: SpecificImplFunction, arg0: inst202, arg1: specific10, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst204:         {kind: RequireCompleteType, arg0: inst181, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst205:         {kind: LookupImplWitness, arg0: inst180, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst206:         {kind: FacetValue, arg0: inst181, arg1: inst_block62, type: type(inst71)}
// CHECK:STDOUT:     inst207:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst206, type: type(TypeType)}
// CHECK:STDOUT:     inst208:         {kind: ImplWitnessAccess, arg0: inst205, arg1: element0, type: type(symbolic_constant124)}
// CHECK:STDOUT:     inst209:         {kind: SpecificImplFunction, arg0: inst208, arg1: specific11, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst210:         {kind: PatternType, arg0: inst146, type: type(TypeType)}
// CHECK:STDOUT:     inst211:         {kind: PatternType, arg0: inst148, type: type(TypeType)}
// CHECK:STDOUT:     inst212:         {kind: LookupImplWitness, arg0: inst27, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst213:         {kind: LookupImplWitness, arg0: inst28, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst214:         {kind: FacetValue, arg0: inst27, arg1: inst_block69, type: type(inst71)}
// CHECK:STDOUT:     inst215:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst214, type: type(TypeType)}
// CHECK:STDOUT:     inst216:         {kind: ImplWitnessAccess, arg0: inst212, arg1: element0, type: type(symbolic_constant136)}
// CHECK:STDOUT:     inst217:         {kind: ImplWitnessAccess, arg0: inst212, arg1: element0, type: type(symbolic_constant133)}
// CHECK:STDOUT:     inst218:         {kind: FacetValue, arg0: inst28, arg1: inst_block70, type: type(inst71)}
// CHECK:STDOUT:     inst219:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst218, type: type(TypeType)}
// CHECK:STDOUT:     inst220:         {kind: ImplWitnessAccess, arg0: inst213, arg1: element0, type: type(symbolic_constant136)}
// CHECK:STDOUT:     inst221:         {kind: BoundMethod, arg0: inst63, arg1: inst216, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst222:         {kind: SpecificImplFunction, arg0: inst216, arg1: specific12, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst223:         {kind: SpecificImplFunction, arg0: inst217, arg1: specific12, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst224:         {kind: SpecificImplFunction, arg0: inst220, arg1: specific13, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst225:         {kind: BoundMethod, arg0: inst63, arg1: inst222, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst226:         {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst227:         {kind: Call, arg0: inst225, arg1: inst_block74, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst228:         {kind: InitializeFrom, arg0: inst227, arg1: inst67, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst229:         {kind: TupleAccess, arg0: inst52, arg1: element1, type: type(inst36)}
// CHECK:STDOUT:     inst230:         {kind: TupleInit, arg0: inst_block_empty, arg1: inst229, type: type(inst36)}
// CHECK:STDOUT:     inst231:         {kind: TupleValue, arg0: inst_block_empty, type: type(inst36)}
// CHECK:STDOUT:     inst232:         {kind: Converted, arg0: inst64, arg1: inst230, type: type(inst36)}
// CHECK:STDOUT:     inst233:         {kind: TupleInit, arg0: inst_block75, arg1: inst52, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst234:         {kind: Converted, arg0: inst65, arg1: inst233, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst235:         {kind: ReturnExpr, arg0: inst234, arg1: inst52}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst16:          concrete_constant(inst16)
// CHECK:STDOUT:       inst17:          concrete_constant(inst17)
// CHECK:STDOUT:       inst18:          symbolic_constant0
// CHECK:STDOUT:       inst19:          symbolic_constant0
// CHECK:STDOUT:       inst20:          symbolic_constant2
// CHECK:STDOUT:       inst21:          symbolic_constant1
// CHECK:STDOUT:       inst22:          symbolic_constant2
// CHECK:STDOUT:       inst23:          concrete_constant(inst23)
// CHECK:STDOUT:       inst24:          concrete_constant(inst24)
// CHECK:STDOUT:       inst25:          symbolic_constant2
// CHECK:STDOUT:       inst26:          symbolic_constant4
// CHECK:STDOUT:       inst27:          symbolic_constant3
// CHECK:STDOUT:       inst28:          symbolic_constant4
// CHECK:STDOUT:       inst30:          symbolic_constant5
// CHECK:STDOUT:       inst31:          concrete_constant(inst31)
// CHECK:STDOUT:       inst32:          symbolic_constant6
// CHECK:STDOUT:       inst33:          concrete_constant(inst33)
// CHECK:STDOUT:       inst34:          symbolic_constant2
// CHECK:STDOUT:       inst35:          symbolic_constant4
// CHECK:STDOUT:       inst36:          concrete_constant(inst36)
// CHECK:STDOUT:       inst38:          concrete_constant(inst38)
// CHECK:STDOUT:       inst40:          concrete_constant(inst40)
// CHECK:STDOUT:       inst41:          concrete_constant(inst36)
// CHECK:STDOUT:       inst42:          symbolic_constant7
// CHECK:STDOUT:       inst43:          symbolic_constant8
// CHECK:STDOUT:       inst44:          symbolic_constant8
// CHECK:STDOUT:       inst45:          symbolic_constant9
// CHECK:STDOUT:       inst46:          concrete_constant(inst46)
// CHECK:STDOUT:       inst47:          symbolic_constant10
// CHECK:STDOUT:       inst48:          concrete_constant(inst48)
// CHECK:STDOUT:       inst50:          symbolic_constant4
// CHECK:STDOUT:       inst53:          concrete_constant(inst55)
// CHECK:STDOUT:       inst54:          concrete_constant(inst54)
// CHECK:STDOUT:       inst55:          concrete_constant(inst55)
// CHECK:STDOUT:       inst56:          symbolic_constant11
// CHECK:STDOUT:       inst57:          symbolic_constant13
// CHECK:STDOUT:       inst58:          symbolic_constant12
// CHECK:STDOUT:       inst59:          symbolic_constant13
// CHECK:STDOUT:       inst60:          symbolic_constant15
// CHECK:STDOUT:       inst61:          symbolic_constant14
// CHECK:STDOUT:       inst62:          symbolic_constant15
// CHECK:STDOUT:       inst66:          symbolic_constant13
// CHECK:STDOUT:       inst68:          symbolic_constant15
// CHECK:STDOUT:       inst69:          concrete_constant(inst71)
// CHECK:STDOUT:       inst70:          concrete_constant(inst71)
// CHECK:STDOUT:       inst71:          concrete_constant(inst71)
// CHECK:STDOUT:       inst72:          symbolic_constant16
// CHECK:STDOUT:       inst73:          constant<none>
// CHECK:STDOUT:       inst74:          concrete_constant(inst78)
// CHECK:STDOUT:       inst75:          constant<none>
// CHECK:STDOUT:       inst76:          concrete_constant(inst76)
// CHECK:STDOUT:       inst77:          concrete_constant(inst81)
// CHECK:STDOUT:       inst78:          concrete_constant(inst78)
// CHECK:STDOUT:       inst79:          concrete_constant(inst81)
// CHECK:STDOUT:       inst80:          concrete_constant(inst80)
// CHECK:STDOUT:       inst81:          concrete_constant(inst81)
// CHECK:STDOUT:       inst82:          symbolic_constant17
// CHECK:STDOUT:       inst83:          symbolic_constant18
// CHECK:STDOUT:       inst84:          concrete_constant(inst84)
// CHECK:STDOUT:       inst85:          concrete_constant(inst85)
// CHECK:STDOUT:       inst86:          concrete_constant(inst86)
// CHECK:STDOUT:       inst87:          concrete_constant(inst87)
// CHECK:STDOUT:       inst88:          symbolic_constant16
// CHECK:STDOUT:       inst89:          symbolic_constant22
// CHECK:STDOUT:       inst90:          symbolic_constant23
// CHECK:STDOUT:       inst91:          symbolic_constant24
// CHECK:STDOUT:       inst92:          symbolic_constant131
// CHECK:STDOUT:       inst93:          constant<none>
// CHECK:STDOUT:       inst94:          concrete_constant(inst94)
// CHECK:STDOUT:       inst95:          concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst96:          concrete_constant(inst71)
// CHECK:STDOUT:       inst97:          constant<none>
// CHECK:STDOUT:       inst98:          concrete_constant(inst98)
// CHECK:STDOUT:       inst99:          concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst100:         concrete_constant(inst71)
// CHECK:STDOUT:       inst101:         constant<none>
// CHECK:STDOUT:       inst102:         concrete_constant(inst102)
// CHECK:STDOUT:       inst103:         concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst104:         concrete_constant(inst71)
// CHECK:STDOUT:       inst105:         constant<none>
// CHECK:STDOUT:       inst106:         concrete_constant(inst106)
// CHECK:STDOUT:       inst107:         concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst108:         concrete_constant(inst71)
// CHECK:STDOUT:       inst109:         symbolic_constant129
// CHECK:STDOUT:       inst110:         concrete_constant(inst110)
// CHECK:STDOUT:       inst111:         concrete_constant(inst111)
// CHECK:STDOUT:       inst112:         symbolic_constant26
// CHECK:STDOUT:       inst113:         symbolic_constant25
// CHECK:STDOUT:       inst114:         concrete_constant(inst71)
// CHECK:STDOUT:       inst115:         constant<none>
// CHECK:STDOUT:       inst116:         concrete_constant(inst116)
// CHECK:STDOUT:       inst117:         symbolic_constant28
// CHECK:STDOUT:       inst118:         symbolic_constant30
// CHECK:STDOUT:       inst119:         symbolic_constant31
// CHECK:STDOUT:       inst120:         symbolic_constant32
// CHECK:STDOUT:       inst121:         symbolic_constant35
// CHECK:STDOUT:       inst122:         symbolic_constant33
// CHECK:STDOUT:       inst123:         symbolic_constant34
// CHECK:STDOUT:       inst124:         concrete_constant(inst124)
// CHECK:STDOUT:       inst125:         concrete_constant(inst125)
// CHECK:STDOUT:       inst126:         concrete_constant(inst126)
// CHECK:STDOUT:       inst127:         concrete_constant(inst127)
// CHECK:STDOUT:       inst128:         symbolic_constant26
// CHECK:STDOUT:       inst129:         symbolic_constant40
// CHECK:STDOUT:       inst130:         symbolic_constant41
// CHECK:STDOUT:       inst131:         symbolic_constant42
// CHECK:STDOUT:       inst132:         symbolic_constant45
// CHECK:STDOUT:       inst133:         symbolic_constant46
// CHECK:STDOUT:       inst134:         symbolic_constant47
// CHECK:STDOUT:       inst135:         constant<none>
// CHECK:STDOUT:       inst136:         concrete_constant(inst136)
// CHECK:STDOUT:       inst137:         concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst138:         concrete_constant(inst71)
// CHECK:STDOUT:       inst139:         constant<none>
// CHECK:STDOUT:       inst140:         concrete_constant(inst140)
// CHECK:STDOUT:       inst141:         concrete_constant(inst36)
// CHECK:STDOUT:       inst142:         concrete_constant(inst71)
// CHECK:STDOUT:       inst143:         constant<none>
// CHECK:STDOUT:       inst144:         concrete_constant(inst144)
// CHECK:STDOUT:       inst145:         symbolic_constant48
// CHECK:STDOUT:       inst146:         symbolic_constant49
// CHECK:STDOUT:       inst147:         symbolic_constant50
// CHECK:STDOUT:       inst148:         symbolic_constant51
// CHECK:STDOUT:       inst149:         symbolic_constant52
// CHECK:STDOUT:       inst150:         concrete_constant(inst150)
// CHECK:STDOUT:       inst151:         concrete_constant(inst151)
// CHECK:STDOUT:       inst152:         concrete_constant(inst152)
// CHECK:STDOUT:       inst153:         symbolic_constant54
// CHECK:STDOUT:       inst154:         symbolic_constant55
// CHECK:STDOUT:       inst155:         symbolic_constant53
// CHECK:STDOUT:       inst156:         concrete_constant(inst71)
// CHECK:STDOUT:       inst157:         constant<none>
// CHECK:STDOUT:       inst158:         concrete_constant(inst158)
// CHECK:STDOUT:       inst159:         symbolic_constant60
// CHECK:STDOUT:       inst160:         symbolic_constant62
// CHECK:STDOUT:       inst161:         symbolic_constant63
// CHECK:STDOUT:       inst162:         symbolic_constant64
// CHECK:STDOUT:       inst163:         symbolic_constant65
// CHECK:STDOUT:       inst164:         symbolic_constant66
// CHECK:STDOUT:       inst165:         symbolic_constant67
// CHECK:STDOUT:       inst166:         symbolic_constant70
// CHECK:STDOUT:       inst167:         symbolic_constant68
// CHECK:STDOUT:       inst168:         symbolic_constant69
// CHECK:STDOUT:       inst169:         symbolic_constant72
// CHECK:STDOUT:       inst170:         concrete_constant(inst170)
// CHECK:STDOUT:       inst171:         concrete_constant(inst171)
// CHECK:STDOUT:       inst172:         concrete_constant(inst172)
// CHECK:STDOUT:       inst173:         concrete_constant(inst173)
// CHECK:STDOUT:       inst174:         symbolic_constant54
// CHECK:STDOUT:       inst175:         symbolic_constant55
// CHECK:STDOUT:       inst176:         symbolic_constant75
// CHECK:STDOUT:       inst177:         symbolic_constant76
// CHECK:STDOUT:       inst178:         symbolic_constant82
// CHECK:STDOUT:       inst179:         symbolic_constant83
// CHECK:STDOUT:       inst180:         symbolic_constant84
// CHECK:STDOUT:       inst181:         symbolic_constant85
// CHECK:STDOUT:       inst182:         symbolic_constant86
// CHECK:STDOUT:       inst183:         symbolic_constant87
// CHECK:STDOUT:       inst184:         symbolic_constant88
// CHECK:STDOUT:       inst185:         symbolic_constant90
// CHECK:STDOUT:       inst186:         symbolic_constant92
// CHECK:STDOUT:       inst187:         symbolic_constant94
// CHECK:STDOUT:       inst188:         symbolic_constant96
// CHECK:STDOUT:       inst189:         symbolic_constant98
// CHECK:STDOUT:       inst190:         symbolic_constant100
// CHECK:STDOUT:       inst191:         symbolic_constant102
// CHECK:STDOUT:       inst192:         symbolic_constant104
// CHECK:STDOUT:       inst193:         symbolic_constant106
// CHECK:STDOUT:       inst194:         symbolic_constant108
// CHECK:STDOUT:       inst195:         symbolic_constant110
// CHECK:STDOUT:       inst196:         symbolic_constant112
// CHECK:STDOUT:       inst197:         symbolic_constant114
// CHECK:STDOUT:       inst198:         symbolic_constant115
// CHECK:STDOUT:       inst199:         symbolic_constant116
// CHECK:STDOUT:       inst200:         symbolic_constant117
// CHECK:STDOUT:       inst201:         symbolic_constant118
// CHECK:STDOUT:       inst202:         symbolic_constant119
// CHECK:STDOUT:       inst203:         symbolic_constant120
// CHECK:STDOUT:       inst204:         symbolic_constant121
// CHECK:STDOUT:       inst205:         symbolic_constant122
// CHECK:STDOUT:       inst206:         symbolic_constant123
// CHECK:STDOUT:       inst207:         symbolic_constant124
// CHECK:STDOUT:       inst208:         symbolic_constant125
// CHECK:STDOUT:       inst209:         symbolic_constant126
// CHECK:STDOUT:       inst210:         symbolic_constant127
// CHECK:STDOUT:       inst211:         symbolic_constant128
// CHECK:STDOUT:       inst212:         symbolic_constant130
// CHECK:STDOUT:       inst213:         symbolic_constant131
// CHECK:STDOUT:       inst214:         symbolic_constant132
// CHECK:STDOUT:       inst215:         symbolic_constant133
// CHECK:STDOUT:       inst216:         symbolic_constant137
// CHECK:STDOUT:       inst217:         symbolic_constant134
// CHECK:STDOUT:       inst218:         symbolic_constant135
// CHECK:STDOUT:       inst219:         symbolic_constant136
// CHECK:STDOUT:       inst220:         symbolic_constant137
// CHECK:STDOUT:       inst222:         symbolic_constant139
// CHECK:STDOUT:       inst223:         symbolic_constant138
// CHECK:STDOUT:       inst224:         symbolic_constant139
// CHECK:STDOUT:       inst226:         symbolic_constant15
// CHECK:STDOUT:       inst230:         concrete_constant(inst231)
// CHECK:STDOUT:       inst231:         concrete_constant(inst231)
// CHECK:STDOUT:       inst232:         concrete_constant(inst231)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst19, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst21, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst21, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst27, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst27, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst30, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst30, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst42, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst42, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst45, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: inst45, generic: generic0, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: inst56, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: inst58, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: inst58, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant14: {inst: inst61, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant15: {inst: inst61, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant16: {inst: inst72, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant17: {inst: inst82, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant18: {inst: inst83, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant19: {inst: inst83, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant20: {inst: inst72, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant21: {inst: inst82, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant22: {inst: inst72, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant23: {inst: inst82, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant24: {inst: inst83, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant25: {inst: inst27, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant26: {inst: inst21, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant27: {inst: inst21, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant28: {inst: inst117, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant29: {inst: inst117, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant30: {inst: inst21, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant31: {inst: inst27, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant32: {inst: inst117, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant33: {inst: inst122, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant34: {inst: inst123, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant35: {inst: inst123, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant36: {inst: inst122, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant37: {inst: inst30, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant38: {inst: inst123, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant39: {inst: inst61, generic: generic2, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant40: {inst: inst122, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant41: {inst: inst123, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant42: {inst: inst61, generic: generic2, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant43: {inst: inst21, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant44: {inst: inst27, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant45: {inst: inst21, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant46: {inst: inst27, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant47: {inst: inst30, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant48: {inst: inst145, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant49: {inst: inst146, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant50: {inst: inst147, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant51: {inst: inst148, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant52: {inst: inst149, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant53: {inst: inst149, generic: generic4, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant54: {inst: inst147, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant55: {inst: inst145, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant56: {inst: inst147, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant57: {inst: inst145, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant58: {inst: inst148, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant59: {inst: inst146, generic: generic4, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60: {inst: inst159, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant61: {inst: inst159, generic: generic4, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant62: {inst: inst147, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant63: {inst: inst145, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant64: {inst: inst148, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant65: {inst: inst146, generic: generic4, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant66: {inst: inst149, generic: generic4, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant67: {inst: inst159, generic: generic4, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant68: {inst: inst167, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant69: {inst: inst168, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant70: {inst: inst168, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant71: {inst: inst167, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant72: {inst: inst169, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant73: {inst: inst169, generic: generic5, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant74: {inst: inst168, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant75: {inst: inst167, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant76: {inst: inst168, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant77: {inst: inst147, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant78: {inst: inst148, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant79: {inst: inst145, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant80: {inst: inst146, generic: generic5, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant81: {inst: inst149, generic: generic5, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant82: {inst: inst147, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant83: {inst: inst148, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant84: {inst: inst145, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant85: {inst: inst146, generic: generic5, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant86: {inst: inst149, generic: generic5, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant87: {inst: inst169, generic: generic5, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant88: {inst: inst184, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant89: {inst: inst184, generic: generic5, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant90: {inst: inst185, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant91: {inst: inst185, generic: generic5, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant92: {inst: inst186, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant93: {inst: inst186, generic: generic5, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant94: {inst: inst187, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant95: {inst: inst187, generic: generic5, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant96: {inst: inst188, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant97: {inst: inst188, generic: generic5, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant98: {inst: inst189, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant99: {inst: inst189, generic: generic5, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant100: {inst: inst190, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant101: {inst: inst190, generic: generic5, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant102: {inst: inst191, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant103: {inst: inst191, generic: generic5, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant104: {inst: inst192, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant105: {inst: inst192, generic: generic5, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant106: {inst: inst193, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant107: {inst: inst193, generic: generic5, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant108: {inst: inst194, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant109: {inst: inst194, generic: generic5, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant110: {inst: inst195, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant111: {inst: inst195, generic: generic5, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant112: {inst: inst196, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant113: {inst: inst196, generic: generic5, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant114: {inst: inst184, generic: generic5, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant115: {inst: inst185, generic: generic5, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant116: {inst: inst186, generic: generic5, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant117: {inst: inst187, generic: generic5, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant118: {inst: inst188, generic: generic5, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant119: {inst: inst189, generic: generic5, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant120: {inst: inst190, generic: generic5, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant121: {inst: inst191, generic: generic5, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant122: {inst: inst192, generic: generic5, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant123: {inst: inst193, generic: generic5, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant124: {inst: inst194, generic: generic5, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant125: {inst: inst195, generic: generic5, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant126: {inst: inst196, generic: generic5, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant127: {inst: inst210, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant128: {inst: inst211, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant129: {inst: inst117, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant130: {inst: inst212, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant131: {inst: inst212, generic: generic0, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant132: {inst: inst214, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant133: {inst: inst215, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant134: {inst: inst217, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant135: {inst: inst214, generic: generic0, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant136: {inst: inst215, generic: generic0, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant137: {inst: inst217, generic: generic0, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant138: {inst: inst223, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant139: {inst: inst223, generic: generic0, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst53
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:       1:               inst69
// CHECK:STDOUT:       2:               inst70
// CHECK:STDOUT:       3:               inst73
// CHECK:STDOUT:       4:               inst74
// CHECK:STDOUT:       5:               inst75
// CHECK:STDOUT:       6:               inst77
// CHECK:STDOUT:       7:               inst79
// CHECK:STDOUT:       8:               inst84
// CHECK:STDOUT:       9:               inst85
// CHECK:STDOUT:       10:              inst86
// CHECK:STDOUT:       11:              inst87
// CHECK:STDOUT:       12:              inst88
// CHECK:STDOUT:       13:              inst93
// CHECK:STDOUT:       14:              inst94
// CHECK:STDOUT:       15:              inst95
// CHECK:STDOUT:       16:              inst96
// CHECK:STDOUT:       17:              inst97
// CHECK:STDOUT:       18:              inst98
// CHECK:STDOUT:       19:              inst99
// CHECK:STDOUT:       20:              inst100
// CHECK:STDOUT:       21:              inst101
// CHECK:STDOUT:       22:              inst102
// CHECK:STDOUT:       23:              inst103
// CHECK:STDOUT:       24:              inst104
// CHECK:STDOUT:       25:              inst105
// CHECK:STDOUT:       26:              inst106
// CHECK:STDOUT:       27:              inst107
// CHECK:STDOUT:       28:              inst108
// CHECK:STDOUT:       29:              inst109
// CHECK:STDOUT:       30:              inst110
// CHECK:STDOUT:       31:              inst111
// CHECK:STDOUT:       32:              inst112
// CHECK:STDOUT:       33:              inst113
// CHECK:STDOUT:       34:              inst114
// CHECK:STDOUT:       35:              inst115
// CHECK:STDOUT:       36:              inst116
// CHECK:STDOUT:       37:              inst121
// CHECK:STDOUT:       38:              inst124
// CHECK:STDOUT:       39:              inst125
// CHECK:STDOUT:       40:              inst126
// CHECK:STDOUT:       41:              inst127
// CHECK:STDOUT:       42:              inst128
// CHECK:STDOUT:       43:              inst135
// CHECK:STDOUT:       44:              inst136
// CHECK:STDOUT:       45:              inst137
// CHECK:STDOUT:       46:              inst138
// CHECK:STDOUT:       47:              inst139
// CHECK:STDOUT:       48:              inst140
// CHECK:STDOUT:       49:              inst141
// CHECK:STDOUT:       50:              inst142
// CHECK:STDOUT:       51:              inst143
// CHECK:STDOUT:       52:              inst144
// CHECK:STDOUT:       53:              inst151
// CHECK:STDOUT:       54:              inst152
// CHECK:STDOUT:       55:              inst153
// CHECK:STDOUT:       56:              inst154
// CHECK:STDOUT:       57:              inst155
// CHECK:STDOUT:       58:              inst156
// CHECK:STDOUT:       59:              inst157
// CHECK:STDOUT:       60:              inst158
// CHECK:STDOUT:       61:              inst166
// CHECK:STDOUT:       62:              inst170
// CHECK:STDOUT:       63:              inst171
// CHECK:STDOUT:       64:              inst172
// CHECK:STDOUT:       65:              inst173
// CHECK:STDOUT:       66:              inst174
// CHECK:STDOUT:       67:              inst175
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst24
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst25
// CHECK:STDOUT:       1:               inst26
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst33
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst35
// CHECK:STDOUT:       1:               inst37
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst35
// CHECK:STDOUT:       1:               inst41
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst27
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst28
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst49
// CHECK:STDOUT:       1:               inst51
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst24
// CHECK:STDOUT:       1:               inst31
// CHECK:STDOUT:       2:               inst33
// CHECK:STDOUT:       3:               inst46
// CHECK:STDOUT:       4:               inst48
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst34
// CHECK:STDOUT:       1:               inst35
// CHECK:STDOUT:       2:               inst37
// CHECK:STDOUT:       3:               inst39
// CHECK:STDOUT:       4:               inst41
// CHECK:STDOUT:       5:               inst43
// CHECK:STDOUT:       6:               inst18
// CHECK:STDOUT:       7:               inst20
// CHECK:STDOUT:       8:               inst49
// CHECK:STDOUT:       9:               inst50
// CHECK:STDOUT:       10:              inst29
// CHECK:STDOUT:       11:              inst51
// CHECK:STDOUT:       12:              inst52
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst20
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst22
// CHECK:STDOUT:       1:               inst28
// CHECK:STDOUT:       2:               inst32
// CHECK:STDOUT:       3:               inst44
// CHECK:STDOUT:       4:               inst47
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst30
// CHECK:STDOUT:       3:               inst42
// CHECK:STDOUT:       4:               inst45
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:       1:               inst64
// CHECK:STDOUT:       2:               inst65
// CHECK:STDOUT:       3:               inst216
// CHECK:STDOUT:       4:               inst221
// CHECK:STDOUT:       5:               inst222
// CHECK:STDOUT:       6:               inst225
// CHECK:STDOUT:       7:               inst227
// CHECK:STDOUT:       8:               inst67
// CHECK:STDOUT:       9:               inst228
// CHECK:STDOUT:       10:              inst229
// CHECK:STDOUT:       11:              inst230
// CHECK:STDOUT:       12:              inst232
// CHECK:STDOUT:       13:              inst233
// CHECK:STDOUT:       14:              inst234
// CHECK:STDOUT:       15:              inst235
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:       1:               inst64
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst75
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst87
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst88
// CHECK:STDOUT:     inst_block25:
// CHECK:STDOUT:       0:               inst89
// CHECK:STDOUT:       1:               inst90
// CHECK:STDOUT:       2:               inst91
// CHECK:STDOUT:     inst_block26:
// CHECK:STDOUT:       0:               inst72
// CHECK:STDOUT:     inst_block27:
// CHECK:STDOUT:       0:               inst72
// CHECK:STDOUT:       1:               inst82
// CHECK:STDOUT:       2:               inst83
// CHECK:STDOUT:     inst_block28:
// CHECK:STDOUT:       0:               inst111
// CHECK:STDOUT:     inst_block29:
// CHECK:STDOUT:       0:               inst112
// CHECK:STDOUT:     inst_block30:
// CHECK:STDOUT:       0:               inst115
// CHECK:STDOUT:     inst_block31:
// CHECK:STDOUT:       0:               inst118
// CHECK:STDOUT:     inst_block32:
// CHECK:STDOUT:       0:               inst118
// CHECK:STDOUT:       1:               inst119
// CHECK:STDOUT:       2:               inst120
// CHECK:STDOUT:     inst_block33:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst117
// CHECK:STDOUT:     inst_block34:
// CHECK:STDOUT:       0:               inst127
// CHECK:STDOUT:     inst_block35:
// CHECK:STDOUT:       0:               inst128
// CHECK:STDOUT:     inst_block36:
// CHECK:STDOUT:       0:               inst118
// CHECK:STDOUT:     inst_block37:
// CHECK:STDOUT:       0:               inst129
// CHECK:STDOUT:       1:               inst130
// CHECK:STDOUT:       2:               inst131
// CHECK:STDOUT:     inst_block38:
// CHECK:STDOUT:       0:               inst132
// CHECK:STDOUT:       1:               inst133
// CHECK:STDOUT:       2:               inst134
// CHECK:STDOUT:     inst_block39:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst30
// CHECK:STDOUT:     inst_block40:
// CHECK:STDOUT:       0:               inst148
// CHECK:STDOUT:       1:               inst146
// CHECK:STDOUT:     inst_block41:
// CHECK:STDOUT:       0:               inst152
// CHECK:STDOUT:       1:               inst151
// CHECK:STDOUT:     inst_block42:
// CHECK:STDOUT:       0:               inst153
// CHECK:STDOUT:       1:               inst154
// CHECK:STDOUT:     inst_block43:
// CHECK:STDOUT:       0:               inst157
// CHECK:STDOUT:     inst_block44:
// CHECK:STDOUT:       0:               inst147
// CHECK:STDOUT:       1:               inst145
// CHECK:STDOUT:     inst_block45:
// CHECK:STDOUT:       0:               inst162
// CHECK:STDOUT:       1:               inst163
// CHECK:STDOUT:     inst_block46:
// CHECK:STDOUT:       0:               inst160
// CHECK:STDOUT:       1:               inst161
// CHECK:STDOUT:     inst_block47:
// CHECK:STDOUT:       0:               inst160
// CHECK:STDOUT:       1:               inst161
// CHECK:STDOUT:       2:               inst162
// CHECK:STDOUT:       3:               inst163
// CHECK:STDOUT:       4:               inst164
// CHECK:STDOUT:       5:               inst165
// CHECK:STDOUT:     inst_block48:
// CHECK:STDOUT:       0:               inst147
// CHECK:STDOUT:       1:               inst145
// CHECK:STDOUT:       2:               inst148
// CHECK:STDOUT:       3:               inst146
// CHECK:STDOUT:       4:               inst149
// CHECK:STDOUT:       5:               inst159
// CHECK:STDOUT:     inst_block49:
// CHECK:STDOUT:       0:               inst173
// CHECK:STDOUT:     inst_block50:
// CHECK:STDOUT:       0:               inst174
// CHECK:STDOUT:       1:               inst175
// CHECK:STDOUT:     inst_block51:
// CHECK:STDOUT:       0:               inst160
// CHECK:STDOUT:       1:               inst161
// CHECK:STDOUT:     inst_block52:
// CHECK:STDOUT:       0:               inst176
// CHECK:STDOUT:       1:               inst177
// CHECK:STDOUT:     inst_block53:
// CHECK:STDOUT:       0:               inst179
// CHECK:STDOUT:       1:               inst181
// CHECK:STDOUT:     inst_block54:
// CHECK:STDOUT:       0:               inst178
// CHECK:STDOUT:       1:               inst179
// CHECK:STDOUT:       2:               inst180
// CHECK:STDOUT:       3:               inst181
// CHECK:STDOUT:       4:               inst182
// CHECK:STDOUT:       5:               inst183
// CHECK:STDOUT:     inst_block55:
// CHECK:STDOUT:       0:               inst147
// CHECK:STDOUT:       1:               inst148
// CHECK:STDOUT:       2:               inst145
// CHECK:STDOUT:       3:               inst146
// CHECK:STDOUT:       4:               inst149
// CHECK:STDOUT:       5:               inst169
// CHECK:STDOUT:     inst_block56:
// CHECK:STDOUT:       0:               inst186
// CHECK:STDOUT:     inst_block57:
// CHECK:STDOUT:       0:               inst187
// CHECK:STDOUT:     inst_block58:
// CHECK:STDOUT:       0:               inst192
// CHECK:STDOUT:     inst_block59:
// CHECK:STDOUT:       0:               inst193
// CHECK:STDOUT:     inst_block60:
// CHECK:STDOUT:       0:               inst199
// CHECK:STDOUT:     inst_block61:
// CHECK:STDOUT:       0:               inst200
// CHECK:STDOUT:     inst_block62:
// CHECK:STDOUT:       0:               inst205
// CHECK:STDOUT:     inst_block63:
// CHECK:STDOUT:       0:               inst206
// CHECK:STDOUT:     inst_block64:
// CHECK:STDOUT:       0:               inst197
// CHECK:STDOUT:       1:               inst198
// CHECK:STDOUT:       2:               inst199
// CHECK:STDOUT:       3:               inst200
// CHECK:STDOUT:       4:               inst201
// CHECK:STDOUT:       5:               inst202
// CHECK:STDOUT:       6:               inst203
// CHECK:STDOUT:       7:               inst204
// CHECK:STDOUT:       8:               inst205
// CHECK:STDOUT:       9:               inst206
// CHECK:STDOUT:       10:              inst207
// CHECK:STDOUT:       11:              inst208
// CHECK:STDOUT:       12:              inst209
// CHECK:STDOUT:     inst_block65:
// CHECK:STDOUT:       0:               inst193
// CHECK:STDOUT:       1:               inst146
// CHECK:STDOUT:       2:               inst210
// CHECK:STDOUT:     inst_block66:
// CHECK:STDOUT:       0:               inst187
// CHECK:STDOUT:       1:               inst148
// CHECK:STDOUT:       2:               inst211
// CHECK:STDOUT:     inst_block67:
// CHECK:STDOUT:       0:               inst167
// CHECK:STDOUT:       1:               inst168
// CHECK:STDOUT:     inst_block68:
// CHECK:STDOUT:       0:               inst122
// CHECK:STDOUT:       1:               inst123
// CHECK:STDOUT:       2:               inst61
// CHECK:STDOUT:     inst_block69:
// CHECK:STDOUT:       0:               inst212
// CHECK:STDOUT:     inst_block70:
// CHECK:STDOUT:       0:               inst213
// CHECK:STDOUT:     inst_block71:
// CHECK:STDOUT:       0:               inst214
// CHECK:STDOUT:     inst_block72:
// CHECK:STDOUT:       0:               inst214
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst30
// CHECK:STDOUT:     inst_block73:
// CHECK:STDOUT:       0:               inst218
// CHECK:STDOUT:     inst_block74:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:     inst_block75:
// CHECK:STDOUT:       0:               inst228
// CHECK:STDOUT:       1:               inst232
// CHECK:STDOUT:     inst_block76:
// CHECK:STDOUT:       0:               inst59
// CHECK:STDOUT:       1:               inst62
// CHECK:STDOUT:       2:               inst213
// CHECK:STDOUT:       3:               inst218
// CHECK:STDOUT:       4:               inst219
// CHECK:STDOUT:       5:               inst220
// CHECK:STDOUT:       6:               inst224
// CHECK:STDOUT:     inst_block77:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst15
// CHECK:STDOUT:       2:               inst53
// CHECK:STDOUT: ...

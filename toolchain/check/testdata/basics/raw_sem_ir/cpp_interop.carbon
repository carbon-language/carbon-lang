// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --builtin-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/cpp_interop.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/cpp_interop.carbon

// --- header.h

struct X {
  X* _Nonnull p;
};

void f(X x = {}) {
}

X* _Nonnull global;

// --- import.carbon

import Cpp library "header.h";

fn G(x: Cpp.X) {
  Cpp.f();
  Cpp.f(x);
  Cpp.f(*Cpp.global);
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        import.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           G
// CHECK:STDOUT:     name1:           x
// CHECK:STDOUT:     name2:           X
// CHECK:STDOUT:     name3:           f
// CHECK:STDOUT:     name4:           global
// CHECK:STDOUT:     name5:           p
// CHECK:STDOUT:     name6:           f__carbon_thunk
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc60000000}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc60000001}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc60000002}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc60000003}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc60000004}
// CHECK:STDOUT:   clang_decls:
// CHECK:STDOUT:     clang_decl_id60000000: {key: "<translation unit>", inst_id: inst60000011}
// CHECK:STDOUT:     clang_decl_id60000001: {key: "struct X {}", inst_id: inst60000014}
// CHECK:STDOUT:     clang_decl_id60000002: {key: "X * _Nonnull p", inst_id: inst60000022}
// CHECK:STDOUT:     clang_decl_id60000003: {key: {decl: "void f(X x = {})", kind: normal, num_params: 0}, inst_id: inst6000002D}
// CHECK:STDOUT:     clang_decl_id60000004: {key: {decl: "extern void f__carbon_thunk()", kind: normal, num_params: 0}, inst_id: inst60000030}
// CHECK:STDOUT:     clang_decl_id60000005: {key: {decl: "void f(X x = {})", kind: normal, num_params: 1}, inst_id: inst6000003B}
// CHECK:STDOUT:     clang_decl_id60000006: {key: {decl: "extern void f__carbon_thunk(X * _Nonnull x)", kind: normal, num_params: 1}, inst_id: inst60000043}
// CHECK:STDOUT:     clang_decl_id60000007: {key: "X * _Nonnull global", inst_id: inst6000004E}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Cpp): inst60000011, name0: inst6000001D}}
// CHECK:STDOUT:     name_scope60000001: {inst: inst60000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name2: inst60000014, name3: inst6000002A, name4: inst6000004E}}
// CHECK:STDOUT:     name_scope60000002: {inst: inst60000014, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0}
// CHECK:STDOUT:     entity_name60000001: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0}
// CHECK:STDOUT:     entity_name60000002: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0}
// CHECK:STDOUT:     entity_name60000003: {name: name4, parent_scope: name_scope60000001, index: -1, is_template: 0, is_unused: 0}
// CHECK:STDOUT:   cpp_global_vars:
// CHECK:STDOUT:     cpp_global_var60000000: {key: {entity_name_id: entity_name60000003}, clang_decl_id: clang_decl_id60000007}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function60000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block60000006, call_params_id: inst_block60000007, body: [inst_block6000000A]}
// CHECK:STDOUT:     function60000001: {name: name3, parent_scope: name_scope60000001, call_param_patterns_id: inst_block_empty, call_params_id: inst_block_empty}
// CHECK:STDOUT:     function60000002: {name: name6, parent_scope: name_scope60000001, call_param_patterns_id: inst_block_empty, call_params_id: inst_block_empty}
// CHECK:STDOUT:     function60000003: {name: name3, parent_scope: name_scope60000001, call_param_patterns_id: inst_block6000000E, call_params_id: inst_block6000000F}
// CHECK:STDOUT:     function60000004: {name: name6, parent_scope: name_scope60000001, call_param_patterns_id: inst_block60000014, call_params_id: inst_block60000015}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class60000000:   {name: name2, parent_scope: name_scope60000001, self_type_id: type(inst60000015), inheritance_kind: Base, is_dynamic: 0, scope_id: name_scope60000002, body_block_id: inst_block6000000B, adapt_id: inst<none>, base_id: inst<none>, complete_type_witness_id: inst60000025, vtable_decl_id: inst<none>}}
// CHECK:STDOUT:   interfaces:      {}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   specific_interfaces: {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:     struct_type_fields60000001:
// CHECK:STDOUT:       0:               {name_id: name5, type_inst_id: inst60000020}
// CHECK:STDOUT:     struct_type_fields60000002:
// CHECK:STDOUT:       0:               {name_id: name5, type_inst_id: inst60000020}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst60000013)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst6000001E)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst60000020)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst60000020)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(inst60000027)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst60000027)}
// CHECK:STDOUT:     'type(inst60000024)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst60000027)}
// CHECK:STDOUT:     'type(inst60000015)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst60000027)}
// CHECK:STDOUT:     'type(inst60000029)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst6000002E)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst60000031)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst6000003C)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:     'type(inst60000044)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000013)}
// CHECK:STDOUT:   facet_types:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: TypeType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(AutoType)':  {kind: AutoType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoolType)':  {kind: BoolType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoundMethodType)': {kind: BoundMethodType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(CharLiteralType)': {kind: CharLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(ErrorInst)': {kind: ErrorInst, type: type(Error)}
// CHECK:STDOUT:     'inst(FloatLiteralType)': {kind: FloatLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(FormType)':  {kind: FormType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(InstType)':  {kind: InstType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(IntLiteralType)': {kind: IntLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(NamespaceType)': {kind: NamespaceType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(RequireSpecificDefinitionType)': {kind: RequireSpecificDefinitionType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(SpecificFunctionType)': {kind: SpecificFunctionType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(VtableType)': {kind: VtableType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(WitnessType)': {kind: WitnessType, type: type(TypeType)}
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000010:    {kind: ImportCppDecl}
// CHECK:STDOUT:     inst60000011:    {kind: Namespace, arg0: name_scope60000001, arg1: inst60000010, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000012:    {kind: NameRef, arg0: name(Cpp), arg1: inst60000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000013:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000014:    {kind: ClassDecl, arg0: class60000000, arg1: inst_block<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000015:    {kind: ClassType, arg0: class60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000016:    {kind: NameRef, arg0: name2, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst60000017:    {kind: ValueBinding, arg0: entity_name60000000, arg1: inst6000001B, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000018:    {kind: PatternType, arg0: inst60000015, type: type(TypeType)}
// CHECK:STDOUT:     inst60000019:    {kind: ValueBindingPattern, arg0: entity_name60000000, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001A:    {kind: ValueParamPattern, arg0: inst60000019, arg1: call_param0, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001B:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst60000015)}
// CHECK:STDOUT:     inst6000001C:    {kind: SpliceBlock, arg0: inst_block60000004, arg1: inst60000016, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block60000009, type: type(inst6000001E)}
// CHECK:STDOUT:     inst6000001E:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001F:    {kind: StructValue, arg0: inst_block_empty, type: type(inst6000001E)}
// CHECK:STDOUT:     inst60000020:    {kind: PointerType, arg0: inst60000015, type: type(TypeType)}
// CHECK:STDOUT:     inst60000021:    {kind: UnboundElementType, arg0: inst60000015, arg1: inst60000020, type: type(TypeType)}
// CHECK:STDOUT:     inst60000022:    {kind: FieldDecl, arg0: name5, arg1: element0, type: type(inst60000021)}
// CHECK:STDOUT:     inst60000023:    {kind: CustomLayoutType, arg0: struct_type_fields60000001, arg1: custom_layout60000001, type: type(TypeType)}
// CHECK:STDOUT:     inst60000024:    {kind: CustomLayoutType, arg0: struct_type_fields60000002, arg1: custom_layout60000001, type: type(TypeType)}
// CHECK:STDOUT:     inst60000025:    {kind: CompleteTypeWitness, arg0: inst60000023, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000026:    {kind: CompleteTypeWitness, arg0: inst60000024, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000027:    {kind: PointerType, arg0: inst60000024, type: type(TypeType)}
// CHECK:STDOUT:     inst60000028:    {kind: NameRef, arg0: name(Cpp), arg1: inst60000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000029:    {kind: CppOverloadSetType, arg0: cpp_overload_set60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002A:    {kind: CppOverloadSetValue, arg0: cpp_overload_set60000000, type: type(inst60000029)}
// CHECK:STDOUT:     inst6000002B:    {kind: CppOverloadSetValue, arg0: cpp_overload_set60000000, type: type(inst60000029)}
// CHECK:STDOUT:     inst6000002C:    {kind: NameRef, arg0: name3, arg1: inst6000002A, type: type(inst60000029)}
// CHECK:STDOUT:     inst6000002D:    {kind: FunctionDecl, arg0: function60000001, arg1: inst_block_empty, type: type(inst6000002E)}
// CHECK:STDOUT:     inst6000002E:    {kind: FunctionType, arg0: function60000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002F:    {kind: StructValue, arg0: inst_block_empty, type: type(inst6000002E)}
// CHECK:STDOUT:     inst60000030:    {kind: FunctionDecl, arg0: function60000002, arg1: inst_block_empty, type: type(inst60000031)}
// CHECK:STDOUT:     inst60000031:    {kind: FunctionType, arg0: function60000002, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000032:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000031)}
// CHECK:STDOUT:     inst60000033:    {kind: Call, arg0: inst60000030, arg1: inst_block_empty, type: type(inst60000013)}
// CHECK:STDOUT:     inst60000034:    {kind: NameRef, arg0: name(Cpp), arg1: inst60000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000035:    {kind: NameRef, arg0: name3, arg1: inst6000002A, type: type(inst60000029)}
// CHECK:STDOUT:     inst60000036:    {kind: NameRef, arg0: name1, arg1: inst60000017, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000037:    {kind: ValueBinding, arg0: entity_name60000001, arg1: inst6000003A, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000038:    {kind: ValueBindingPattern, arg0: entity_name60000001, type: type(inst60000018)}
// CHECK:STDOUT:     inst60000039:    {kind: ValueParamPattern, arg0: inst60000038, arg1: call_param0, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000003A:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst60000015)}
// CHECK:STDOUT:     inst6000003B:    {kind: FunctionDecl, arg0: function60000003, arg1: inst_block60000011, type: type(inst6000003C)}
// CHECK:STDOUT:     inst6000003C:    {kind: FunctionType, arg0: function60000003, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000003D:    {kind: StructValue, arg0: inst_block_empty, type: type(inst6000003C)}
// CHECK:STDOUT:     inst6000003E:    {kind: ValueBinding, arg0: entity_name60000002, arg1: inst60000042, type: type(inst60000020)}
// CHECK:STDOUT:     inst6000003F:    {kind: PatternType, arg0: inst60000020, type: type(TypeType)}
// CHECK:STDOUT:     inst60000040:    {kind: ValueBindingPattern, arg0: entity_name60000002, type: type(inst6000003F)}
// CHECK:STDOUT:     inst60000041:    {kind: ValueParamPattern, arg0: inst60000040, arg1: call_param0, type: type(inst6000003F)}
// CHECK:STDOUT:     inst60000042:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst60000020)}
// CHECK:STDOUT:     inst60000043:    {kind: FunctionDecl, arg0: function60000004, arg1: inst_block60000017, type: type(inst60000044)}
// CHECK:STDOUT:     inst60000044:    {kind: FunctionType, arg0: function60000004, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000045:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000044)}
// CHECK:STDOUT:     inst60000046:    {kind: ValueAsRef, arg0: inst60000036, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000047:    {kind: AddrOf, arg0: inst60000046, type: type(inst60000020)}
// CHECK:STDOUT:     inst60000048:    {kind: Call, arg0: inst60000043, arg1: inst_block60000019, type: type(inst60000013)}
// CHECK:STDOUT:     inst60000049:    {kind: NameRef, arg0: name(Cpp), arg1: inst60000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000004A:    {kind: NameRef, arg0: name3, arg1: inst6000002A, type: type(inst60000029)}
// CHECK:STDOUT:     inst6000004B:    {kind: NameRef, arg0: name(Cpp), arg1: inst60000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000004C:    {kind: RefBindingPattern, arg0: entity_name60000003, type: type(inst6000003F)}
// CHECK:STDOUT:     inst6000004D:    {kind: VarPattern, arg0: inst6000004C, type: type(inst6000003F)}
// CHECK:STDOUT:     inst6000004E:    {kind: VarStorage, arg0: inst6000004D, type: type(inst60000020)}
// CHECK:STDOUT:     inst6000004F:    {kind: NameRef, arg0: name4, arg1: inst6000004E, type: type(inst60000020)}
// CHECK:STDOUT:     inst60000050:    {kind: AcquireValue, arg0: inst6000004F, type: type(inst60000020)}
// CHECK:STDOUT:     inst60000051:    {kind: Deref, arg0: inst60000050, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000052:    {kind: AcquireValue, arg0: inst60000051, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000053:    {kind: ValueAsRef, arg0: inst60000052, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000054:    {kind: AddrOf, arg0: inst60000053, type: type(inst60000020)}
// CHECK:STDOUT:     inst60000055:    {kind: Call, arg0: inst60000043, arg1: inst_block6000001B, type: type(inst60000013)}
// CHECK:STDOUT:     inst60000056:    {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(AutoType)':  concrete_constant(inst(AutoType))
// CHECK:STDOUT:       'inst(BoolType)':  concrete_constant(inst(BoolType))
// CHECK:STDOUT:       'inst(BoundMethodType)': concrete_constant(inst(BoundMethodType))
// CHECK:STDOUT:       'inst(CharLiteralType)': concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       'inst(ErrorInst)': concrete_constant(inst(ErrorInst))
// CHECK:STDOUT:       'inst(FloatLiteralType)': concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       'inst(FormType)':  concrete_constant(inst(FormType))
// CHECK:STDOUT:       'inst(InstType)':  concrete_constant(inst(InstType))
// CHECK:STDOUT:       'inst(IntLiteralType)': concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       'inst(NamespaceType)': concrete_constant(inst(NamespaceType))
// CHECK:STDOUT:       'inst(RequireSpecificDefinitionType)': concrete_constant(inst(RequireSpecificDefinitionType))
// CHECK:STDOUT:       'inst(SpecificFunctionType)': concrete_constant(inst(SpecificFunctionType))
// CHECK:STDOUT:       'inst(VtableType)': concrete_constant(inst(VtableType))
// CHECK:STDOUT:       'inst(WitnessType)': concrete_constant(inst(WitnessType))
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000012:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000013:    concrete_constant(inst60000013)
// CHECK:STDOUT:       inst60000014:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst60000015:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst60000016:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst60000018:    concrete_constant(inst60000018)
// CHECK:STDOUT:       inst60000019:    concrete_constant(inst60000019)
// CHECK:STDOUT:       inst6000001A:    concrete_constant(inst6000001A)
// CHECK:STDOUT:       inst6000001C:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst6000001D:    concrete_constant(inst6000001F)
// CHECK:STDOUT:       inst6000001E:    concrete_constant(inst6000001E)
// CHECK:STDOUT:       inst6000001F:    concrete_constant(inst6000001F)
// CHECK:STDOUT:       inst60000020:    concrete_constant(inst60000020)
// CHECK:STDOUT:       inst60000021:    concrete_constant(inst60000021)
// CHECK:STDOUT:       inst60000022:    concrete_constant(inst60000022)
// CHECK:STDOUT:       inst60000023:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst60000024:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst60000025:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000026:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000027:    concrete_constant(inst60000027)
// CHECK:STDOUT:       inst60000028:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000029:    concrete_constant(inst60000029)
// CHECK:STDOUT:       inst6000002A:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst6000002B:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst6000002C:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst6000002D:    concrete_constant(inst6000002F)
// CHECK:STDOUT:       inst6000002E:    concrete_constant(inst6000002E)
// CHECK:STDOUT:       inst6000002F:    concrete_constant(inst6000002F)
// CHECK:STDOUT:       inst60000030:    concrete_constant(inst60000032)
// CHECK:STDOUT:       inst60000031:    concrete_constant(inst60000031)
// CHECK:STDOUT:       inst60000032:    concrete_constant(inst60000032)
// CHECK:STDOUT:       inst60000034:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000035:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst60000038:    concrete_constant(inst60000038)
// CHECK:STDOUT:       inst60000039:    concrete_constant(inst60000039)
// CHECK:STDOUT:       inst6000003B:    concrete_constant(inst6000003D)
// CHECK:STDOUT:       inst6000003C:    concrete_constant(inst6000003C)
// CHECK:STDOUT:       inst6000003D:    concrete_constant(inst6000003D)
// CHECK:STDOUT:       inst6000003F:    concrete_constant(inst6000003F)
// CHECK:STDOUT:       inst60000040:    concrete_constant(inst60000040)
// CHECK:STDOUT:       inst60000041:    concrete_constant(inst60000041)
// CHECK:STDOUT:       inst60000043:    concrete_constant(inst60000045)
// CHECK:STDOUT:       inst60000044:    concrete_constant(inst60000044)
// CHECK:STDOUT:       inst60000045:    concrete_constant(inst60000045)
// CHECK:STDOUT:       inst60000049:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst6000004A:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst6000004B:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst6000004C:    concrete_constant(inst6000004C)
// CHECK:STDOUT:       inst6000004D:    concrete_constant(inst6000004D)
// CHECK:STDOUT:       inst6000004E:    concrete_constant(inst6000004E)
// CHECK:STDOUT:       inst6000004F:    concrete_constant(inst6000004E)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst6000001D
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst60000011
// CHECK:STDOUT:       1:               inst60000014
// CHECK:STDOUT:       2:               inst6000002A
// CHECK:STDOUT:       3:               inst6000002D
// CHECK:STDOUT:       4:               inst60000030
// CHECK:STDOUT:       5:               inst6000003B
// CHECK:STDOUT:       6:               inst60000043
// CHECK:STDOUT:       7:               inst6000004C
// CHECK:STDOUT:       8:               inst6000004D
// CHECK:STDOUT:       9:               inst6000004E
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block60000004:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:       1:               inst60000016
// CHECK:STDOUT:     inst_block60000005:
// CHECK:STDOUT:       0:               inst6000001A
// CHECK:STDOUT:     inst_block60000006:
// CHECK:STDOUT:       0:               inst6000001A
// CHECK:STDOUT:     inst_block60000007:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:     inst_block60000008:
// CHECK:STDOUT:       0:               inst60000019
// CHECK:STDOUT:       1:               inst6000001A
// CHECK:STDOUT:     inst_block60000009:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:       2:               inst60000017
// CHECK:STDOUT:     inst_block6000000A:
// CHECK:STDOUT:       0:               inst60000028
// CHECK:STDOUT:       1:               inst6000002C
// CHECK:STDOUT:       2:               inst60000033
// CHECK:STDOUT:       3:               inst60000034
// CHECK:STDOUT:       4:               inst60000035
// CHECK:STDOUT:       5:               inst60000036
// CHECK:STDOUT:       6:               inst60000046
// CHECK:STDOUT:       7:               inst60000047
// CHECK:STDOUT:       8:               inst60000048
// CHECK:STDOUT:       9:               inst60000049
// CHECK:STDOUT:       10:              inst6000004A
// CHECK:STDOUT:       11:              inst6000004B
// CHECK:STDOUT:       12:              inst6000004F
// CHECK:STDOUT:       13:              inst60000050
// CHECK:STDOUT:       14:              inst60000051
// CHECK:STDOUT:       15:              inst60000052
// CHECK:STDOUT:       16:              inst60000053
// CHECK:STDOUT:       17:              inst60000054
// CHECK:STDOUT:       18:              inst60000055
// CHECK:STDOUT:       19:              inst60000056
// CHECK:STDOUT:     inst_block6000000B:
// CHECK:STDOUT:       0:               inst60000022
// CHECK:STDOUT:       1:               inst60000023
// CHECK:STDOUT:       2:               inst60000025
// CHECK:STDOUT:     inst_block6000000C: {}
// CHECK:STDOUT:     inst_block6000000D:
// CHECK:STDOUT:       0:               inst60000039
// CHECK:STDOUT:     inst_block6000000E:
// CHECK:STDOUT:       0:               inst60000039
// CHECK:STDOUT:     inst_block6000000F:
// CHECK:STDOUT:       0:               inst6000003A
// CHECK:STDOUT:     inst_block60000010:
// CHECK:STDOUT:       0:               inst60000038
// CHECK:STDOUT:       1:               inst60000039
// CHECK:STDOUT:     inst_block60000011:
// CHECK:STDOUT:       0:               inst6000003A
// CHECK:STDOUT:       1:               inst60000037
// CHECK:STDOUT:     inst_block60000012: {}
// CHECK:STDOUT:     inst_block60000013:
// CHECK:STDOUT:       0:               inst60000041
// CHECK:STDOUT:     inst_block60000014:
// CHECK:STDOUT:       0:               inst60000041
// CHECK:STDOUT:     inst_block60000015:
// CHECK:STDOUT:       0:               inst60000042
// CHECK:STDOUT:     inst_block60000016:
// CHECK:STDOUT:       0:               inst60000040
// CHECK:STDOUT:       1:               inst60000041
// CHECK:STDOUT:     inst_block60000017:
// CHECK:STDOUT:       0:               inst60000042
// CHECK:STDOUT:       1:               inst6000003E
// CHECK:STDOUT:     inst_block60000018:
// CHECK:STDOUT:       0:               inst60000036
// CHECK:STDOUT:     inst_block60000019:
// CHECK:STDOUT:       0:               inst60000047
// CHECK:STDOUT:     inst_block6000001A:
// CHECK:STDOUT:       0:               inst60000052
// CHECK:STDOUT:     inst_block6000001B:
// CHECK:STDOUT:       0:               inst60000054
// CHECK:STDOUT:     inst_block6000001C:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst60000010
// CHECK:STDOUT:       2:               inst6000001D
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     G
// CHECK:STDOUT:         identifier1:     x
// CHECK:STDOUT:         identifier2:     X
// CHECK:STDOUT:         identifier3:     f
// CHECK:STDOUT:         identifier4:     global
// CHECK:STDOUT:         identifier5:     p
// CHECK:STDOUT:         identifier6:     f__carbon_thunk
// CHECK:STDOUT:       strings:
// CHECK:STDOUT:         string0:         header.h
// CHECK:STDOUT: ...

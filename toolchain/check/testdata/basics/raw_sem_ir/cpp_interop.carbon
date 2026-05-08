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
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc50000000}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc50000001}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc50000002}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc50000003}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc50000004}
// CHECK:STDOUT:   clang_decls:
// CHECK:STDOUT:     clang_decl_id50000000: {key: "namespace Carbon {\n}", inst_id: instF}
// CHECK:STDOUT:     clang_decl_id50000001: {key: "<translation unit>", inst_id: inst50000011}
// CHECK:STDOUT:     clang_decl_id50000002: {key: "struct X {}", inst_id: inst50000013}
// CHECK:STDOUT:     clang_decl_id50000003: {key: "X * _Nonnull p", inst_id: inst50000022}
// CHECK:STDOUT:     clang_decl_id50000004: {key: {decl: "void f(X x = {})", kind: normal, num_params: 0}, inst_id: inst5000002D}
// CHECK:STDOUT:     clang_decl_id50000005: {key: {decl: "inline void f__carbon_thunk()", kind: normal, num_params: 0}, inst_id: inst50000030}
// CHECK:STDOUT:     clang_decl_id50000006: {key: {decl: "void f(X x = {})", kind: normal, num_params: 1}, inst_id: inst5000003B}
// CHECK:STDOUT:     clang_decl_id50000007: {key: {decl: "inline void f__carbon_thunk(X * _Nonnull x)", kind: normal, num_params: 1}, inst_id: inst50000043}
// CHECK:STDOUT:     clang_decl_id50000008: {key: "X * _Nonnull global", inst_id: inst5000004E}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Cpp): inst50000011, name0: inst5000001C}}
// CHECK:STDOUT:     name_scope50000001: {inst: inst50000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name2: inst50000013, name3: inst5000002A, name4: inst5000004E}}
// CHECK:STDOUT:     name_scope50000002: {inst: inst50000013, parent_scope: name_scope50000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name50000000: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000001: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000002: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:     entity_name50000003: {name: name4, parent_scope: name_scope50000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:   cpp_global_vars:
// CHECK:STDOUT:     cpp_global_var50000000: {key: {entity_name_id: entity_name50000003}, clang_decl_id: clang_decl_id50000008}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function50000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block50000007, call_params_id: inst_block50000008, body: [inst_block5000000B]}
// CHECK:STDOUT:     function50000001: {name: name3, parent_scope: name_scope50000001, call_param_patterns_id: inst_block_empty, call_params_id: inst_block_empty}
// CHECK:STDOUT:     function50000002: {name: name6, parent_scope: name_scope50000001, call_param_patterns_id: inst_block_empty, call_params_id: inst_block_empty}
// CHECK:STDOUT:     function50000003: {name: name3, parent_scope: name_scope50000001, call_param_patterns_id: inst_block5000000F, call_params_id: inst_block50000010}
// CHECK:STDOUT:     function50000004: {name: name6, parent_scope: name_scope50000001, call_param_patterns_id: inst_block50000015, call_params_id: inst_block50000016}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class50000000:   {name: name2, parent_scope: name_scope50000001, self_type_id: type(inst50000014), inheritance_kind: Base, is_dynamic: 0, scope_id: name_scope50000002, body_block_id: inst_block5000000C, adapt_id: inst<none>, base_id: inst<none>, complete_type_witness_id: inst50000025, vtable_decl_id: inst<none>}}
// CHECK:STDOUT:   interfaces:      {}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   specific_interfaces: {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:     struct_type_fields50000001:
// CHECK:STDOUT:       0:               {name_id: name5, type_inst_id: inst50000020}
// CHECK:STDOUT:     struct_type_fields50000002:
// CHECK:STDOUT:       0:               {name_id: name5, type_inst_id: inst50000020}
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
// CHECK:STDOUT:     'type(inst5000001D)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000001E)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000020)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000020)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000027)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst50000027)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst50000024)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst50000027)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst50000014)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst50000027)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst50000029)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000002E)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000031)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000003C)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000044)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst5000001E)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
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
// CHECK:STDOUT:     inst50000010:    {kind: ImportCppDecl}
// CHECK:STDOUT:     inst50000011:    {kind: Namespace, arg0: name_scope50000001, arg1: inst50000010, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000012:    {kind: NameRef, arg0: name(Cpp), arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000013:    {kind: ClassDecl, arg0: class50000000, arg1: inst_block<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000014:    {kind: ClassType, arg0: class50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000015:    {kind: NameRef, arg0: name2, arg1: inst50000013, type: type(TypeType)}
// CHECK:STDOUT:     inst50000016:    {kind: PatternType, arg0: inst50000014, type: type(TypeType)}
// CHECK:STDOUT:     inst50000017:    {kind: ValueParamPattern, arg0: name1, type: type(inst50000016)}
// CHECK:STDOUT:     inst50000018:    {kind: ValueBinding, arg0: entity_name50000000, arg1: inst5000001A, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000019:    {kind: WrapperBindingPattern, arg0: entity_name50000000, arg1: inst50000017, type: type(inst50000016)}
// CHECK:STDOUT:     inst5000001A:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst50000014)}
// CHECK:STDOUT:     inst5000001B:    {kind: SpliceBlock, arg0: inst_block50000005, arg1: inst50000015, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001C:    {kind: FunctionDecl, arg0: function50000000, arg1: inst_block5000000A, type: type(inst5000001D)}
// CHECK:STDOUT:     inst5000001D:    {kind: FunctionType, arg0: function50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001E:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001F:    {kind: StructValue, arg0: inst_block_empty, type: type(inst5000001D)}
// CHECK:STDOUT:     inst50000020:    {kind: PointerType, arg0: inst50000014, type: type(TypeType)}
// CHECK:STDOUT:     inst50000021:    {kind: UnboundElementType, arg0: inst50000014, arg1: inst50000020, type: type(TypeType)}
// CHECK:STDOUT:     inst50000022:    {kind: FieldDecl, arg0: name5, arg1: element0, type: type(inst50000021)}
// CHECK:STDOUT:     inst50000023:    {kind: CustomLayoutType, arg0: struct_type_fields50000001, arg1: custom_layout50000001, type: type(TypeType)}
// CHECK:STDOUT:     inst50000024:    {kind: CustomLayoutType, arg0: struct_type_fields50000002, arg1: custom_layout50000001, type: type(TypeType)}
// CHECK:STDOUT:     inst50000025:    {kind: CompleteTypeWitness, arg0: inst50000023, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000026:    {kind: CompleteTypeWitness, arg0: inst50000024, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50000027:    {kind: PointerType, arg0: inst50000024, type: type(TypeType)}
// CHECK:STDOUT:     inst50000028:    {kind: NameRef, arg0: name(Cpp), arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000029:    {kind: CppOverloadSetType, arg0: cpp_overload_set50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002A:    {kind: CppOverloadSetValue, arg0: cpp_overload_set50000000, type: type(inst50000029)}
// CHECK:STDOUT:     inst5000002B:    {kind: CppOverloadSetValue, arg0: cpp_overload_set50000000, type: type(inst50000029)}
// CHECK:STDOUT:     inst5000002C:    {kind: NameRef, arg0: name3, arg1: inst5000002A, type: type(inst50000029)}
// CHECK:STDOUT:     inst5000002D:    {kind: FunctionDecl, arg0: function50000001, arg1: inst_block_empty, type: type(inst5000002E)}
// CHECK:STDOUT:     inst5000002E:    {kind: FunctionType, arg0: function50000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002F:    {kind: StructValue, arg0: inst_block_empty, type: type(inst5000002E)}
// CHECK:STDOUT:     inst50000030:    {kind: FunctionDecl, arg0: function50000002, arg1: inst_block_empty, type: type(inst50000031)}
// CHECK:STDOUT:     inst50000031:    {kind: FunctionType, arg0: function50000002, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000032:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000031)}
// CHECK:STDOUT:     inst50000033:    {kind: Call, arg0: inst50000030, arg1: inst_block_empty, type: type(inst5000001E)}
// CHECK:STDOUT:     inst50000034:    {kind: NameRef, arg0: name(Cpp), arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000035:    {kind: NameRef, arg0: name3, arg1: inst5000002A, type: type(inst50000029)}
// CHECK:STDOUT:     inst50000036:    {kind: NameRef, arg0: name1, arg1: inst50000018, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000037:    {kind: ValueParamPattern, arg0: name1, type: type(inst50000016)}
// CHECK:STDOUT:     inst50000038:    {kind: ValueBinding, arg0: entity_name50000001, arg1: inst5000003A, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000039:    {kind: WrapperBindingPattern, arg0: entity_name50000001, arg1: inst50000037, type: type(inst50000016)}
// CHECK:STDOUT:     inst5000003A:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst50000014)}
// CHECK:STDOUT:     inst5000003B:    {kind: FunctionDecl, arg0: function50000003, arg1: inst_block50000012, type: type(inst5000003C)}
// CHECK:STDOUT:     inst5000003C:    {kind: FunctionType, arg0: function50000003, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003D:    {kind: StructValue, arg0: inst_block_empty, type: type(inst5000003C)}
// CHECK:STDOUT:     inst5000003E:    {kind: PatternType, arg0: inst50000020, type: type(TypeType)}
// CHECK:STDOUT:     inst5000003F:    {kind: ValueParamPattern, arg0: name1, type: type(inst5000003E)}
// CHECK:STDOUT:     inst50000040:    {kind: ValueBinding, arg0: entity_name50000002, arg1: inst50000042, type: type(inst50000020)}
// CHECK:STDOUT:     inst50000041:    {kind: WrapperBindingPattern, arg0: entity_name50000002, arg1: inst5000003F, type: type(inst5000003E)}
// CHECK:STDOUT:     inst50000042:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst50000020)}
// CHECK:STDOUT:     inst50000043:    {kind: FunctionDecl, arg0: function50000004, arg1: inst_block50000018, type: type(inst50000044)}
// CHECK:STDOUT:     inst50000044:    {kind: FunctionType, arg0: function50000004, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst50000045:    {kind: StructValue, arg0: inst_block_empty, type: type(inst50000044)}
// CHECK:STDOUT:     inst50000046:    {kind: ValueAsRef, arg0: inst50000036, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000047:    {kind: AddrOf, arg0: inst50000046, type: type(inst50000020)}
// CHECK:STDOUT:     inst50000048:    {kind: Call, arg0: inst50000043, arg1: inst_block5000001A, type: type(inst5000001E)}
// CHECK:STDOUT:     inst50000049:    {kind: NameRef, arg0: name(Cpp), arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst5000004A:    {kind: NameRef, arg0: name3, arg1: inst5000002A, type: type(inst50000029)}
// CHECK:STDOUT:     inst5000004B:    {kind: NameRef, arg0: name(Cpp), arg1: inst50000011, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst5000004C:    {kind: RefBindingPattern, arg0: entity_name50000003, type: type(inst5000003E)}
// CHECK:STDOUT:     inst5000004D:    {kind: VarPattern, arg0: inst5000004C, type: type(inst5000003E)}
// CHECK:STDOUT:     inst5000004E:    {kind: VarStorage, arg0: inst5000004D, type: type(inst50000020)}
// CHECK:STDOUT:     inst5000004F:    {kind: NameRef, arg0: name4, arg1: inst5000004E, type: type(inst50000020)}
// CHECK:STDOUT:     inst50000050:    {kind: AcquireValue, arg0: inst5000004F, type: type(inst50000020)}
// CHECK:STDOUT:     inst50000051:    {kind: Deref, arg0: inst50000050, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000052:    {kind: AcquireValue, arg0: inst50000051, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000053:    {kind: ValueAsRef, arg0: inst50000052, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000054:    {kind: AddrOf, arg0: inst50000053, type: type(inst50000020)}
// CHECK:STDOUT:     inst50000055:    {kind: Call, arg0: inst50000043, arg1: inst_block5000001C, type: type(inst5000001E)}
// CHECK:STDOUT:     inst50000056:    {kind: Return}
// CHECK:STDOUT:   bundles:         {}
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
// CHECK:STDOUT:       inst50000011:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000012:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000013:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst50000014:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst50000015:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst50000016:    concrete_constant(inst50000016)
// CHECK:STDOUT:       inst50000017:    concrete_constant(inst50000017)
// CHECK:STDOUT:       inst50000019:    concrete_constant(inst50000019)
// CHECK:STDOUT:       inst5000001B:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst5000001C:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst5000001D:    concrete_constant(inst5000001D)
// CHECK:STDOUT:       inst5000001E:    concrete_constant(inst5000001E)
// CHECK:STDOUT:       inst5000001F:    concrete_constant(inst5000001F)
// CHECK:STDOUT:       inst50000020:    concrete_constant(inst50000020)
// CHECK:STDOUT:       inst50000021:    concrete_constant(inst50000021)
// CHECK:STDOUT:       inst50000022:    concrete_constant(inst50000022)
// CHECK:STDOUT:       inst50000023:    concrete_constant(inst50000024)
// CHECK:STDOUT:       inst50000024:    concrete_constant(inst50000024)
// CHECK:STDOUT:       inst50000025:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst50000026:    concrete_constant(inst50000026)
// CHECK:STDOUT:       inst50000027:    concrete_constant(inst50000027)
// CHECK:STDOUT:       inst50000028:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000029:    concrete_constant(inst50000029)
// CHECK:STDOUT:       inst5000002A:    concrete_constant(inst5000002B)
// CHECK:STDOUT:       inst5000002B:    concrete_constant(inst5000002B)
// CHECK:STDOUT:       inst5000002C:    concrete_constant(inst5000002B)
// CHECK:STDOUT:       inst5000002D:    concrete_constant(inst5000002F)
// CHECK:STDOUT:       inst5000002E:    concrete_constant(inst5000002E)
// CHECK:STDOUT:       inst5000002F:    concrete_constant(inst5000002F)
// CHECK:STDOUT:       inst50000030:    concrete_constant(inst50000032)
// CHECK:STDOUT:       inst50000031:    concrete_constant(inst50000031)
// CHECK:STDOUT:       inst50000032:    concrete_constant(inst50000032)
// CHECK:STDOUT:       inst50000034:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst50000035:    concrete_constant(inst5000002B)
// CHECK:STDOUT:       inst50000037:    concrete_constant(inst50000037)
// CHECK:STDOUT:       inst50000039:    concrete_constant(inst50000039)
// CHECK:STDOUT:       inst5000003B:    concrete_constant(inst5000003D)
// CHECK:STDOUT:       inst5000003C:    concrete_constant(inst5000003C)
// CHECK:STDOUT:       inst5000003D:    concrete_constant(inst5000003D)
// CHECK:STDOUT:       inst5000003E:    concrete_constant(inst5000003E)
// CHECK:STDOUT:       inst5000003F:    concrete_constant(inst5000003F)
// CHECK:STDOUT:       inst50000041:    concrete_constant(inst50000041)
// CHECK:STDOUT:       inst50000043:    concrete_constant(inst50000045)
// CHECK:STDOUT:       inst50000044:    concrete_constant(inst50000044)
// CHECK:STDOUT:       inst50000045:    concrete_constant(inst50000045)
// CHECK:STDOUT:       inst50000049:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst5000004A:    concrete_constant(inst5000002B)
// CHECK:STDOUT:       inst5000004B:    concrete_constant(inst50000011)
// CHECK:STDOUT:       inst5000004C:    concrete_constant(inst5000004C)
// CHECK:STDOUT:       inst5000004D:    concrete_constant(inst5000004D)
// CHECK:STDOUT:       inst5000004E:    concrete_constant(inst5000004E)
// CHECK:STDOUT:       inst5000004F:    concrete_constant(inst5000004E)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst5000001C
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst50000011
// CHECK:STDOUT:       1:               inst50000013
// CHECK:STDOUT:       2:               inst5000002A
// CHECK:STDOUT:       3:               inst5000002D
// CHECK:STDOUT:       4:               inst50000030
// CHECK:STDOUT:       5:               inst5000003B
// CHECK:STDOUT:       6:               inst50000043
// CHECK:STDOUT:       7:               inst5000004C
// CHECK:STDOUT:       8:               inst5000004D
// CHECK:STDOUT:       9:               inst5000004E
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block50000005:
// CHECK:STDOUT:       0:               inst50000012
// CHECK:STDOUT:       1:               inst50000015
// CHECK:STDOUT:     inst_block50000006:
// CHECK:STDOUT:       0:               inst50000019
// CHECK:STDOUT:     inst_block50000007:
// CHECK:STDOUT:       0:               inst50000017
// CHECK:STDOUT:     inst_block50000008:
// CHECK:STDOUT:       0:               inst5000001A
// CHECK:STDOUT:     inst_block50000009:
// CHECK:STDOUT:       0:               inst50000017
// CHECK:STDOUT:       1:               inst50000019
// CHECK:STDOUT:     inst_block5000000A:
// CHECK:STDOUT:       0:               inst5000001A
// CHECK:STDOUT:       1:               inst5000001B
// CHECK:STDOUT:       2:               inst50000018
// CHECK:STDOUT:     inst_block5000000B:
// CHECK:STDOUT:       0:               inst50000028
// CHECK:STDOUT:       1:               inst5000002C
// CHECK:STDOUT:       2:               inst50000033
// CHECK:STDOUT:       3:               inst50000034
// CHECK:STDOUT:       4:               inst50000035
// CHECK:STDOUT:       5:               inst50000036
// CHECK:STDOUT:       6:               inst50000046
// CHECK:STDOUT:       7:               inst50000047
// CHECK:STDOUT:       8:               inst50000048
// CHECK:STDOUT:       9:               inst50000049
// CHECK:STDOUT:       10:              inst5000004A
// CHECK:STDOUT:       11:              inst5000004B
// CHECK:STDOUT:       12:              inst5000004F
// CHECK:STDOUT:       13:              inst50000050
// CHECK:STDOUT:       14:              inst50000051
// CHECK:STDOUT:       15:              inst50000052
// CHECK:STDOUT:       16:              inst50000053
// CHECK:STDOUT:       17:              inst50000054
// CHECK:STDOUT:       18:              inst50000055
// CHECK:STDOUT:       19:              inst50000056
// CHECK:STDOUT:     inst_block5000000C:
// CHECK:STDOUT:       0:               inst50000022
// CHECK:STDOUT:       1:               inst50000023
// CHECK:STDOUT:       2:               inst50000025
// CHECK:STDOUT:     inst_block5000000D: {}
// CHECK:STDOUT:     inst_block5000000E:
// CHECK:STDOUT:       0:               inst50000039
// CHECK:STDOUT:     inst_block5000000F:
// CHECK:STDOUT:       0:               inst50000037
// CHECK:STDOUT:     inst_block50000010:
// CHECK:STDOUT:       0:               inst5000003A
// CHECK:STDOUT:     inst_block50000011:
// CHECK:STDOUT:       0:               inst50000037
// CHECK:STDOUT:       1:               inst50000039
// CHECK:STDOUT:     inst_block50000012:
// CHECK:STDOUT:       0:               inst5000003A
// CHECK:STDOUT:       1:               inst50000038
// CHECK:STDOUT:     inst_block50000013: {}
// CHECK:STDOUT:     inst_block50000014:
// CHECK:STDOUT:       0:               inst50000041
// CHECK:STDOUT:     inst_block50000015:
// CHECK:STDOUT:       0:               inst5000003F
// CHECK:STDOUT:     inst_block50000016:
// CHECK:STDOUT:       0:               inst50000042
// CHECK:STDOUT:     inst_block50000017:
// CHECK:STDOUT:       0:               inst5000003F
// CHECK:STDOUT:       1:               inst50000041
// CHECK:STDOUT:     inst_block50000018:
// CHECK:STDOUT:       0:               inst50000042
// CHECK:STDOUT:       1:               inst50000040
// CHECK:STDOUT:     inst_block50000019:
// CHECK:STDOUT:       0:               inst50000036
// CHECK:STDOUT:     inst_block5000001A:
// CHECK:STDOUT:       0:               inst50000047
// CHECK:STDOUT:     inst_block5000001B:
// CHECK:STDOUT:       0:               inst50000052
// CHECK:STDOUT:     inst_block5000001C:
// CHECK:STDOUT:       0:               inst50000054
// CHECK:STDOUT:     inst_block5000001D:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst50000010
// CHECK:STDOUT:       2:               inst5000001C
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

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
  X *p;
};

void f(X x = {}) {
}

// --- import.carbon

import Cpp library "header.h";

fn G(x: Cpp.X) {
  Cpp.f();
  Cpp.f(x);
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        import.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir1, clang_source_loc_id: clang_source_loc0}
// CHECK:STDOUT:   clang_decls:
// CHECK:STDOUT:     clang_decl_id0:  {key: "<translation unit>", inst_id: inst16}
// CHECK:STDOUT:     clang_decl_id1:  {key: "struct X {}", inst_id: inst18}
// CHECK:STDOUT:     clang_decl_id2:  {key: {decl: "void f(X x = {})", num_params: 0}, inst_id: inst41}
// CHECK:STDOUT:     clang_decl_id3:  {key: {decl: "extern void f__carbon_thunk()", num_params: 0}, inst_id: inst44}
// CHECK:STDOUT:     clang_decl_id4:  {key: {decl: "void f(X x = {})", num_params: 1}, inst_id: inst55}
// CHECK:STDOUT:     clang_decl_id5:  {key: {decl: "extern void f__carbon_thunk(X * _Nonnull x)", num_params: 1}, inst_id: inst64}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst16, name1: inst27}}
// CHECK:STDOUT:     name_scope1:     {inst: inst16, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst18, name4: inst38}}
// CHECK:STDOUT:     name_scope2:     {inst: inst18, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name1, parent_scope: name_scope0, call_params_id: inst_block6, body: [inst_block9]}
// CHECK:STDOUT:     function1:       {name: name4, parent_scope: name_scope1, call_params_id: inst_block_empty}
// CHECK:STDOUT:     function2:       {name: name6, parent_scope: name_scope1, call_params_id: inst_block_empty}
// CHECK:STDOUT:     function3:       {name: name4, parent_scope: name_scope1, call_params_id: inst_block13}
// CHECK:STDOUT:     function4:       {name: name6, parent_scope: name_scope1, call_params_id: inst_block18}
// CHECK:STDOUT:   classes:
// CHECK:STDOUT:     class0:          {name: name3, parent_scope: name_scope1, self_type_id: type(inst19), inheritance_kind: Base, is_dynamic: 0, scope_id: name_scope2, body_block_id: inst_block10, adapt_id: inst<none>, base_id: inst<none>, complete_type_witness_id: inst33, vtable_decl_id: inst<none>}}
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
// CHECK:STDOUT:     'type(inst28)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst29)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(inst32)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst35)}
// CHECK:STDOUT:     'type(inst35)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst35)}
// CHECK:STDOUT:     'type(inst19)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst35)}
// CHECK:STDOUT:     'type(inst37)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst42)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst45)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst56)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst65)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst29)}
// CHECK:STDOUT:     'type(inst58)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst58)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: TypeType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(AutoType)':  {kind: AutoType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoolType)':  {kind: BoolType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoundMethodType)': {kind: BoundMethodType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(CharLiteralType)': {kind: CharLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(ErrorInst)': {kind: ErrorInst, type: type(Error)}
// CHECK:STDOUT:     'inst(FloatLiteralType)': {kind: FloatLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(ImplWitnessTablePlaceholder)': {kind: ImplWitnessTablePlaceholder, type: type(TypeType)}
// CHECK:STDOUT:     'inst(InstType)':  {kind: InstType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(IntLiteralType)': {kind: IntLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(NamespaceType)': {kind: NamespaceType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(SpecificFunctionType)': {kind: SpecificFunctionType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(VtableType)': {kind: VtableType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(WitnessType)': {kind: WitnessType, type: type(TypeType)}
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst15:          {kind: ImportCppDecl}
// CHECK:STDOUT:     inst16:          {kind: Namespace, arg0: name_scope1, arg1: inst15, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst17:          {kind: NameRef, arg0: name0, arg1: inst16, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst18:          {kind: ClassDecl, arg0: class0, arg1: inst_block<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst19:          {kind: ClassType, arg0: class0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst20:          {kind: NameRef, arg0: name3, arg1: inst18, type: type(TypeType)}
// CHECK:STDOUT:     inst21:          {kind: BindName, arg0: entity_name0, arg1: inst25, type: type(inst19)}
// CHECK:STDOUT:     inst22:          {kind: PatternType, arg0: inst19, type: type(TypeType)}
// CHECK:STDOUT:     inst23:          {kind: BindingPattern, arg0: entity_name0, type: type(inst22)}
// CHECK:STDOUT:     inst24:          {kind: ValueParamPattern, arg0: inst23, arg1: call_param0, type: type(inst22)}
// CHECK:STDOUT:     inst25:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(inst19)}
// CHECK:STDOUT:     inst26:          {kind: SpliceBlock, arg0: inst_block4, arg1: inst20, type: type(TypeType)}
// CHECK:STDOUT:     inst27:          {kind: FunctionDecl, arg0: function0, arg1: inst_block8, type: type(inst28)}
// CHECK:STDOUT:     inst28:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst29:          {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst30:          {kind: StructValue, arg0: inst_block_empty, type: type(inst28)}
// CHECK:STDOUT:     inst31:          {kind: CustomLayoutType, arg0: struct_type_fields0, arg1: custom_layout1, type: type(TypeType)}
// CHECK:STDOUT:     inst32:          {kind: CustomLayoutType, arg0: struct_type_fields0, arg1: custom_layout1, type: type(TypeType)}
// CHECK:STDOUT:     inst33:          {kind: CompleteTypeWitness, arg0: inst31, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst34:          {kind: CompleteTypeWitness, arg0: inst32, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst35:          {kind: PointerType, arg0: inst32, type: type(TypeType)}
// CHECK:STDOUT:     inst36:          {kind: NameRef, arg0: name0, arg1: inst16, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst37:          {kind: CppOverloadSetType, arg0: cpp_overload_set0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst38:          {kind: CppOverloadSetValue, arg0: cpp_overload_set0, type: type(inst37)}
// CHECK:STDOUT:     inst39:          {kind: StructValue, arg0: inst_block_empty, type: type(inst37)}
// CHECK:STDOUT:     inst40:          {kind: NameRef, arg0: name4, arg1: inst38, type: type(inst37)}
// CHECK:STDOUT:     inst41:          {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(inst42)}
// CHECK:STDOUT:     inst42:          {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst43:          {kind: StructValue, arg0: inst_block_empty, type: type(inst42)}
// CHECK:STDOUT:     inst44:          {kind: FunctionDecl, arg0: function2, arg1: inst_block_empty, type: type(inst45)}
// CHECK:STDOUT:     inst45:          {kind: FunctionType, arg0: function2, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst46:          {kind: StructValue, arg0: inst_block_empty, type: type(inst45)}
// CHECK:STDOUT:     inst47:          {kind: Call, arg0: inst44, arg1: inst_block_empty, type: type(inst29)}
// CHECK:STDOUT:     inst48:          {kind: NameRef, arg0: name0, arg1: inst16, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst49:          {kind: NameRef, arg0: name4, arg1: inst38, type: type(inst37)}
// CHECK:STDOUT:     inst50:          {kind: NameRef, arg0: name2, arg1: inst21, type: type(inst19)}
// CHECK:STDOUT:     inst51:          {kind: BindName, arg0: entity_name1, arg1: inst54, type: type(inst19)}
// CHECK:STDOUT:     inst52:          {kind: BindingPattern, arg0: entity_name1, type: type(inst22)}
// CHECK:STDOUT:     inst53:          {kind: ValueParamPattern, arg0: inst52, arg1: call_param0, type: type(inst22)}
// CHECK:STDOUT:     inst54:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(inst19)}
// CHECK:STDOUT:     inst55:          {kind: FunctionDecl, arg0: function3, arg1: inst_block15, type: type(inst56)}
// CHECK:STDOUT:     inst56:          {kind: FunctionType, arg0: function3, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst57:          {kind: StructValue, arg0: inst_block_empty, type: type(inst56)}
// CHECK:STDOUT:     inst58:          {kind: PointerType, arg0: inst19, type: type(TypeType)}
// CHECK:STDOUT:     inst59:          {kind: BindName, arg0: entity_name2, arg1: inst63, type: type(inst58)}
// CHECK:STDOUT:     inst60:          {kind: PatternType, arg0: inst58, type: type(TypeType)}
// CHECK:STDOUT:     inst61:          {kind: BindingPattern, arg0: entity_name2, type: type(inst60)}
// CHECK:STDOUT:     inst62:          {kind: ValueParamPattern, arg0: inst61, arg1: call_param0, type: type(inst60)}
// CHECK:STDOUT:     inst63:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(inst58)}
// CHECK:STDOUT:     inst64:          {kind: FunctionDecl, arg0: function4, arg1: inst_block20, type: type(inst65)}
// CHECK:STDOUT:     inst65:          {kind: FunctionType, arg0: function4, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst66:          {kind: StructValue, arg0: inst_block_empty, type: type(inst65)}
// CHECK:STDOUT:     inst67:          {kind: ValueAsRef, arg0: inst50, type: type(inst19)}
// CHECK:STDOUT:     inst68:          {kind: AddrOf, arg0: inst67, type: type(inst58)}
// CHECK:STDOUT:     inst69:          {kind: Call, arg0: inst64, arg1: inst_block22, type: type(inst29)}
// CHECK:STDOUT:     inst70:          {kind: Return}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(AutoType)':  concrete_constant(inst(AutoType))
// CHECK:STDOUT:       'inst(BoolType)':  concrete_constant(inst(BoolType))
// CHECK:STDOUT:       'inst(BoundMethodType)': concrete_constant(inst(BoundMethodType))
// CHECK:STDOUT:       'inst(CharLiteralType)': concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       'inst(ErrorInst)': concrete_constant(inst(ErrorInst))
// CHECK:STDOUT:       'inst(FloatLiteralType)': concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       'inst(ImplWitnessTablePlaceholder)': concrete_constant(inst(ImplWitnessTablePlaceholder))
// CHECK:STDOUT:       'inst(InstType)':  concrete_constant(inst(InstType))
// CHECK:STDOUT:       'inst(IntLiteralType)': concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       'inst(NamespaceType)': concrete_constant(inst(NamespaceType))
// CHECK:STDOUT:       'inst(SpecificFunctionType)': concrete_constant(inst(SpecificFunctionType))
// CHECK:STDOUT:       'inst(VtableType)': concrete_constant(inst(VtableType))
// CHECK:STDOUT:       'inst(WitnessType)': concrete_constant(inst(WitnessType))
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst16:          concrete_constant(inst16)
// CHECK:STDOUT:       inst17:          concrete_constant(inst16)
// CHECK:STDOUT:       inst18:          concrete_constant(inst19)
// CHECK:STDOUT:       inst19:          concrete_constant(inst19)
// CHECK:STDOUT:       inst20:          concrete_constant(inst19)
// CHECK:STDOUT:       inst22:          concrete_constant(inst22)
// CHECK:STDOUT:       inst23:          concrete_constant(inst23)
// CHECK:STDOUT:       inst24:          concrete_constant(inst24)
// CHECK:STDOUT:       inst26:          concrete_constant(inst19)
// CHECK:STDOUT:       inst27:          concrete_constant(inst30)
// CHECK:STDOUT:       inst28:          concrete_constant(inst28)
// CHECK:STDOUT:       inst29:          concrete_constant(inst29)
// CHECK:STDOUT:       inst30:          concrete_constant(inst30)
// CHECK:STDOUT:       inst31:          concrete_constant(inst32)
// CHECK:STDOUT:       inst32:          concrete_constant(inst32)
// CHECK:STDOUT:       inst33:          concrete_constant(inst34)
// CHECK:STDOUT:       inst34:          concrete_constant(inst34)
// CHECK:STDOUT:       inst35:          concrete_constant(inst35)
// CHECK:STDOUT:       inst36:          concrete_constant(inst16)
// CHECK:STDOUT:       inst37:          concrete_constant(inst37)
// CHECK:STDOUT:       inst38:          concrete_constant(inst39)
// CHECK:STDOUT:       inst39:          concrete_constant(inst39)
// CHECK:STDOUT:       inst40:          concrete_constant(inst39)
// CHECK:STDOUT:       inst41:          concrete_constant(inst43)
// CHECK:STDOUT:       inst42:          concrete_constant(inst42)
// CHECK:STDOUT:       inst43:          concrete_constant(inst43)
// CHECK:STDOUT:       inst44:          concrete_constant(inst46)
// CHECK:STDOUT:       inst45:          concrete_constant(inst45)
// CHECK:STDOUT:       inst46:          concrete_constant(inst46)
// CHECK:STDOUT:       inst48:          concrete_constant(inst16)
// CHECK:STDOUT:       inst49:          concrete_constant(inst39)
// CHECK:STDOUT:       inst52:          concrete_constant(inst52)
// CHECK:STDOUT:       inst53:          concrete_constant(inst53)
// CHECK:STDOUT:       inst55:          concrete_constant(inst57)
// CHECK:STDOUT:       inst56:          concrete_constant(inst56)
// CHECK:STDOUT:       inst57:          concrete_constant(inst57)
// CHECK:STDOUT:       inst58:          concrete_constant(inst58)
// CHECK:STDOUT:       inst60:          concrete_constant(inst60)
// CHECK:STDOUT:       inst61:          concrete_constant(inst61)
// CHECK:STDOUT:       inst62:          concrete_constant(inst62)
// CHECK:STDOUT:       inst64:          concrete_constant(inst66)
// CHECK:STDOUT:       inst65:          concrete_constant(inst65)
// CHECK:STDOUT:       inst66:          concrete_constant(inst66)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst27
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:       1:               inst18
// CHECK:STDOUT:       2:               inst38
// CHECK:STDOUT:       3:               inst41
// CHECK:STDOUT:       4:               inst44
// CHECK:STDOUT:       5:               inst55
// CHECK:STDOUT:       6:               inst64
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:       1:               inst20
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst24
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst25
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst23
// CHECK:STDOUT:       1:               inst24
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst25
// CHECK:STDOUT:       1:               inst26
// CHECK:STDOUT:       2:               inst21
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst36
// CHECK:STDOUT:       1:               inst40
// CHECK:STDOUT:       2:               inst47
// CHECK:STDOUT:       3:               inst48
// CHECK:STDOUT:       4:               inst49
// CHECK:STDOUT:       5:               inst50
// CHECK:STDOUT:       6:               inst67
// CHECK:STDOUT:       7:               inst68
// CHECK:STDOUT:       8:               inst69
// CHECK:STDOUT:       9:               inst70
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst31
// CHECK:STDOUT:       1:               inst33
// CHECK:STDOUT:     inst_block11:    {}
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst53
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst54
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst52
// CHECK:STDOUT:       1:               inst53
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst54
// CHECK:STDOUT:       1:               inst51
// CHECK:STDOUT:     inst_block16:    {}
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst62
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst61
// CHECK:STDOUT:       1:               inst62
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:       1:               inst59
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst50
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst68
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst15
// CHECK:STDOUT:       2:               inst27
// CHECK:STDOUT: ...

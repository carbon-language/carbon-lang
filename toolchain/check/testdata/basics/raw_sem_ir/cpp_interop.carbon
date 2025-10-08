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
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir(Cpp), clang_source_loc_id: clang_source_loc0}
// CHECK:STDOUT:   clang_decls:
// CHECK:STDOUT:     clang_decl_id0:  {key: "<translation unit>", inst_id: ir0.inst16}
// CHECK:STDOUT:     clang_decl_id1:  {key: "struct X {}", inst_id: ir0.inst18}
// CHECK:STDOUT:     clang_decl_id2:  {key: {decl: "void f(X x = {})", num_params: 0}, inst_id: ir0.inst41}
// CHECK:STDOUT:     clang_decl_id3:  {key: {decl: "extern void f__carbon_thunk()", num_params: 0}, inst_id: ir0.inst44}
// CHECK:STDOUT:     clang_decl_id4:  {key: {decl: "void f(X x = {})", num_params: 1}, inst_id: ir0.inst55}
// CHECK:STDOUT:     clang_decl_id5:  {key: {decl: "extern void f__carbon_thunk(X * _Nonnull x)", num_params: 1}, inst_id: ir0.inst64}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: ir0.inst16, name1: ir0.inst27}}
// CHECK:STDOUT:     name_scope1:     {inst: ir0.inst16, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: ir0.inst18, name4: ir0.inst38}}
// CHECK:STDOUT:     name_scope2:     {inst: ir0.inst18, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
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
// CHECK:STDOUT:     class0:          {name: name3, parent_scope: name_scope1, self_type_id: type(ir0.inst19), inheritance_kind: Base, is_dynamic: 0, scope_id: name_scope2, body_block_id: inst_block10, adapt_id: inst<none>, base_id: inst<none>, complete_type_witness_id: ir0.inst33, vtable_decl_id: inst<none>}}
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
// CHECK:STDOUT:     'type(ir0.inst28)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst29)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(ir0.inst32)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(ir0.inst35)}
// CHECK:STDOUT:     'type(ir0.inst35)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(ir0.inst35)}
// CHECK:STDOUT:     'type(ir0.inst19)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(ir0.inst35)}
// CHECK:STDOUT:     'type(ir0.inst37)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst42)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst45)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst56)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst65)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst58)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(ir0.inst58)}
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
// CHECK:STDOUT:     ir0.inst15:      {kind: ImportCppDecl}
// CHECK:STDOUT:     ir0.inst16:      {kind: Namespace, arg0: name_scope1, arg1: ir0.inst15, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst17:      {kind: NameRef, arg0: name0, arg1: ir0.inst16, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst18:      {kind: ClassDecl, arg0: class0, arg1: inst_block<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst19:      {kind: ClassType, arg0: class0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst20:      {kind: NameRef, arg0: name3, arg1: ir0.inst18, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst21:      {kind: BindName, arg0: entity_name0, arg1: ir0.inst25, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst22:      {kind: PatternType, arg0: ir0.inst19, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst23:      {kind: BindingPattern, arg0: entity_name0, type: type(ir0.inst22)}
// CHECK:STDOUT:     ir0.inst24:      {kind: ValueParamPattern, arg0: ir0.inst23, arg1: call_param0, type: type(ir0.inst22)}
// CHECK:STDOUT:     ir0.inst25:      {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst26:      {kind: SpliceBlock, arg0: inst_block4, arg1: ir0.inst20, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst27:      {kind: FunctionDecl, arg0: function0, arg1: inst_block8, type: type(ir0.inst28)}
// CHECK:STDOUT:     ir0.inst28:      {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst29:      {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst30:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst28)}
// CHECK:STDOUT:     ir0.inst31:      {kind: CustomLayoutType, arg0: struct_type_fields0, arg1: custom_layout1, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst32:      {kind: CustomLayoutType, arg0: struct_type_fields0, arg1: custom_layout1, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst33:      {kind: CompleteTypeWitness, arg0: ir0.inst31, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst34:      {kind: CompleteTypeWitness, arg0: ir0.inst32, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst35:      {kind: PointerType, arg0: ir0.inst32, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst36:      {kind: NameRef, arg0: name0, arg1: ir0.inst16, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst37:      {kind: CppOverloadSetType, arg0: cpp_overload_set0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst38:      {kind: CppOverloadSetValue, arg0: cpp_overload_set0, type: type(ir0.inst37)}
// CHECK:STDOUT:     ir0.inst39:      {kind: CppOverloadSetValue, arg0: cpp_overload_set0, type: type(ir0.inst37)}
// CHECK:STDOUT:     ir0.inst40:      {kind: NameRef, arg0: name4, arg1: ir0.inst38, type: type(ir0.inst37)}
// CHECK:STDOUT:     ir0.inst41:      {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(ir0.inst42)}
// CHECK:STDOUT:     ir0.inst42:      {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst43:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst42)}
// CHECK:STDOUT:     ir0.inst44:      {kind: FunctionDecl, arg0: function2, arg1: inst_block_empty, type: type(ir0.inst45)}
// CHECK:STDOUT:     ir0.inst45:      {kind: FunctionType, arg0: function2, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst46:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst45)}
// CHECK:STDOUT:     ir0.inst47:      {kind: Call, arg0: ir0.inst44, arg1: inst_block_empty, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst48:      {kind: NameRef, arg0: name0, arg1: ir0.inst16, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst49:      {kind: NameRef, arg0: name4, arg1: ir0.inst38, type: type(ir0.inst37)}
// CHECK:STDOUT:     ir0.inst50:      {kind: NameRef, arg0: name2, arg1: ir0.inst21, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst51:      {kind: BindName, arg0: entity_name1, arg1: ir0.inst54, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst52:      {kind: BindingPattern, arg0: entity_name1, type: type(ir0.inst22)}
// CHECK:STDOUT:     ir0.inst53:      {kind: ValueParamPattern, arg0: ir0.inst52, arg1: call_param0, type: type(ir0.inst22)}
// CHECK:STDOUT:     ir0.inst54:      {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst55:      {kind: FunctionDecl, arg0: function3, arg1: inst_block15, type: type(ir0.inst56)}
// CHECK:STDOUT:     ir0.inst56:      {kind: FunctionType, arg0: function3, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst57:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst56)}
// CHECK:STDOUT:     ir0.inst58:      {kind: PointerType, arg0: ir0.inst19, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst59:      {kind: BindName, arg0: entity_name2, arg1: ir0.inst63, type: type(ir0.inst58)}
// CHECK:STDOUT:     ir0.inst60:      {kind: PatternType, arg0: ir0.inst58, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst61:      {kind: BindingPattern, arg0: entity_name2, type: type(ir0.inst60)}
// CHECK:STDOUT:     ir0.inst62:      {kind: ValueParamPattern, arg0: ir0.inst61, arg1: call_param0, type: type(ir0.inst60)}
// CHECK:STDOUT:     ir0.inst63:      {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(ir0.inst58)}
// CHECK:STDOUT:     ir0.inst64:      {kind: FunctionDecl, arg0: function4, arg1: inst_block20, type: type(ir0.inst65)}
// CHECK:STDOUT:     ir0.inst65:      {kind: FunctionType, arg0: function4, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst66:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst65)}
// CHECK:STDOUT:     ir0.inst67:      {kind: ValueAsRef, arg0: ir0.inst50, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst68:      {kind: AddrOf, arg0: ir0.inst67, type: type(ir0.inst58)}
// CHECK:STDOUT:     ir0.inst69:      {kind: Call, arg0: ir0.inst64, arg1: inst_block22, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst70:      {kind: Return}
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
// CHECK:STDOUT:       ir0.inst16:      concrete_constant(ir0.inst16)
// CHECK:STDOUT:       ir0.inst17:      concrete_constant(ir0.inst16)
// CHECK:STDOUT:       ir0.inst18:      concrete_constant(ir0.inst19)
// CHECK:STDOUT:       ir0.inst19:      concrete_constant(ir0.inst19)
// CHECK:STDOUT:       ir0.inst20:      concrete_constant(ir0.inst19)
// CHECK:STDOUT:       ir0.inst22:      concrete_constant(ir0.inst22)
// CHECK:STDOUT:       ir0.inst23:      concrete_constant(ir0.inst23)
// CHECK:STDOUT:       ir0.inst24:      concrete_constant(ir0.inst24)
// CHECK:STDOUT:       ir0.inst26:      concrete_constant(ir0.inst19)
// CHECK:STDOUT:       ir0.inst27:      concrete_constant(ir0.inst30)
// CHECK:STDOUT:       ir0.inst28:      concrete_constant(ir0.inst28)
// CHECK:STDOUT:       ir0.inst29:      concrete_constant(ir0.inst29)
// CHECK:STDOUT:       ir0.inst30:      concrete_constant(ir0.inst30)
// CHECK:STDOUT:       ir0.inst31:      concrete_constant(ir0.inst32)
// CHECK:STDOUT:       ir0.inst32:      concrete_constant(ir0.inst32)
// CHECK:STDOUT:       ir0.inst33:      concrete_constant(ir0.inst34)
// CHECK:STDOUT:       ir0.inst34:      concrete_constant(ir0.inst34)
// CHECK:STDOUT:       ir0.inst35:      concrete_constant(ir0.inst35)
// CHECK:STDOUT:       ir0.inst36:      concrete_constant(ir0.inst16)
// CHECK:STDOUT:       ir0.inst37:      concrete_constant(ir0.inst37)
// CHECK:STDOUT:       ir0.inst38:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst39:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst40:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst41:      concrete_constant(ir0.inst43)
// CHECK:STDOUT:       ir0.inst42:      concrete_constant(ir0.inst42)
// CHECK:STDOUT:       ir0.inst43:      concrete_constant(ir0.inst43)
// CHECK:STDOUT:       ir0.inst44:      concrete_constant(ir0.inst46)
// CHECK:STDOUT:       ir0.inst45:      concrete_constant(ir0.inst45)
// CHECK:STDOUT:       ir0.inst46:      concrete_constant(ir0.inst46)
// CHECK:STDOUT:       ir0.inst48:      concrete_constant(ir0.inst16)
// CHECK:STDOUT:       ir0.inst49:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst52:      concrete_constant(ir0.inst52)
// CHECK:STDOUT:       ir0.inst53:      concrete_constant(ir0.inst53)
// CHECK:STDOUT:       ir0.inst55:      concrete_constant(ir0.inst57)
// CHECK:STDOUT:       ir0.inst56:      concrete_constant(ir0.inst56)
// CHECK:STDOUT:       ir0.inst57:      concrete_constant(ir0.inst57)
// CHECK:STDOUT:       ir0.inst58:      concrete_constant(ir0.inst58)
// CHECK:STDOUT:       ir0.inst60:      concrete_constant(ir0.inst60)
// CHECK:STDOUT:       ir0.inst61:      concrete_constant(ir0.inst61)
// CHECK:STDOUT:       ir0.inst62:      concrete_constant(ir0.inst62)
// CHECK:STDOUT:       ir0.inst64:      concrete_constant(ir0.inst66)
// CHECK:STDOUT:       ir0.inst65:      concrete_constant(ir0.inst65)
// CHECK:STDOUT:       ir0.inst66:      concrete_constant(ir0.inst66)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               ir0.inst27
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               ir0.inst16
// CHECK:STDOUT:       1:               ir0.inst18
// CHECK:STDOUT:       2:               ir0.inst38
// CHECK:STDOUT:       3:               ir0.inst41
// CHECK:STDOUT:       4:               ir0.inst44
// CHECK:STDOUT:       5:               ir0.inst55
// CHECK:STDOUT:       6:               ir0.inst64
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               ir0.inst17
// CHECK:STDOUT:       1:               ir0.inst20
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               ir0.inst24
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               ir0.inst25
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               ir0.inst23
// CHECK:STDOUT:       1:               ir0.inst24
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               ir0.inst25
// CHECK:STDOUT:       1:               ir0.inst26
// CHECK:STDOUT:       2:               ir0.inst21
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               ir0.inst36
// CHECK:STDOUT:       1:               ir0.inst40
// CHECK:STDOUT:       2:               ir0.inst47
// CHECK:STDOUT:       3:               ir0.inst48
// CHECK:STDOUT:       4:               ir0.inst49
// CHECK:STDOUT:       5:               ir0.inst50
// CHECK:STDOUT:       6:               ir0.inst67
// CHECK:STDOUT:       7:               ir0.inst68
// CHECK:STDOUT:       8:               ir0.inst69
// CHECK:STDOUT:       9:               ir0.inst70
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               ir0.inst31
// CHECK:STDOUT:       1:               ir0.inst33
// CHECK:STDOUT:     inst_block11:    {}
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               ir0.inst53
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               ir0.inst54
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               ir0.inst52
// CHECK:STDOUT:       1:               ir0.inst53
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               ir0.inst54
// CHECK:STDOUT:       1:               ir0.inst51
// CHECK:STDOUT:     inst_block16:    {}
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               ir0.inst62
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               ir0.inst63
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               ir0.inst61
// CHECK:STDOUT:       1:               ir0.inst62
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               ir0.inst63
// CHECK:STDOUT:       1:               ir0.inst59
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               ir0.inst50
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               ir0.inst68
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               ir0.inst15
// CHECK:STDOUT:       2:               ir0.inst27
// CHECK:STDOUT: ...

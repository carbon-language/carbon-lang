// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --builtin-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/builtins.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/builtins.carbon

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        builtins.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope00000000: {inst: inst0000000E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:    {}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields00000000: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: TypeType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(AutoType)':  {kind: AutoType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoolType)':  {kind: BoolType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoundMethodType)': {kind: BoundMethodType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(CharLiteralType)': {kind: CharLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(CppVoidType)': {kind: CppVoidType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(ErrorInst)': {kind: ErrorInst, type: type(Error)}
// CHECK:STDOUT:     'inst(FloatLiteralType)': {kind: FloatLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(InstType)':  {kind: InstType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(IntLiteralType)': {kind: IntLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(NamespaceType)': {kind: NamespaceType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(SpecificFunctionType)': {kind: SpecificFunctionType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(VtableType)': {kind: VtableType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(WitnessType)': {kind: WitnessType, type: type(TypeType)}
// CHECK:STDOUT:     inst0000000E:    {kind: Namespace, arg0: name_scope00000000, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(AutoType)':  concrete_constant(inst(AutoType))
// CHECK:STDOUT:       'inst(BoolType)':  concrete_constant(inst(BoolType))
// CHECK:STDOUT:       'inst(BoundMethodType)': concrete_constant(inst(BoundMethodType))
// CHECK:STDOUT:       'inst(CharLiteralType)': concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       'inst(CppVoidType)': concrete_constant(inst(CppVoidType))
// CHECK:STDOUT:       'inst(ErrorInst)': concrete_constant(inst(ErrorInst))
// CHECK:STDOUT:       'inst(FloatLiteralType)': concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       'inst(InstType)':  concrete_constant(inst(InstType))
// CHECK:STDOUT:       'inst(IntLiteralType)': concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       'inst(NamespaceType)': concrete_constant(inst(NamespaceType))
// CHECK:STDOUT:       'inst(SpecificFunctionType)': concrete_constant(inst(SpecificFunctionType))
// CHECK:STDOUT:       'inst(VtableType)': concrete_constant(inst(VtableType))
// CHECK:STDOUT:       'inst(WitnessType)': concrete_constant(inst(WitnessType))
// CHECK:STDOUT:       inst0000000E:    concrete_constant(inst0000000E)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block60000004:
// CHECK:STDOUT:       0:               inst0000000E
// CHECK:STDOUT: ...

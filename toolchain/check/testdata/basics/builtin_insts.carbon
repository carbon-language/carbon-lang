// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-prelude-import --phase=check --dump-raw-sem-ir --builtin-sem-ir %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/builtin_insts.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/builtin_insts.carbon

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        builtin_insts.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst12, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:    {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
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
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: TypeType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(AutoType)':  {kind: AutoType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoolType)':  {kind: BoolType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(BoundMethodType)': {kind: BoundMethodType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(ErrorInst)': {kind: ErrorInst, type: type(Error)}
// CHECK:STDOUT:     'inst(IntLiteralType)': {kind: IntLiteralType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(LegacyFloatType)': {kind: LegacyFloatType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(NamespaceType)': {kind: NamespaceType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(SpecificFunctionType)': {kind: SpecificFunctionType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(StringType)': {kind: StringType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(VtableType)': {kind: VtableType, type: type(TypeType)}
// CHECK:STDOUT:     'inst(WitnessType)': {kind: WitnessType, type: type(TypeType)}
// CHECK:STDOUT:     inst12:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:     'inst(AutoType)':  concrete_constant(inst(AutoType))
// CHECK:STDOUT:     'inst(BoolType)':  concrete_constant(inst(BoolType))
// CHECK:STDOUT:     'inst(BoundMethodType)': concrete_constant(inst(BoundMethodType))
// CHECK:STDOUT:     'inst(ErrorInst)': concrete_constant(inst(ErrorInst))
// CHECK:STDOUT:     'inst(IntLiteralType)': concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:     'inst(LegacyFloatType)': concrete_constant(inst(LegacyFloatType))
// CHECK:STDOUT:     'inst(NamespaceType)': concrete_constant(inst(NamespaceType))
// CHECK:STDOUT:     'inst(SpecificFunctionType)': concrete_constant(inst(SpecificFunctionType))
// CHECK:STDOUT:     'inst(StringType)': concrete_constant(inst(StringType))
// CHECK:STDOUT:     'inst(VtableType)': concrete_constant(inst(VtableType))
// CHECK:STDOUT:     'inst(WitnessType)': concrete_constant(inst(WitnessType))
// CHECK:STDOUT:     inst12:          concrete_constant(inst12)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst12
// CHECK:STDOUT: ...

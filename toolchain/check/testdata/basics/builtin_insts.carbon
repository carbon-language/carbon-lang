// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-raw-sem-ir --builtin-sem-ir %s
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
// CHECK:STDOUT:     ir0:             {decl_id: inst<invalid>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:    {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instTypeType:    {kind: TypeType, type: typeTypeType}
// CHECK:STDOUT:     instErrorInst:   {kind: ErrorInst, type: typeError}
// CHECK:STDOUT:     instAutoType:    {kind: AutoType, type: typeTypeType}
// CHECK:STDOUT:     instBoolType:    {kind: BoolType, type: typeTypeType}
// CHECK:STDOUT:     instIntLiteralType: {kind: IntLiteralType, type: typeTypeType}
// CHECK:STDOUT:     instLegacyFloatType: {kind: LegacyFloatType, type: typeTypeType}
// CHECK:STDOUT:     instStringType:  {kind: StringType, type: typeTypeType}
// CHECK:STDOUT:     instBoundMethodType: {kind: BoundMethodType, type: typeTypeType}
// CHECK:STDOUT:     instSpecificFunctionType: {kind: SpecificFunctionType, type: typeTypeType}
// CHECK:STDOUT:     instNamespaceType: {kind: NamespaceType, type: typeTypeType}
// CHECK:STDOUT:     instWitnessType: {kind: WitnessType, type: typeTypeType}
// CHECK:STDOUT:     instVtableType:  {kind: VtableType, type: typeTypeType}
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     instTypeType:    templateConstant(instTypeType)
// CHECK:STDOUT:     instErrorInst:   templateConstant(instErrorInst)
// CHECK:STDOUT:     instAutoType:    templateConstant(instAutoType)
// CHECK:STDOUT:     instBoolType:    templateConstant(instBoolType)
// CHECK:STDOUT:     instIntLiteralType: templateConstant(instIntLiteralType)
// CHECK:STDOUT:     instLegacyFloatType: templateConstant(instLegacyFloatType)
// CHECK:STDOUT:     instStringType:  templateConstant(instStringType)
// CHECK:STDOUT:     instBoundMethodType: templateConstant(instBoundMethodType)
// CHECK:STDOUT:     instSpecificFunctionType: templateConstant(instSpecificFunctionType)
// CHECK:STDOUT:     instNamespaceType: templateConstant(instNamespaceType)
// CHECK:STDOUT:     instWitnessType: templateConstant(instWitnessType)
// CHECK:STDOUT:     instVtableType:  templateConstant(instVtableType)
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT: ...

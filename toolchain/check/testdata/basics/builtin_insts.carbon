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
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   generic_instances: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     type7:           {kind: copy, type: type7}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instTypeType:    {kind: BuiltinInst, arg0: TypeType, type: typeTypeType}
// CHECK:STDOUT:     instError:       {kind: BuiltinInst, arg0: Error, type: typeError}
// CHECK:STDOUT:     instBoolType:    {kind: BuiltinInst, arg0: BoolType, type: typeTypeType}
// CHECK:STDOUT:     instIntType:     {kind: BuiltinInst, arg0: IntType, type: typeTypeType}
// CHECK:STDOUT:     instFloatType:   {kind: BuiltinInst, arg0: FloatType, type: typeTypeType}
// CHECK:STDOUT:     instStringType:  {kind: BuiltinInst, arg0: StringType, type: typeTypeType}
// CHECK:STDOUT:     instBoundMethodType: {kind: BuiltinInst, arg0: BoundMethodType, type: typeTypeType}
// CHECK:STDOUT:     instNamespaceType: {kind: BuiltinInst, arg0: NamespaceType, type: typeTypeType}
// CHECK:STDOUT:     instWitnessType: {kind: BuiltinInst, arg0: WitnessType, type: typeTypeType}
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type7}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     instTypeType:    template instTypeType
// CHECK:STDOUT:     instError:       template instError
// CHECK:STDOUT:     instBoolType:    template instBoolType
// CHECK:STDOUT:     instIntType:     template instIntType
// CHECK:STDOUT:     instFloatType:   template instFloatType
// CHECK:STDOUT:     instStringType:  template instStringType
// CHECK:STDOUT:     instBoundMethodType: template instBoundMethodType
// CHECK:STDOUT:     instNamespaceType: template instNamespaceType
// CHECK:STDOUT:     instWitnessType: template instWitnessType
// CHECK:STDOUT:     'inst+0':          template inst+0
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT: ...

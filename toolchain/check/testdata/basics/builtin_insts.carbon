// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --disable-prelude-import --phase=check --dump-raw-sem-ir --builtin-sem-ir %s
//
// AUTOUPDATE

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        builtin_insts.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 2
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instTypeType:    {kind: ImportRefLoaded, arg0: 0, type: typeTypeType}
// CHECK:STDOUT:     instError:       {kind: ImportRefLoaded, arg0: 1, type: typeError}
// CHECK:STDOUT:     instBoolType:    {kind: ImportRefLoaded, arg0: 2, type: typeTypeType}
// CHECK:STDOUT:     instIntType:     {kind: ImportRefLoaded, arg0: 3, type: typeTypeType}
// CHECK:STDOUT:     instFloatType:   {kind: ImportRefLoaded, arg0: 4, type: typeTypeType}
// CHECK:STDOUT:     instStringType:  {kind: ImportRefLoaded, arg0: 5, type: typeTypeType}
// CHECK:STDOUT:     instFunctionType: {kind: ImportRefLoaded, arg0: 6, type: typeTypeType}
// CHECK:STDOUT:     instBoundMethodType: {kind: ImportRefLoaded, arg0: 7, type: typeTypeType}
// CHECK:STDOUT:     instNamespaceType: {kind: ImportRefLoaded, arg0: 8, type: typeTypeType}
// CHECK:STDOUT:     instWitnessType: {kind: ImportRefLoaded, arg0: 9, type: typeTypeType}
// CHECK:STDOUT:     inst+0:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     instTypeType:    template instTypeType
// CHECK:STDOUT:     instError:       template instError
// CHECK:STDOUT:     instBoolType:    template instBoolType
// CHECK:STDOUT:     instIntType:     template instIntType
// CHECK:STDOUT:     instFloatType:   template instFloatType
// CHECK:STDOUT:     instStringType:  template instStringType
// CHECK:STDOUT:     instFunctionType: template instFunctionType
// CHECK:STDOUT:     instBoundMethodType: template instBoundMethodType
// CHECK:STDOUT:     instNamespaceType: template instNamespaceType
// CHECK:STDOUT:     instWitnessType: template instWitnessType
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT: ...

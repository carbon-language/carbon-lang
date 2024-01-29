// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-raw-sem-ir --builtin-sem-ir %s
//
// AUTOUPDATE

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        builtin_insts.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   bind_names:      {}
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instTypeType:    {kind: ImportRefUsed, arg0: ir0, arg1: instTypeType, type: typeTypeType}
// CHECK:STDOUT:     instError:       {kind: ImportRefUsed, arg0: ir0, arg1: instError, type: typeError}
// CHECK:STDOUT:     instBoolType:    {kind: ImportRefUsed, arg0: ir0, arg1: instBoolType, type: typeTypeType}
// CHECK:STDOUT:     instIntType:     {kind: ImportRefUsed, arg0: ir0, arg1: instIntType, type: typeTypeType}
// CHECK:STDOUT:     instFloatType:   {kind: ImportRefUsed, arg0: ir0, arg1: instFloatType, type: typeTypeType}
// CHECK:STDOUT:     instStringType:  {kind: ImportRefUsed, arg0: ir0, arg1: instStringType, type: typeTypeType}
// CHECK:STDOUT:     instFunctionType: {kind: ImportRefUsed, arg0: ir0, arg1: instFunctionType, type: typeTypeType}
// CHECK:STDOUT:     instBoundMethodType: {kind: ImportRefUsed, arg0: ir0, arg1: instBoundMethodType, type: typeTypeType}
// CHECK:STDOUT:     instNamespaceType: {kind: ImportRefUsed, arg0: ir0, arg1: instNamespaceType, type: typeTypeType}
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
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:         {}
// CHECK:STDOUT:     block2:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT: ...

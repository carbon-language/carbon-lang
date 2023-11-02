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
// CHECK:STDOUT:   cross_reference_irs_size: 1
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:           {}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instTypeType:    {kind: CrossReference, arg0: ir0, arg1: instTypeType, type: typeTypeType}
// CHECK:STDOUT:     instError:       {kind: CrossReference, arg0: ir0, arg1: instError, type: typeError}
// CHECK:STDOUT:     instBoolType:    {kind: CrossReference, arg0: ir0, arg1: instBoolType, type: typeTypeType}
// CHECK:STDOUT:     instIntegerType: {kind: CrossReference, arg0: ir0, arg1: instIntegerType, type: typeTypeType}
// CHECK:STDOUT:     instFloatingPointType: {kind: CrossReference, arg0: ir0, arg1: instFloatingPointType, type: typeTypeType}
// CHECK:STDOUT:     instStringType:  {kind: CrossReference, arg0: ir0, arg1: instStringType, type: typeTypeType}
// CHECK:STDOUT:     instFunctionType: {kind: CrossReference, arg0: ir0, arg1: instFunctionType, type: typeTypeType}
// CHECK:STDOUT:     instBoundMethodType: {kind: CrossReference, arg0: ir0, arg1: instBoundMethodType, type: typeTypeType}
// CHECK:STDOUT:     instNamespaceType: {kind: CrossReference, arg0: ir0, arg1: instNamespaceType, type: typeTypeType}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     block0:          {}
// CHECK:STDOUT: ...

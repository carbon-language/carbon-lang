// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-raw-sem-ir --builtin-sem-ir %s
//
// AUTOUPDATE

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        builtin_nodes.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   cross_reference_irs_size: 1
// CHECK:STDOUT:   functions:       {}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:           {}
// CHECK:STDOUT:   type_blocks:     {}
// CHECK:STDOUT:   nodes:
// CHECK:STDOUT:     nodeTypeType:    {kind: CrossReference, arg0: ir0, arg1: nodeTypeType, type: typeTypeType}
// CHECK:STDOUT:     nodeError:       {kind: CrossReference, arg0: ir0, arg1: nodeError, type: typeError}
// CHECK:STDOUT:     nodeBoolType:    {kind: CrossReference, arg0: ir0, arg1: nodeBoolType, type: typeTypeType}
// CHECK:STDOUT:     nodeIntegerType: {kind: CrossReference, arg0: ir0, arg1: nodeIntegerType, type: typeTypeType}
// CHECK:STDOUT:     nodeFloatingPointType: {kind: CrossReference, arg0: ir0, arg1: nodeFloatingPointType, type: typeTypeType}
// CHECK:STDOUT:     nodeStringType:  {kind: CrossReference, arg0: ir0, arg1: nodeStringType, type: typeTypeType}
// CHECK:STDOUT:     nodeFunctionType: {kind: CrossReference, arg0: ir0, arg1: nodeFunctionType, type: typeTypeType}
// CHECK:STDOUT:     nodeBoundMethodType: {kind: CrossReference, arg0: ir0, arg1: nodeBoundMethodType, type: typeTypeType}
// CHECK:STDOUT:     nodeNamespaceType: {kind: CrossReference, arg0: ir0, arg1: nodeNamespaceType, type: typeTypeType}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     block0:          {}
// CHECK:STDOUT: ...

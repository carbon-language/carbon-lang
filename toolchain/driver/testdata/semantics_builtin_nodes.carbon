// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump semantics-ir --include_builtins %s | \
// RUN:   %{FileCheck-strict}
// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: calls: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: callables: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeTypeType, type: nodeTypeType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeInvalidType, type: nodeInvalidType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeIntegerType, type: nodeTypeType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeFloatingPointType, type: nodeTypeType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeStringType, type: nodeTypeType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeEmptyStructType, type: nodeTypeType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeEmptyStruct, type: nodeEmptyStructType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeEmptyTupleType, type: nodeTypeType},
// CHECK:STDOUT:   {kind: CrossReference, arg0: ir0, arg1: nodeEmptyTuple, type: nodeEmptyTupleType},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

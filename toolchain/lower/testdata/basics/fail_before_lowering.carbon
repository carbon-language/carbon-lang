// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This validates that earlier errors prevent lowering, without crashing.
// AUTOUPDATE

// CHECK:STDERR: fail_before_lowering.carbon:[[@LINE+6]]:1: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: a;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_before_lowering.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `HandleInvalidParseStart`.
// CHECK:STDERR: a;
// CHECK:STDERR: ^
a;

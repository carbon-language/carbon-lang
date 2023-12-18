// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_undefined_bound.carbon:[[@LINE+3]]:8: ERROR: Semantics TODO: `HandleArrayExprWithoutBounds`.
// CHECK:STDERR: var a: [i32; ];
// CHECK:STDERR:        ^~~~~~~
var a: [i32; ];

// CHECK:STDOUT: --- fail_undefined_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:

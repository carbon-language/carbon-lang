// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: ERROR: Semantics TODO: `Handle -`.
// CHECK:STDERR: var b: i32 = a[-10];
// CHECK:STDERR:                ^
var b: i32 = a[-10];

// CHECK:STDOUT: file "fail_negative_indexing.carbon" {
// CHECK:STDOUT: }

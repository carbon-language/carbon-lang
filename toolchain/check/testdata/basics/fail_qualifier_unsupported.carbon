// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32;
// CHECK:STDERR: fail_qualifier_unsupported.carbon:[[@LINE+3]]:15: ERROR: Type `i32` does not support qualified expressions.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:               ^
var y: i32 = x.b;

// CHECK:STDOUT: file "fail_qualifier_unsupported.carbon" {
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %x.ref: ref i32 = name_reference "x", %x
// CHECK:STDOUT:   assign %y, <error>
// CHECK:STDOUT: }

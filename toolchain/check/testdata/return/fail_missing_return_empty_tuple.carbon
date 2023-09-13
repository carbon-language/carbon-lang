// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> () {
// CHECK:STDERR: fail_missing_return_empty_tuple.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: file "fail_missing_return_empty_tuple.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }

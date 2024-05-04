// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> () {
// CHECK:STDERR: fail_missing_return_empty_tuple.carbon:[[@LINE+3]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: --- fail_missing_return_empty_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %.loc7_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:

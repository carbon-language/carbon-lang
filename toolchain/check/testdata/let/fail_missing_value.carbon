// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_missing_value.carbon:[[@LINE+4]]:11: ERROR: Expected `=`; `let` declaration must have an initializer.
// CHECK:STDERR: let n: i32;
// CHECK:STDERR:           ^
// CHECK:STDERR:
let n: i32;

fn F() {
  // CHECK:STDERR: fail_missing_value.carbon:[[@LINE+3]]:13: ERROR: Expected `=`; `let` declaration must have an initializer.
  // CHECK:STDERR:   let n: i32;
  // CHECK:STDERR:             ^
  let n: i32;
}

// CHECK:STDOUT: --- fail_missing_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: i32 = bind_name n, <error>
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n: i32 = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

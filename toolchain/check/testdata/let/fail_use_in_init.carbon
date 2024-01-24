// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_use_in_init.carbon:[[@LINE+3]]:16: ERROR: Name `a` not found.
  // CHECK:STDERR:   let a: i32 = a;
  // CHECK:STDERR:                ^
  let a: i32 = a;
}

// CHECK:STDOUT: --- fail_use_in_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [template = <error>]
// CHECK:STDOUT:   %a: i32 = bind_name a, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

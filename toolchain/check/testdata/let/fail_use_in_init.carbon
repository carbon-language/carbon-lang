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

// CHECK:STDOUT: file "fail_use_in_init.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: <error> = name_reference "a", <error>
// CHECK:STDOUT:   %a: i32 = bind_name "a", <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_name_lookup.carbon:[[@LINE+3]]:3: ERROR: Name `x` not found.
  // CHECK:STDERR:   x;
  // CHECK:STDERR:   ^
  x;
}

// CHECK:STDOUT: file "fail_name_lookup.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: <error> = name_reference "x", <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

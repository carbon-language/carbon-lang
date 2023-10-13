// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


// CHECK:STDERR: fail_access_into_invalid.carbon:[[@LINE+3]]:10: ERROR: Name `a` not found.
// CHECK:STDERR: fn F() { a.b; }
// CHECK:STDERR:          ^
fn F() { a.b; }

// CHECK:STDOUT: file "fail_access_into_invalid.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: <error> = name_reference "a", <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

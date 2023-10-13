// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace A;

fn F() {
  // CHECK:STDERR: fail_dereference_namespace.carbon:[[@LINE+3]]:4: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   *A;
  // CHECK:STDERR:    ^
  *A;
}

// CHECK:STDOUT: file "fail_dereference_namespace.carbon" {
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <namespace> = name_reference "A", package.%.loc7
// CHECK:STDOUT:   %.loc13: ref <error> = dereference <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

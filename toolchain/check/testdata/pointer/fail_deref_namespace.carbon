// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace A;

fn F() {
  // CHECK:STDERR: fail_deref_namespace.carbon:[[@LINE+4]]:4: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   *A;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  *A;
  // CHECK:STDERR: fail_deref_namespace.carbon:[[@LINE+3]]:3: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   A->foo;
  // CHECK:STDERR:   ^
  A->foo;
}

// CHECK:STDOUT: --- fail_deref_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc14: <namespace> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %.loc14: ref <error> = deref <error>
// CHECK:STDOUT:   %A.ref.loc18: <namespace> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %.loc18: ref <error> = deref <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

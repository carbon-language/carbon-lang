// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() {
  // CHECK:STDERR: fail_deref_function.carbon:[[@LINE+4]]:4: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   *A;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  *A;
  // CHECK:STDERR: fail_deref_function.carbon:[[@LINE+3]]:3: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   A->foo;
  // CHECK:STDERR:   ^
  A->foo;
}

// CHECK:STDOUT: --- fail_deref_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc12: <function> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %.loc12: ref <error> = deref <error>
// CHECK:STDOUT:   %A.ref.loc16: <function> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %.loc16: ref <error> = deref <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

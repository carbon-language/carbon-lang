// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() {
  // CHECK:STDERR: fail_deref_function.carbon:[[@LINE+6]]:3: ERROR: Cannot dereference operand of non-pointer type `<function>`.
  // CHECK:STDERR:   *A;
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_deref_function.carbon:[[@LINE+3]]:4: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   *A;
  // CHECK:STDERR:    ^
  *A;
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
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %.loc14: ref <error> = deref <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

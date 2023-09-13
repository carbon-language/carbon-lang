// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn ConstMismatch(p: const {}*) -> const ({}*) {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:11: Cannot implicitly convert from `const {}*` to `const ({}*)`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:           ^
  return p;
}

// CHECK:STDOUT: file "fail_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @ConstMismatch
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p: const {}*) -> const ({}*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }

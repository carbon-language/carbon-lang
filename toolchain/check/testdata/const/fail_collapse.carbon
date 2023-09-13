// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:9: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: fn G(p: const (const i32)**) -> i32** {
// CHECK:STDERR:         ^
fn G(p: const (const i32)**) -> i32** {
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:11: Cannot implicitly convert from `const i32**` to `i32**`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:           ^
  return p;
}

// CHECK:STDOUT: file "fail_collapse.carbon" {
// CHECK:STDOUT:   %.loc10 = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: const i32**) -> i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }

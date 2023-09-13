// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// OK, `const (const i32)` is the same type as `const i32`.
// CHECK:STDERR: collapse.carbon:[[@LINE+3]]:25: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: fn F(p: const i32**) -> const (const i32)** {
// CHECK:STDERR:                         ^
fn F(p: const i32**) -> const (const i32)** {
  return p;
}

// CHECK:STDOUT: file "collapse.carbon" {
// CHECK:STDOUT:   %.loc11 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: const i32**) -> const i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return %p
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Simple {
  fn F();
}

// CHECK:STDERR: basic.carbon:[[@LINE+3]]:13: ERROR: Semantics TODO: `Unimplemented use of interface`.
// CHECK:STDERR: impl i32 as Simple {
// CHECK:STDERR:             ^~~~~~
impl i32 as Simple {
  fn F() {}
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:

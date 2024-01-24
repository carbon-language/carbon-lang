// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: i32) -> i32 {
  let b: i32 = a;
  return b;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b: i32 = bind_name b, %a.ref
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   return %b.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

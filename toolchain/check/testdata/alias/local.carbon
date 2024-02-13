// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  alias a = 0;
  var b: i32 = a;
  return a;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   %a: i32 = bind_template_name a, %.loc8 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref.loc9: i32 = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   assign %b.var, %a.ref.loc9
// CHECK:STDOUT:   %a.ref.loc10: i32 = name_ref a, %a [template = constants.%.1]
// CHECK:STDOUT:   return %a.ref.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:

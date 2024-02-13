// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var a: i32 = 0;
  // What should be the syntax for this? Should the expectation be that we
  // transparently detect the name_ref?
  // CHECK:STDERR: fail_todo_local.carbon:[[@LINE+3]]:13: ERROR: Alias initializer must be a template constant.
  // CHECK:STDERR:   alias b = a;
  // CHECK:STDERR:             ^
  alias b = a;
  return b;
}

// CHECK:STDOUT: --- fail_todo_local.carbon
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
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %a.ref: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %b: <error> = bind_template_name b, %a.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

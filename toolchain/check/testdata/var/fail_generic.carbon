// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:11: ERROR: `var` declaration cannot declare a compile-time binding.
  // CHECK:STDERR:   var x:! i32 = 0;
  // CHECK:STDERR:           ^~~
  var x:! i32 = 0;
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: i32 = bind_symbolic_name x, %x.var [symbolic]
// CHECK:STDOUT:   %.loc11: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, %.loc11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

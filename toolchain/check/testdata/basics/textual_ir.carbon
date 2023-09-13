// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE

fn Foo(n: i32) -> (i32, f64) {
  return (n + 2, 3.4);
}

// CHECK:STDOUT: file "textual_ir.carbon" {
// CHECK:STDOUT:   %.loc11 = fn_decl @Foo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: i32) -> %return: (i32, f64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_15: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_13.1: i32 = add %n, %.loc12_15
// CHECK:STDOUT:   %.loc12_13.2: i32 = stub_reference %.loc12_13.1
// CHECK:STDOUT:   %.loc12_18.1: f64 = real_literal 34e-1
// CHECK:STDOUT:   %.loc12_18.2: f64 = stub_reference %.loc12_18.1
// CHECK:STDOUT:   %.loc12_21: (i32, f64) = tuple_literal (%.loc12_13.2, %.loc12_18.2)
// CHECK:STDOUT:   return %.loc12_21
// CHECK:STDOUT: }

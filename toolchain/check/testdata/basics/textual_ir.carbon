// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-semantics-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE

fn Foo(n: i32) -> (i32, f64) {
  return (n + 2, 3.4);
}

// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc11 = fn_decl @Foo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: i32) -> %return: (i32, f64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_15: i32 = int_literal 2
// CHECK:STDOUT:   %.loc15_13.1: i32 = add %n, %.loc15_15
// CHECK:STDOUT:   %.loc15_13.2: i32 = stub_reference %.loc15_13.1
// CHECK:STDOUT:   %.loc15_18.1: f64 = real_literal 34e-1
// CHECK:STDOUT:   %.loc15_18.2: f64 = stub_reference %.loc15_18.1
// CHECK:STDOUT:   %.loc15_21: (i32, f64) = tuple_value (%.loc15_13.2, %.loc15_18.2)
// CHECK:STDOUT:   return %.loc15_21
// CHECK:STDOUT: }

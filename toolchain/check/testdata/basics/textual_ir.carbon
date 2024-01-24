// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE

fn Foo(n: i32) -> (i32, i32, f64) {
  return (n, 2, 3.4);
}

// CHECK:STDOUT: --- textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32, f64) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32, f64) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: f64 = real_literal 34e-1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: i32) -> %return: (i32, i32, f64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc12_14: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_17: f64 = real_literal 34e-1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_20.1: (i32, i32, f64) = tuple_literal (%n.ref, %.loc12_14, %.loc12_17)
// CHECK:STDOUT:   %.loc12_20.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc12_20.3: init i32 = initialize_from %n.ref to %.loc12_20.2
// CHECK:STDOUT:   %.loc12_20.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc12_20.5: init i32 = initialize_from %.loc12_14 to %.loc12_20.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_20.6: ref f64 = tuple_access %return, element2
// CHECK:STDOUT:   %.loc12_20.7: init f64 = initialize_from %.loc12_17 to %.loc12_20.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_20.8: init (i32, i32, f64) = tuple_init (%.loc12_20.3, %.loc12_20.5, %.loc12_20.7) to %return
// CHECK:STDOUT:   %.loc12_20.9: init (i32, i32, f64) = converted %.loc12_20.1, %.loc12_20.8
// CHECK:STDOUT:   return %.loc12_20.9
// CHECK:STDOUT: }
// CHECK:STDOUT:

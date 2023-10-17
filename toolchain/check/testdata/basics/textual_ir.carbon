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
// CHECK:STDOUT:   %.loc11: type = ptr_type (i32, f64)
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: i32) -> %return: (i32, f64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", %n
// CHECK:STDOUT:   %.loc12_15: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_13: i32 = add %n.ref, %.loc12_15
// CHECK:STDOUT:   %.loc12_18: f64 = real_literal 34e-1
// CHECK:STDOUT:   %.loc12_21.1: (i32, f64) = tuple_literal (%.loc12_13, %.loc12_18)
// CHECK:STDOUT:   %.loc12_21.2: ref i32 = tuple_access %return, member0
// CHECK:STDOUT:   %.loc12_21.3: init i32 = initialize_from %.loc12_13 to %.loc12_21.2
// CHECK:STDOUT:   %.loc12_21.4: ref f64 = tuple_access %return, member1
// CHECK:STDOUT:   %.loc12_21.5: init f64 = initialize_from %.loc12_18 to %.loc12_21.4
// CHECK:STDOUT:   %.loc12_21.6: init (i32, f64) = tuple_init %.loc12_21.1, (%.loc12_21.3, %.loc12_21.5)
// CHECK:STDOUT:   return %.loc12_21.6
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Deref(n: i32) {
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  *n;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `()`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  *();
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `{}`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  *{};
}

// CHECK:STDOUT: --- fail_deref_not_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: () = tuple_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Deref = %Deref} [template]
// CHECK:STDOUT:   %Deref: <function> = fn_decl @Deref [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc11: ref <error> = deref %n.ref
// CHECK:STDOUT:   %.loc15_5.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_5.2: () = tuple_value () [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_5.3: () = converted %.loc15_5.1, %.loc15_5.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_3: ref <error> = deref %.loc15_5.3
// CHECK:STDOUT:   %.loc19_5.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc19_5.2: {} = struct_value () [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_5.3: {} = converted %.loc19_5.1, %.loc19_5.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_3: ref <error> = deref %.loc19_5.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

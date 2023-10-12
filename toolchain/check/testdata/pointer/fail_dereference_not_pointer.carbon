// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Deref(n: i32) {
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  *n;
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `() as type`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  *();
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `{} as type`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  *{};
}

// CHECK:STDOUT: file "fail_dereference_not_pointer.carbon" {
// CHECK:STDOUT:   %Deref: <function> = fn_decl @Deref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", %n
// CHECK:STDOUT:   %.loc11: ref <error> = dereference %n.ref
// CHECK:STDOUT:   %.loc15_5.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc15_5.2: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_5.3: () = tuple_value %.loc15_5.2, ()
// CHECK:STDOUT:   %.loc15_3: ref <error> = dereference %.loc15_5.3
// CHECK:STDOUT:   %.loc19_5.1: type = struct_type {}
// CHECK:STDOUT:   %.loc19_5.2: {} = struct_literal ()
// CHECK:STDOUT:   %.loc19_5.3: {} = struct_value %.loc19_5.2, ()
// CHECK:STDOUT:   %.loc19_3: ref <error> = dereference %.loc19_5.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

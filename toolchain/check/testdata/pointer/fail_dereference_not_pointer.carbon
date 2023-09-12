// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_dereference_not_pointer.carbon" {

fn Deref(n: i32) {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Deref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  *n;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: ref <error> = dereference %n
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: Cannot dereference operand of non-pointer type `() as type`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  *();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: () = tuple_literal ()
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_3: ref <error> = dereference %.loc[[@LINE-3]]_5.2
  // CHECK:STDERR: fail_dereference_not_pointer.carbon:[[@LINE+3]]:3: Cannot dereference operand of non-pointer type `{} as type`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  *{};
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: type = struct_type {}
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: {} = struct_literal ()
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_3: ref <error> = dereference %.loc[[@LINE-3]]_5.2
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}

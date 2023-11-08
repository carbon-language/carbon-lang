// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Procedure() {
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE+3]]:3: ERROR: Cannot declare a `returned var` in function with no declared return type.
  // CHECK:STDERR:   returned var v: () = ();
  // CHECK:STDERR:   ^
  returned var v: () = ();
  return;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc11: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_returned_var_no_return_type.carbon" {
// CHECK:STDOUT:   %Procedure: <function> = fn_decl @Procedure
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: () = tuple_literal ()
// CHECK:STDOUT:   %v: () = bind_name "v", <error>
// CHECK:STDOUT:   %.loc11_25: () = tuple_literal ()
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

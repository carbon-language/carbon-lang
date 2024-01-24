// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Procedure() {
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE+6]]:3: ERROR: Cannot declare a `returned var` in this function.
  // CHECK:STDERR:   returned var v: () = ();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE-4]]:1: There was no return type provided.
  // CHECK:STDERR: fn Procedure() {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~
  returned var v: () = ();
  return;
}

// CHECK:STDOUT: --- fail_returned_var_no_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Procedure = %Procedure} [template]
// CHECK:STDOUT:   %Procedure: <function> = fn_decl @Procedure [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_20.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc14_20.2: type = converted %.loc14_20.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %v: () = bind_name v, <error>
// CHECK:STDOUT:   %.loc14_25: () = tuple_literal ()
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

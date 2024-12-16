// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_no_return_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_no_return_type.carbon

fn Procedure() {
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE+7]]:3: error: cannot declare a `returned var` in this function [ReturnedVarWithNoReturnType]
  // CHECK:STDERR:   returned var v: () = ();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE-4]]:1: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR: fn Procedure() {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  returned var v: () = ();
  return;
}

fn ForgotReturnType() {
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE+6]]:3: error: cannot declare a `returned var` in this function [ReturnedVarWithNoReturnType]
  // CHECK:STDERR:   returned var v: () = ();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_returned_var_no_return_type.carbon:[[@LINE-4]]:1: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR: fn ForgotReturnType() {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  returned var v: () = ();
  return var;
}

// CHECK:STDOUT: --- fail_returned_var_no_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Procedure.type: type = fn_type @Procedure [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Procedure: %Procedure.type = struct_value () [template]
// CHECK:STDOUT:   %ForgotReturnType.type: type = fn_type @ForgotReturnType [template]
// CHECK:STDOUT:   %ForgotReturnType: %ForgotReturnType.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Procedure = %Procedure.decl
// CHECK:STDOUT:     .ForgotReturnType = %ForgotReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Procedure.decl: %Procedure.type = fn_decl @Procedure [template = constants.%Procedure] {} {}
// CHECK:STDOUT:   %ForgotReturnType.decl: %ForgotReturnType.type = fn_decl @ForgotReturnType [template = constants.%ForgotReturnType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc18_20.2: type = converted %.loc18_20.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v: %empty_tuple.type = bind_name v, <error>
// CHECK:STDOUT:   %.loc18_25: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForgotReturnType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_20.2: type = converted %.loc23_20.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v: %empty_tuple.type = bind_name v, <error>
// CHECK:STDOUT:   %.loc23_25: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_16: %empty_tuple.type = converted %v, %tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc23_16
// CHECK:STDOUT: }
// CHECK:STDOUT:

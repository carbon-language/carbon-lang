// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_init_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_init_type_mismatch.carbon

fn Main() {
  // CHECK:STDERR: fail_init_type_mismatch.carbon:[[@LINE+3]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   var x: {} = ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  var x: {} = ();
}

// CHECK:STDOUT: --- fail_init_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_11.2: type = converted %.loc15_11.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %x.var: ref %empty_struct_type = var x
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %x.var
// CHECK:STDOUT:   %.loc15_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_16.2: %empty_tuple.type = converted %.loc15_16.1, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_17: %empty_struct_type = converted %.loc15_16.1, <error> [template = <error>]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

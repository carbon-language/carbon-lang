// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_duplicate_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_duplicate_decl.carbon


fn Main() {
  var x: () = ();
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: error: duplicate name being declared in the same scope
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: name is previously declared here
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  var x: () = ();
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_11.2: type = converted %.loc13_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var.loc13: ref %.1 = var x
// CHECK:STDOUT:   %x.loc13: ref %.1 = bind_name x, %x.var.loc13
// CHECK:STDOUT:   %.loc13_16.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_16.2: init %.1 = tuple_init () to %x.var.loc13 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_17: init %.1 = converted %.loc13_16.1, %.loc13_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var.loc13, %.loc13_17
// CHECK:STDOUT:   %.loc20_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc20_11.2: type = converted %.loc20_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var.loc20: ref %.1 = var x
// CHECK:STDOUT:   %x.loc20: ref %.1 = bind_name x, %x.var.loc20
// CHECK:STDOUT:   %.loc20_16.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc20_16.2: init %.1 = tuple_init () to %x.var.loc20 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc20_17: init %.1 = converted %.loc20_16.1, %.loc20_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var.loc20, %.loc20_17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

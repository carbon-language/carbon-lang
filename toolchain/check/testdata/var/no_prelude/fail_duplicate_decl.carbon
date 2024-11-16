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
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: error: duplicate name being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  var x: () = ();
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
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
// CHECK:STDOUT:   %.loc13_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_11.2: type = converted %.loc13_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x.var.loc13: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %x.loc13: ref %empty_tuple.type = bind_name x, %x.var.loc13
// CHECK:STDOUT:   %.loc13_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc13 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_17: init %empty_tuple.type = converted %.loc13_16.1, %.loc13_16.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc13, %.loc13_17
// CHECK:STDOUT:   %.loc20_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_11.2: type = converted %.loc20_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x.var.loc20: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %x.loc20: ref %empty_tuple.type = bind_name x, %x.var.loc20
// CHECK:STDOUT:   %.loc20_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc20 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20_17: init %empty_tuple.type = converted %.loc20_16.1, %.loc20_16.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc20, %.loc20_17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

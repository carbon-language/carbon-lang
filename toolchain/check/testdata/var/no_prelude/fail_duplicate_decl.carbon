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
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+7]]:7: error: duplicate name `x` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var x: () = ();
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc13: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc13_3.1: %empty_tuple.type = var_pattern %x.patt.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc13: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc13_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_3.2: init %empty_tuple.type = converted %.loc13_16.1, %.loc13_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc13, %.loc13_3.2
// CHECK:STDOUT:   %.loc13_11.1: type = splice_block %.loc13_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc13_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_11.3: type = converted %.loc13_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc13: ref %empty_tuple.type = bind_name x, %x.var.loc13
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc21: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc21_3.1: %empty_tuple.type = var_pattern %x.patt.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc21: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc21_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc21_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc21 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21_3.2: init %empty_tuple.type = converted %.loc21_16.1, %.loc21_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc21, %.loc21_3.2
// CHECK:STDOUT:   %.loc21_11.1: type = splice_block %.loc21_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc21_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc21_11.3: type = converted %.loc21_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc21: ref %empty_tuple.type = bind_name x, %x.var.loc21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

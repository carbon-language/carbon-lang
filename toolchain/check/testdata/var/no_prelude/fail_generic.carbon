// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_generic.carbon

fn Main() {
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:11: error: `var` declaration cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var x:! () = ();
  // CHECK:STDERR:           ^~
  var x:! () = ();
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_symbolic_name x [symbolic]
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
// CHECK:STDOUT:   %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_symbolic_name x, %x.var [symbolic = constants.%x]
// CHECK:STDOUT:   %.loc15_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_17.2: init %empty_tuple.type = tuple_init () to %x.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_18: init %empty_tuple.type = converted %.loc15_17.1, %.loc15_17.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %.loc15_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

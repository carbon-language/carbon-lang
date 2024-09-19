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
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:11: error: `var` declaration cannot declare a compile-time binding
  // CHECK:STDERR:   var x:! () = ();
  // CHECK:STDERR:           ^~
  var x:! () = ();
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0 [symbolic]
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
// CHECK:STDOUT:   %.loc15_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc15_12.2: type = converted %.loc15_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0, %x.var [symbolic = constants.%x]
// CHECK:STDOUT:   %.loc15_17.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc15_17.2: init %.1 = tuple_init () to %x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_18: init %.1 = converted %.loc15_17.1, %.loc15_17.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var, %.loc15_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

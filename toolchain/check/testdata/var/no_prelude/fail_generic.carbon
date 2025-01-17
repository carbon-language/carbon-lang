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
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:11: error: `var` declaration cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var x:! () = ();
  // CHECK:STDERR:           ^~
  // CHECK:STDERR:
  var x:! () = ();
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_symbolic_name x [symbolic]
// CHECK:STDOUT:   %x.patt: %empty_tuple.type = symbolic_binding_pattern x [symbolic]
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
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = symbolic_binding_pattern x [symbolic = constants.%x.patt]
// CHECK:STDOUT:     %.loc16_3.1: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %empty_tuple.type = var <invalid>
// CHECK:STDOUT:   %.loc16_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_17.2: init %empty_tuple.type = tuple_init () to %.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_3.2: init %empty_tuple.type = converted %.loc16_17.1, %.loc16_17.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.var, %.loc16_3.2
// CHECK:STDOUT:   %.loc16_12.1: type = splice_block %.loc16_12.3 [template = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_12.3: type = converted %.loc16_12.2, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_symbolic_name x, %.var [symbolic = constants.%x]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

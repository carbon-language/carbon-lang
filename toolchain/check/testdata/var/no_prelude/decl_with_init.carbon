// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/decl_with_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/decl_with_init.carbon

fn Main() {
  var x: () = ();
}

// CHECK:STDOUT: --- decl_with_init.carbon
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
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc12_3.1: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc12_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.2: init %empty_tuple.type = tuple_init () to %x.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_3.2: init %empty_tuple.type = converted %.loc12_16.1, %.loc12_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_11.1: type = splice_block %.loc12_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_11.3: type = converted %.loc12_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

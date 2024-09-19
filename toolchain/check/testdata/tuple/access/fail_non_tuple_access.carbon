// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon

fn Main() {
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:3: error: type `i32` does not support indexing
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  0[1];

  var non_tuple: [i32; 2] = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+3]]:20: error: type `[i32; 2]` does not support tuple indexing; only tuples can be indexed that way
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  var first: i32 = non_tuple.0;
}

// CHECK:STDOUT: --- fail_non_tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%.7, %.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_3: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_5: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_24: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_19.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_19.2: type = converted %int.make_type_32.loc18, %.loc18_19.1 [template = i32]
// CHECK:STDOUT:   %.loc18_25: type = array_type %.loc18_24, i32 [template = constants.%.5]
// CHECK:STDOUT:   %non_tuple.var: ref %.5 = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %.5 = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %.loc18_30: i32 = int_literal 5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_33: i32 = int_literal 5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_34.1: %.8 = tuple_literal (%.loc18_30, %.loc18_33)
// CHECK:STDOUT:   %.loc18_34.2: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_34.3: ref i32 = array_index %non_tuple.var, %.loc18_34.2
// CHECK:STDOUT:   %.loc18_34.4: init i32 = initialize_from %.loc18_30 to %.loc18_34.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_34.5: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc18_34.6: ref i32 = array_index %non_tuple.var, %.loc18_34.5
// CHECK:STDOUT:   %.loc18_34.7: init i32 = initialize_from %.loc18_33 to %.loc18_34.6 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_34.8: init %.5 = array_init (%.loc18_34.4, %.loc18_34.7) to %non_tuple.var [template = constants.%array]
// CHECK:STDOUT:   %.loc18_35: init %.5 = converted %.loc18_34.1, %.loc18_34.8 [template = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_35
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_14.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_14.2: type = converted %int.make_type_32.loc22, %.loc22_14.1 [template = i32]
// CHECK:STDOUT:   %first.var: ref i32 = var first
// CHECK:STDOUT:   %first: ref i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %.5 = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %.loc22_30: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:

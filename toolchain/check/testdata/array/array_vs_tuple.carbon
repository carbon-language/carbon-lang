// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon

fn G() {
  // These should have two different constant values.
  var a: [i32; 3] = (1, 2, 3);
  var b: (i32, i32, i32) = (1, 2, 3);
}

// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT:   %.9: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %tuple: %.7 = tuple_value (%.5, %.6, %.2) [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_11.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_11.2: type = converted %int.make_type_32.loc13, %.loc13_11.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17: type = array_type %.loc13_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_25: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_28: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_29.1: %.7 = tuple_literal (%.loc13_22, %.loc13_25, %.loc13_28)
// CHECK:STDOUT:   %.loc13_29.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_29.3: ref i32 = array_index %a.var, %.loc13_29.2
// CHECK:STDOUT:   %.loc13_29.4: init i32 = initialize_from %.loc13_22 to %.loc13_29.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_29.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_29.6: ref i32 = array_index %a.var, %.loc13_29.5
// CHECK:STDOUT:   %.loc13_29.7: init i32 = initialize_from %.loc13_25 to %.loc13_29.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_29.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_29.9: ref i32 = array_index %a.var, %.loc13_29.8
// CHECK:STDOUT:   %.loc13_29.10: init i32 = initialize_from %.loc13_28 to %.loc13_29.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_29.11: init %.3 = array_init (%.loc13_29.4, %.loc13_29.7, %.loc13_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_30: init %.3 = converted %.loc13_29.1, %.loc13_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_30
// CHECK:STDOUT:   %int.make_type_32.loc14_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_24.1: %.9 = tuple_literal (%int.make_type_32.loc14_11, %int.make_type_32.loc14_16, %int.make_type_32.loc14_21)
// CHECK:STDOUT:   %.loc14_24.2: type = value_of_initializer %int.make_type_32.loc14_11 [template = i32]
// CHECK:STDOUT:   %.loc14_24.3: type = converted %int.make_type_32.loc14_11, %.loc14_24.2 [template = i32]
// CHECK:STDOUT:   %.loc14_24.4: type = value_of_initializer %int.make_type_32.loc14_16 [template = i32]
// CHECK:STDOUT:   %.loc14_24.5: type = converted %int.make_type_32.loc14_16, %.loc14_24.4 [template = i32]
// CHECK:STDOUT:   %.loc14_24.6: type = value_of_initializer %int.make_type_32.loc14_21 [template = i32]
// CHECK:STDOUT:   %.loc14_24.7: type = converted %int.make_type_32.loc14_21, %.loc14_24.6 [template = i32]
// CHECK:STDOUT:   %.loc14_24.8: type = converted %.loc14_24.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %b.var: ref %.7 = var b
// CHECK:STDOUT:   %b: ref %.7 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14_29: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_32: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_35: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_36.1: %.7 = tuple_literal (%.loc14_29, %.loc14_32, %.loc14_35)
// CHECK:STDOUT:   %.loc14_36.2: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_36.3: init i32 = initialize_from %.loc14_29 to %.loc14_36.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_36.4: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_36.5: init i32 = initialize_from %.loc14_32 to %.loc14_36.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_36.6: ref i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_36.7: init i32 = initialize_from %.loc14_35 to %.loc14_36.6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_36.8: init %.7 = tuple_init (%.loc14_36.3, %.loc14_36.5, %.loc14_36.7) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_37: init %.7 = converted %.loc14_36.1, %.loc14_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc14_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:

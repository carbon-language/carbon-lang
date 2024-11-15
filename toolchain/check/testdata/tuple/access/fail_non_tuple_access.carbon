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
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:3: error: type `i32` does not support indexing [TypeNotIndexable]
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  0[1];

  var non_tuple: [i32; 2] = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+3]]:20: error: type `[i32; 2]` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  var first: i32 = non_tuple.0;
}

// CHECK:STDOUT: --- fail_non_tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 1 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.32: type = array_type %.31, i32 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %array: %.32 = tuple_value (%.34, %.34) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.6
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_3: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_5: i32 = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_24.1: i32 = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_19.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_19.2: type = converted %int.make_type_32.loc18, %.loc18_19.1 [template = i32]
// CHECK:STDOUT:   %.loc18_24.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc18_24.3: <bound method> = bound_method %.loc18_24.1, %.loc18_24.2 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc18_24.3(%.loc18_24.1) [template = constants.%.31]
// CHECK:STDOUT:   %.loc18_24.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:   %.loc18_24.5: Core.IntLiteral = converted %.loc18_24.1, %.loc18_24.4 [template = constants.%.31]
// CHECK:STDOUT:   %.loc18_25: type = array_type %.loc18_24.5, i32 [template = constants.%.32]
// CHECK:STDOUT:   %non_tuple.var: ref %.32 = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %.32 = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %.loc18_30: i32 = int_value 5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc18_33: i32 = int_value 5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc18_34.1: %tuple.type = tuple_literal (%.loc18_30, %.loc18_33)
// CHECK:STDOUT:   %.loc18_34.2: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_34.3: ref i32 = array_index %non_tuple.var, %.loc18_34.2
// CHECK:STDOUT:   %.loc18_34.4: init i32 = initialize_from %.loc18_30 to %.loc18_34.3 [template = constants.%.34]
// CHECK:STDOUT:   %.loc18_34.5: i32 = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_34.6: ref i32 = array_index %non_tuple.var, %.loc18_34.5
// CHECK:STDOUT:   %.loc18_34.7: init i32 = initialize_from %.loc18_33 to %.loc18_34.6 [template = constants.%.34]
// CHECK:STDOUT:   %.loc18_34.8: init %.32 = array_init (%.loc18_34.4, %.loc18_34.7) to %non_tuple.var [template = constants.%array]
// CHECK:STDOUT:   %.loc18_35: init %.32 = converted %.loc18_34.1, %.loc18_34.8 [template = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_35
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_14.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_14.2: type = converted %int.make_type_32.loc22, %.loc22_14.1 [template = i32]
// CHECK:STDOUT:   %first.var: ref i32 = var first
// CHECK:STDOUT:   %first: ref i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %.32 = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %.loc22_30: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

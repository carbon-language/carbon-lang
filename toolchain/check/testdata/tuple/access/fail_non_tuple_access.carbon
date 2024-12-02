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
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:3: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.6 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.8: type = array_type %.7, %i32 [template]
// CHECK:STDOUT:   %.10: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.6) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.10, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.6) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %array: %.8 = tuple_value (%.35, %.35) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.6
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
// CHECK:STDOUT:   %.loc16_3: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_5: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_19.1: Core.IntLiteral = int_value 32 [template = constants.%.6]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_19.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_24: Core.IntLiteral = int_value 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_19.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.3: type = converted %int.make_type_signed.loc18, %.loc18_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_25: type = array_type %.loc18_24, %i32 [template = constants.%.8]
// CHECK:STDOUT:   %non_tuple.var: ref %.8 = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %.8 = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %.loc18_30: Core.IntLiteral = int_value 5 [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_33: Core.IntLiteral = int_value 5 [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_34.1: %tuple.type = tuple_literal (%.loc18_30, %.loc18_33)
// CHECK:STDOUT:   %.loc18_34.2: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_34.3: <bound method> = bound_method %.loc18_30, %.loc18_34.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc18_34.4: <specific function> = specific_function %.loc18_34.3, @Convert.2(constants.%.6) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc18_34.1: init %i32 = call %.loc18_34.4(%.loc18_30) [template = constants.%.35]
// CHECK:STDOUT:   %.loc18_34.5: init %i32 = converted %.loc18_30, %int.convert_checked.loc18_34.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc18_34.6: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_34.7: ref %i32 = array_index %non_tuple.var, %.loc18_34.6
// CHECK:STDOUT:   %.loc18_34.8: init %i32 = initialize_from %.loc18_34.5 to %.loc18_34.7 [template = constants.%.35]
// CHECK:STDOUT:   %.loc18_34.9: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_34.10: <bound method> = bound_method %.loc18_33, %.loc18_34.9 [template = constants.%.33]
// CHECK:STDOUT:   %.loc18_34.11: <specific function> = specific_function %.loc18_34.10, @Convert.2(constants.%.6) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc18_34.2: init %i32 = call %.loc18_34.11(%.loc18_33) [template = constants.%.35]
// CHECK:STDOUT:   %.loc18_34.12: init %i32 = converted %.loc18_33, %int.convert_checked.loc18_34.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc18_34.13: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_34.14: ref %i32 = array_index %non_tuple.var, %.loc18_34.13
// CHECK:STDOUT:   %.loc18_34.15: init %i32 = initialize_from %.loc18_34.12 to %.loc18_34.14 [template = constants.%.35]
// CHECK:STDOUT:   %.loc18_34.16: init %.8 = array_init (%.loc18_34.8, %.loc18_34.15) to %non_tuple.var [template = constants.%array]
// CHECK:STDOUT:   %.loc18_35: init %.8 = converted %.loc18_34.1, %.loc18_34.16 [template = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_35
// CHECK:STDOUT:   %.loc22_14.1: Core.IntLiteral = int_value 32 [template = constants.%.6]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%.loc22_14.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_14.2: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_14.3: type = converted %int.make_type_signed.loc22, %.loc22_14.2 [template = constants.%i32]
// CHECK:STDOUT:   %first.var: ref %i32 = var first
// CHECK:STDOUT:   %first: ref %i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %.8 = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %.loc22_30: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

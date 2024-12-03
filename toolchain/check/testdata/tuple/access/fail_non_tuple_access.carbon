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
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.5: type = array_type %int.2, %i32 [template]
// CHECK:STDOUT:   %int.5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.22: <bound method> = bound_method %int.5.1, %Convert.14 [template]
// CHECK:STDOUT:   %.23: <specific function> = specific_function %.22, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%int.5.2, %int.5.2) [template]
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
// CHECK:STDOUT:   %int.0.loc16: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %int.1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %int.32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%int.32.loc18) [template = constants.%i32]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2]
// CHECK:STDOUT:   %.loc18_19.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.2: type = converted %int.make_type_signed.loc18, %.loc18_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_25: type = array_type %int.2, %i32 [template = constants.%.5]
// CHECK:STDOUT:   %non_tuple.var: ref %.5 = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %.5 = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %int.5.loc18_30: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %int.5.loc18_33: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %.loc18_34.1: %tuple.type = tuple_literal (%int.5.loc18_30, %int.5.loc18_33)
// CHECK:STDOUT:   %.loc18_34.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_34.3: <bound method> = bound_method %int.5.loc18_30, %.loc18_34.2 [template = constants.%.22]
// CHECK:STDOUT:   %.loc18_34.4: <specific function> = specific_function %.loc18_34.3, @Convert.2(constants.%int.32) [template = constants.%.23]
// CHECK:STDOUT:   %int.convert_checked.loc18_34.1: init %i32 = call %.loc18_34.4(%int.5.loc18_30) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc18_34.5: init %i32 = converted %int.5.loc18_30, %int.convert_checked.loc18_34.1 [template = constants.%int.5.2]
// CHECK:STDOUT:   %int.0.loc18: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc18_34.6: ref %i32 = array_index %non_tuple.var, %int.0.loc18
// CHECK:STDOUT:   %.loc18_34.7: init %i32 = initialize_from %.loc18_34.5 to %.loc18_34.6 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc18_34.8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_34.9: <bound method> = bound_method %int.5.loc18_33, %.loc18_34.8 [template = constants.%.22]
// CHECK:STDOUT:   %.loc18_34.10: <specific function> = specific_function %.loc18_34.9, @Convert.2(constants.%int.32) [template = constants.%.23]
// CHECK:STDOUT:   %int.convert_checked.loc18_34.2: init %i32 = call %.loc18_34.10(%int.5.loc18_33) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc18_34.11: init %i32 = converted %int.5.loc18_33, %int.convert_checked.loc18_34.2 [template = constants.%int.5.2]
// CHECK:STDOUT:   %int.1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc18_34.12: ref %i32 = array_index %non_tuple.var, %int.1.loc18
// CHECK:STDOUT:   %.loc18_34.13: init %i32 = initialize_from %.loc18_34.11 to %.loc18_34.12 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc18_34.14: init %.5 = array_init (%.loc18_34.7, %.loc18_34.13) to %non_tuple.var [template = constants.%array]
// CHECK:STDOUT:   %.loc18_35: init %.5 = converted %.loc18_34.1, %.loc18_34.14 [template = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_35
// CHECK:STDOUT:   %int.32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%int.32.loc22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_14.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_14.2: type = converted %int.make_type_signed.loc22, %.loc22_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %first.var: ref %i32 = var first
// CHECK:STDOUT:   %first: ref %i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %.5 = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %int.0.loc22: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

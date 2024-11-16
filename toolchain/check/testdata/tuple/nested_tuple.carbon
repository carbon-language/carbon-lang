// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.3, i32) [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 76 [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%tuple.type.6, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 12 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 76 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.3 = tuple_value (%.31, %.33) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.4 = tuple_value (%tuple.1, %.35) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_18: %tuple.type.1 = tuple_literal (%int.make_type_32.loc11_10, %int.make_type_32.loc11_15)
// CHECK:STDOUT:   %int.make_type_32.loc11_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_24.1: %tuple.type.2 = tuple_literal (%.loc11_18, %int.make_type_32.loc11_21)
// CHECK:STDOUT:   %.loc11_24.2: type = value_of_initializer %int.make_type_32.loc11_10 [template = i32]
// CHECK:STDOUT:   %.loc11_24.3: type = converted %int.make_type_32.loc11_10, %.loc11_24.2 [template = i32]
// CHECK:STDOUT:   %.loc11_24.4: type = value_of_initializer %int.make_type_32.loc11_15 [template = i32]
// CHECK:STDOUT:   %.loc11_24.5: type = converted %int.make_type_32.loc11_15, %.loc11_24.4 [template = i32]
// CHECK:STDOUT:   %.loc11_24.6: type = converted %.loc11_18, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %.loc11_24.7: type = value_of_initializer %int.make_type_32.loc11_21 [template = i32]
// CHECK:STDOUT:   %.loc11_24.8: type = converted %int.make_type_32.loc11_21, %.loc11_24.7 [template = i32]
// CHECK:STDOUT:   %.loc11_24.9: type = converted %.loc11_24.1, constants.%tuple.type.4 [template = constants.%tuple.type.4]
// CHECK:STDOUT:   %x.var: ref %tuple.type.4 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.4 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_30: Core.IntLiteral = int_value 12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_34: Core.IntLiteral = int_value 76 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_36.1: %tuple.type.6 = tuple_literal (%.loc11_30, %.loc11_34)
// CHECK:STDOUT:   %.loc11_39: Core.IntLiteral = int_value 6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_40.1: %tuple.type.7 = tuple_literal (%.loc11_36.1, %.loc11_39)
// CHECK:STDOUT:   %.loc11_36.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_36.3: <bound method> = bound_method %.loc11_30, %.loc11_36.2 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.1: init i32 = call %.loc11_36.3(%.loc11_30) [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_36.4: init i32 = converted %.loc11_30, %int.convert_checked.loc11_36.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_40.2: ref %tuple.type.3 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_36.5: ref i32 = tuple_access %.loc11_40.2, element0
// CHECK:STDOUT:   %.loc11_36.6: init i32 = initialize_from %.loc11_36.4 to %.loc11_36.5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_36.7: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_36.8: <bound method> = bound_method %.loc11_34, %.loc11_36.7 [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.2: init i32 = call %.loc11_36.8(%.loc11_34) [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_36.9: init i32 = converted %.loc11_34, %int.convert_checked.loc11_36.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_36.10: ref i32 = tuple_access %.loc11_40.2, element1
// CHECK:STDOUT:   %.loc11_36.11: init i32 = initialize_from %.loc11_36.9 to %.loc11_36.10 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_36.12: init %tuple.type.3 = tuple_init (%.loc11_36.6, %.loc11_36.11) to %.loc11_40.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_40.3: init %tuple.type.3 = converted %.loc11_36.1, %.loc11_36.12 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_40.4: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_40.5: <bound method> = bound_method %.loc11_39, %.loc11_40.4 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc11_40: init i32 = call %.loc11_40.5(%.loc11_39) [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_40.6: init i32 = converted %.loc11_39, %int.convert_checked.loc11_40 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_40.7: ref i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_40.8: init i32 = initialize_from %.loc11_40.6 to %.loc11_40.7 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_40.9: init %tuple.type.4 = tuple_init (%.loc11_40.3, %.loc11_40.8) to file.%x.var [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc11_41: init %tuple.type.4 = converted %.loc11_40.1, %.loc11_40.9 [template = constants.%tuple.2]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_41
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

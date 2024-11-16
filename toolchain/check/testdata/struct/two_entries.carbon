// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/two_entries.carbon

let v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
let w: {.a: i32, .b: i32} = v;

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: --- two_entries.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %.2, .b: %.2} [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.32: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.35: %.2 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.3 = struct_value (%.32, %.35) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_13: init type = call constants.%Int(%.loc11_13.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %int.make_type_signed.loc11_13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %int.make_type_signed.loc11_13, %.loc11_13.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_22: init type = call constants.%Int(%.loc11_22.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.2: type = value_of_initializer %int.make_type_signed.loc11_22 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.3: type = converted %int.make_type_signed.loc11_22, %.loc11_22.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_25: type = struct_type {.a: %.2, .b: %.2} [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_13: init type = call constants.%Int(%.loc12_13.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_13.2: type = value_of_initializer %int.make_type_signed.loc12_13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_13.3: type = converted %int.make_type_signed.loc12_13, %.loc12_13.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_22: init type = call constants.%Int(%.loc12_22.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_22.2: type = value_of_initializer %int.make_type_signed.loc12_22 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_22.3: type = converted %int.make_type_signed.loc12_22, %.loc12_22.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_25: type = struct_type {.a: %.2, .b: %.2} [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_13: init type = call constants.%Int(%.loc14_13.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_13.2: type = value_of_initializer %int.make_type_signed.loc14_13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_13.3: type = converted %int.make_type_signed.loc14_13, %.loc14_13.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_22: init type = call constants.%Int(%.loc14_22.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_22.2: type = value_of_initializer %int.make_type_signed.loc14_22 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_22.3: type = converted %int.make_type_signed.loc14_22, %.loc14_22.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_25: type = struct_type {.a: %.2, .b: %.2} [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref %.3 = var x
// CHECK:STDOUT:   %x: ref %.3 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc15_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15_13: init type = call constants.%Int(%.loc15_13.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_13.2: type = value_of_initializer %int.make_type_signed.loc15_13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_13.3: type = converted %int.make_type_signed.loc15_13, %.loc15_13.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15_22: init type = call constants.%Int(%.loc15_22.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_22.2: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_22.3: type = converted %int.make_type_signed.loc15_22, %.loc15_22.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_25: type = struct_type {.a: %.2, .b: %.2} [template = constants.%.3]
// CHECK:STDOUT:   %y.var: ref %.3 = var y
// CHECK:STDOUT:   %y: ref %.3 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_35: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_43: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_44.1: %.7 = struct_literal (%.loc11_35, %.loc11_43)
// CHECK:STDOUT:   %.loc11_44.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_44.3: <bound method> = bound_method %.loc11_35, %.loc11_44.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_44.4: <specific function> = specific_function %.loc11_44.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.1: init %.2 = call %.loc11_44.4(%.loc11_35) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_44.5: %.2 = value_of_initializer %int.convert_checked.loc11_44.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_44.6: %.2 = converted %.loc11_35, %.loc11_44.5 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_44.7: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_44.8: <bound method> = bound_method %.loc11_43, %.loc11_44.7 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_44.9: <specific function> = specific_function %.loc11_44.8, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.2: init %.2 = call %.loc11_44.9(%.loc11_43) [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_44.10: %.2 = value_of_initializer %int.convert_checked.loc11_44.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_44.11: %.2 = converted %.loc11_43, %.loc11_44.10 [template = constants.%.35]
// CHECK:STDOUT:   %struct: %.3 = struct_value (%.loc11_44.6, %.loc11_44.11) [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_45: %.3 = converted %.loc11_44.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %v: %.3 = bind_name v, %.loc11_45
// CHECK:STDOUT:   %v.ref: %.3 = name_ref v, %v
// CHECK:STDOUT:   %w: %.3 = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc14_35: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_43: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_44.1: %.7 = struct_literal (%.loc14_35, %.loc14_43)
// CHECK:STDOUT:   %.loc14_44.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_44.3: <bound method> = bound_method %.loc14_35, %.loc14_44.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_44.4: <specific function> = specific_function %.loc14_44.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.1: init %.2 = call %.loc14_44.4(%.loc14_35) [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_44.5: init %.2 = converted %.loc14_35, %int.convert_checked.loc14_44.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_44.6: ref %.2 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_44.7: init %.2 = initialize_from %.loc14_44.5 to %.loc14_44.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_44.8: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_44.9: <bound method> = bound_method %.loc14_43, %.loc14_44.8 [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_44.10: <specific function> = specific_function %.loc14_44.9, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.2: init %.2 = call %.loc14_44.10(%.loc14_43) [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_44.11: init %.2 = converted %.loc14_43, %int.convert_checked.loc14_44.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_44.12: ref %.2 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_44.13: init %.2 = initialize_from %.loc14_44.11 to %.loc14_44.12 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_44.14: init %.3 = struct_init (%.loc14_44.7, %.loc14_44.13) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_45: init %.3 = converted %.loc14_44.1, %.loc14_44.14 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_45
// CHECK:STDOUT:   %x.ref: ref %.3 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_29.1: ref %.2 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_29.2: %.2 = bind_value %.loc15_29.1
// CHECK:STDOUT:   %.loc15_29.3: ref %.2 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_29.4: init %.2 = initialize_from %.loc15_29.2 to %.loc15_29.3
// CHECK:STDOUT:   %.loc15_29.5: ref %.2 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_29.6: %.2 = bind_value %.loc15_29.5
// CHECK:STDOUT:   %.loc15_29.7: ref %.2 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_29.8: init %.2 = initialize_from %.loc15_29.6 to %.loc15_29.7
// CHECK:STDOUT:   %.loc15_29.9: init %.3 = struct_init (%.loc15_29.4, %.loc15_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_30: init %.3 = converted %x.ref, %.loc15_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

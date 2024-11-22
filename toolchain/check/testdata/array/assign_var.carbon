// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_var.carbon

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.32: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.32, %.35, %.38) [template]
// CHECK:STDOUT:   %.39: type = array_type %.5, %i32 [template]
// CHECK:STDOUT:   %.41: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_9: init type = call constants.%Int(%.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_14: init type = call constants.%Int(%.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_19: init type = call constants.%Int(%.loc11_19) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_9, %int.make_type_signed.loc11_14, %int.make_type_signed.loc11_19)
// CHECK:STDOUT:   %.loc11_22.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.3: type = converted %int.make_type_signed.loc11_9, %.loc11_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.4: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.5: type = converted %int.make_type_signed.loc11_14, %.loc11_22.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.6: type = value_of_initializer %int.make_type_signed.loc11_19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.7: type = converted %int.make_type_signed.loc11_19, %.loc11_22.6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.8: type = converted %.loc11_22.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_9.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_9.3: type = converted %int.make_type_signed.loc12, %.loc12_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15: type = array_type %.loc12_14, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %b.var: ref %.39 = var b
// CHECK:STDOUT:   %b: ref %.39 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_27: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_30: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_33: Core.IntLiteral = int_value 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_34.1: %tuple.type.3 = tuple_literal (%.loc11_27, %.loc11_30, %.loc11_33)
// CHECK:STDOUT:   %.loc11_34.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_34.3: <bound method> = bound_method %.loc11_27, %.loc11_34.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_34.4: <specific function> = specific_function %.loc11_34.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.1: init %i32 = call %.loc11_34.4(%.loc11_27) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_34.5: init %i32 = converted %.loc11_27, %int.convert_checked.loc11_34.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_34.6: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_34.7: init %i32 = initialize_from %.loc11_34.5 to %.loc11_34.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_34.8: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_34.9: <bound method> = bound_method %.loc11_30, %.loc11_34.8 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_34.10: <specific function> = specific_function %.loc11_34.9, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.2: init %i32 = call %.loc11_34.10(%.loc11_30) [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_34.11: init %i32 = converted %.loc11_30, %int.convert_checked.loc11_34.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_34.12: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_34.13: init %i32 = initialize_from %.loc11_34.11 to %.loc11_34.12 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_34.14: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_34.15: <bound method> = bound_method %.loc11_33, %.loc11_34.14 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_34.16: <specific function> = specific_function %.loc11_34.15, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.3: init %i32 = call %.loc11_34.16(%.loc11_33) [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_34.17: init %i32 = converted %.loc11_33, %int.convert_checked.loc11_34.3 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_34.18: ref %i32 = tuple_access file.%a.var, element2
// CHECK:STDOUT:   %.loc11_34.19: init %i32 = initialize_from %.loc11_34.17 to %.loc11_34.18 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_34.20: init %tuple.type.2 = tuple_init (%.loc11_34.7, %.loc11_34.13, %.loc11_34.19) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_35: init %tuple.type.2 = converted %.loc11_34.1, %.loc11_34.20 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_35
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_19.1: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_19.2: %i32 = bind_value %.loc12_19.1
// CHECK:STDOUT:   %.loc12_19.3: Core.IntLiteral = int_value 0 [template = constants.%.41]
// CHECK:STDOUT:   %.loc12_19.4: ref %i32 = array_index file.%b.var, %.loc12_19.3
// CHECK:STDOUT:   %.loc12_19.5: init %i32 = initialize_from %.loc12_19.2 to %.loc12_19.4
// CHECK:STDOUT:   %.loc12_19.6: ref %i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc12_19.7: %i32 = bind_value %.loc12_19.6
// CHECK:STDOUT:   %.loc12_19.8: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_19.9: ref %i32 = array_index file.%b.var, %.loc12_19.8
// CHECK:STDOUT:   %.loc12_19.10: init %i32 = initialize_from %.loc12_19.7 to %.loc12_19.9
// CHECK:STDOUT:   %.loc12_19.11: ref %i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc12_19.12: %i32 = bind_value %.loc12_19.11
// CHECK:STDOUT:   %.loc12_19.13: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_19.14: ref %i32 = array_index file.%b.var, %.loc12_19.13
// CHECK:STDOUT:   %.loc12_19.15: init %i32 = initialize_from %.loc12_19.12 to %.loc12_19.14
// CHECK:STDOUT:   %.loc12_19.16: init %.39 = array_init (%.loc12_19.5, %.loc12_19.10, %.loc12_19.15) to file.%b.var
// CHECK:STDOUT:   %.loc12_20: init %.39 = converted %a.ref, %.loc12_19.16
// CHECK:STDOUT:   assign file.%b.var, %.loc12_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

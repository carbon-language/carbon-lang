// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/array_element_access.carbon

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: --- array_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 24 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %.35: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 24 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.34, %.38) [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.35, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.43: <specific function> = specific_function %.42, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.44: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.3: type = converted %int.make_type_signed.loc11, %.loc11_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_signed.loc12, %.loc12_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc13_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.3: type = converted %int.make_type_signed.loc13, %.loc13_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc14_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%.loc14_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8.2: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8.3: type = converted %int.make_type_signed.loc14, %.loc14_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_24: Core.IntLiteral = int_value 24 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_24)
// CHECK:STDOUT:   %.loc11_26.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_26.3: <bound method> = bound_method %.loc11_20, %.loc11_26.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_26.4: <specific function> = specific_function %.loc11_26.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.1: init %i32 = call %.loc11_26.4(%.loc11_20) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_26.5: init %i32 = converted %.loc11_20, %int.convert_checked.loc11_26.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_26.6: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_26.7: ref %i32 = array_index file.%a.var, %.loc11_26.6
// CHECK:STDOUT:   %.loc11_26.8: init %i32 = initialize_from %.loc11_26.5 to %.loc11_26.7 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_26.9: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_26.10: <bound method> = bound_method %.loc11_24, %.loc11_26.9 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_26.11: <specific function> = specific_function %.loc11_26.10, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.2: init %i32 = call %.loc11_26.11(%.loc11_24) [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_26.12: init %i32 = converted %.loc11_24, %int.convert_checked.loc11_26.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_26.13: Core.IntLiteral = int_value 1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_26.14: ref %i32 = array_index file.%a.var, %.loc11_26.13
// CHECK:STDOUT:   %.loc11_26.15: init %i32 = initialize_from %.loc11_26.12 to %.loc11_26.14 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_26.16: init %.3 = array_init (%.loc11_26.8, %.loc11_26.15) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %.3 = converted %.loc11_26.1, %.loc11_26.16 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_27
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc12_15.1: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_15.2: <bound method> = bound_method %.loc12_14, %.loc12_15.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc12_15.3: <specific function> = specific_function %.loc12_15.2, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_15.3(%.loc12_14) [template = constants.%.41]
// CHECK:STDOUT:   %.loc12_15.4: init %i32 = converted %.loc12_14, %int.convert_checked.loc12 [template = constants.%.41]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15.4
// CHECK:STDOUT:   %a.ref.loc13: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc13_16.1: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_16.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_16.3: <bound method> = bound_method %.loc13_16.1, %.loc13_16.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc13_16.4: <specific function> = specific_function %.loc13_16.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %.loc13_16.4(%.loc13_16.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc13_16.5: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%.44]
// CHECK:STDOUT:   %.loc13_16.6: %i32 = converted %.loc13_16.1, %.loc13_16.5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc13_17.1: ref %i32 = array_index %a.ref.loc13, %.loc13_16.6
// CHECK:STDOUT:   %.loc13_17.2: %i32 = bind_value %.loc13_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_17.2
// CHECK:STDOUT:   %a.ref.loc14: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: ref %i32 = name_ref b, file.%b
// CHECK:STDOUT:   %.loc14_16: %i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc14_17.1: ref %i32 = array_index %a.ref.loc14, %.loc14_16
// CHECK:STDOUT:   %.loc14_17.2: %i32 = bind_value %.loc14_17.1
// CHECK:STDOUT:   assign file.%d.var, %.loc14_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

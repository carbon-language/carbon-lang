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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 24 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 12 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 24 [template]
// CHECK:STDOUT:   %array: %.2 = tuple_value (%.32, %.35) [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.36, %Convert.15 [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.38, %Convert.15 [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 2 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_32.loc12, %.loc12_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_8.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_8.2: type = converted %int.make_type_32.loc14, %.loc14_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_24: Core.IntLiteral = int_value 24 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_24)
// CHECK:STDOUT:   %.loc11_26.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_26.3: <bound method> = bound_method %.loc11_20, %.loc11_26.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.1: init i32 = call %.loc11_26.3(%.loc11_20) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_26.4: init i32 = converted %.loc11_20, %int.convert_checked.loc11_26.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_26.5: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26.6: ref i32 = array_index file.%a.var, %.loc11_26.5
// CHECK:STDOUT:   %.loc11_26.7: init i32 = initialize_from %.loc11_26.4 to %.loc11_26.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_26.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_26.9: <bound method> = bound_method %.loc11_24, %.loc11_26.8 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.2: init i32 = call %.loc11_26.9(%.loc11_24) [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_26.10: init i32 = converted %.loc11_24, %int.convert_checked.loc11_26.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_26.11: i32 = int_value 1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_26.12: ref i32 = array_index file.%a.var, %.loc11_26.11
// CHECK:STDOUT:   %.loc11_26.13: init i32 = initialize_from %.loc11_26.10 to %.loc11_26.12 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_26.14: init %.2 = array_init (%.loc11_26.7, %.loc11_26.13) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %.2 = converted %.loc11_26.1, %.loc11_26.14 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_27
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_15.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_15.2: <bound method> = bound_method %.loc12_14, %.loc12_15.1 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc12: init i32 = call %.loc12_15.2(%.loc12_14) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_15.3: init i32 = converted %.loc12_14, %int.convert_checked.loc12 [template = constants.%.33]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15.3
// CHECK:STDOUT:   %a.ref.loc13: ref %.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc13_16.1: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc13_16.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_16.3: <bound method> = bound_method %.loc13_16.1, %.loc13_16.2 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc13: init i32 = call %.loc13_16.3(%.loc13_16.1) [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_16.4: i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_16.5: i32 = converted %.loc13_16.1, %.loc13_16.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_17.1: ref i32 = array_index %a.ref.loc13, %.loc13_16.5
// CHECK:STDOUT:   %.loc13_17.2: i32 = bind_value %.loc13_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_17.2
// CHECK:STDOUT:   %a.ref.loc14: ref %.2 = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: ref i32 = name_ref b, file.%b
// CHECK:STDOUT:   %.loc14_16: i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc14_17.1: ref i32 = array_index %a.ref.loc14, %.loc14_16
// CHECK:STDOUT:   %.loc14_17.2: i32 = bind_value %.loc14_17.1
// CHECK:STDOUT:   assign file.%d.var, %.loc14_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

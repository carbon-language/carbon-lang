// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/index_not_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/index_not_literal.carbon

var a: [i32; 3] = (1, 2, 3);
var b: i32 = a[{.index = 2}.index];

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.2 = tuple_value (%.32, %.34, %.36) [template]
// CHECK:STDOUT:   %.37: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %.37 = struct_value (%.5) [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 3 [template = constants.%.1]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_27.3: <bound method> = bound_method %.loc11_20, %.loc11_27.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init i32 = call %.loc11_27.3(%.loc11_20) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_27.4: init i32 = converted %.loc11_20, %int.convert_checked.loc11_27.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_27.5: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_27.6: ref i32 = array_index file.%a.var, %.loc11_27.5
// CHECK:STDOUT:   %.loc11_27.7: init i32 = initialize_from %.loc11_27.4 to %.loc11_27.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_27.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_27.9: <bound method> = bound_method %.loc11_23, %.loc11_27.8 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init i32 = call %.loc11_27.9(%.loc11_23) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_27.10: init i32 = converted %.loc11_23, %int.convert_checked.loc11_27.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_27.11: i32 = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_27.12: ref i32 = array_index file.%a.var, %.loc11_27.11
// CHECK:STDOUT:   %.loc11_27.13: init i32 = initialize_from %.loc11_27.10 to %.loc11_27.12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_27.14: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_27.15: <bound method> = bound_method %.loc11_26, %.loc11_27.14 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.3: init i32 = call %.loc11_27.15(%.loc11_26) [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_27.16: init i32 = converted %.loc11_26, %int.convert_checked.loc11_27.3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_27.17: i32 = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_27.18: ref i32 = array_index file.%a.var, %.loc11_27.17
// CHECK:STDOUT:   %.loc11_27.19: init i32 = initialize_from %.loc11_27.16 to %.loc11_27.18 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_27.20: init %.2 = array_init (%.loc11_27.7, %.loc11_27.13, %.loc11_27.19) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_28: init %.2 = converted %.loc11_27.1, %.loc11_27.20 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_26: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_27.1: %.37 = struct_literal (%.loc12_26)
// CHECK:STDOUT:   %struct: %.37 = struct_value (%.loc12_26) [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_27.2: %.37 = converted %.loc12_27.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_28.1: Core.IntLiteral = struct_access %.loc12_27.2, element0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_28.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_28.3: <bound method> = bound_method %.loc12_28.1, %.loc12_28.2 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc12: init i32 = call %.loc12_28.3(%.loc12_28.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_28.4: i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_28.5: i32 = converted %.loc12_28.1, %.loc12_28.4 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_34.1: ref i32 = array_index %a.ref, %.loc12_28.5
// CHECK:STDOUT:   %.loc12_34.2: i32 = bind_value %.loc12_34.1
// CHECK:STDOUT:   assign file.%b.var, %.loc12_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

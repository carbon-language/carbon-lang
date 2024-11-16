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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.31, %.33) [template]
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
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_32.loc11_13 [template = i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_32.loc11_13, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_22.1: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:   %.loc11_22.2: type = converted %int.make_type_32.loc11_22, %.loc11_22.1 [template = i32]
// CHECK:STDOUT:   %.loc11_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_32.loc12_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_13.1: type = value_of_initializer %int.make_type_32.loc12_13 [template = i32]
// CHECK:STDOUT:   %.loc12_13.2: type = converted %int.make_type_32.loc12_13, %.loc12_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc12_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_32.loc12_22 [template = i32]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_32.loc12_22, %.loc12_22.1 [template = i32]
// CHECK:STDOUT:   %.loc12_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_32.loc14_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_32.loc14_13 [template = i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_32.loc14_13, %.loc14_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_22.1: type = value_of_initializer %int.make_type_32.loc14_22 [template = i32]
// CHECK:STDOUT:   %.loc14_22.2: type = converted %int.make_type_32.loc14_22, %.loc14_22.1 [template = i32]
// CHECK:STDOUT:   %.loc14_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc15_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_13.1: type = value_of_initializer %int.make_type_32.loc15_13 [template = i32]
// CHECK:STDOUT:   %.loc15_13.2: type = converted %int.make_type_32.loc15_13, %.loc15_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc15_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_22.1: type = value_of_initializer %int.make_type_32.loc15_22 [template = i32]
// CHECK:STDOUT:   %.loc15_22.2: type = converted %int.make_type_32.loc15_22, %.loc15_22.1 [template = i32]
// CHECK:STDOUT:   %.loc15_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref %.1 = var y
// CHECK:STDOUT:   %y: ref %.1 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_35: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_43: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_44.1: %.5 = struct_literal (%.loc11_35, %.loc11_43)
// CHECK:STDOUT:   %.loc11_44.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_44.3: <bound method> = bound_method %.loc11_35, %.loc11_44.2 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.1: init i32 = call %.loc11_44.3(%.loc11_35) [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_44.4: i32 = value_of_initializer %int.convert_checked.loc11_44.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_44.5: i32 = converted %.loc11_35, %.loc11_44.4 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_44.6: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_44.7: <bound method> = bound_method %.loc11_43, %.loc11_44.6 [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.2: init i32 = call %.loc11_44.7(%.loc11_43) [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_44.8: i32 = value_of_initializer %int.convert_checked.loc11_44.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_44.9: i32 = converted %.loc11_43, %.loc11_44.8 [template = constants.%.33]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.loc11_44.5, %.loc11_44.9) [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_45: %.1 = converted %.loc11_44.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %v: %.1 = bind_name v, %.loc11_45
// CHECK:STDOUT:   %v.ref: %.1 = name_ref v, %v
// CHECK:STDOUT:   %w: %.1 = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc14_35: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_43: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_44.1: %.5 = struct_literal (%.loc14_35, %.loc14_43)
// CHECK:STDOUT:   %.loc14_44.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_44.3: <bound method> = bound_method %.loc14_35, %.loc14_44.2 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.1: init i32 = call %.loc14_44.3(%.loc14_35) [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_44.4: init i32 = converted %.loc14_35, %int.convert_checked.loc14_44.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_44.5: ref i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_44.6: init i32 = initialize_from %.loc14_44.4 to %.loc14_44.5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_44.7: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_44.8: <bound method> = bound_method %.loc14_43, %.loc14_44.7 [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.2: init i32 = call %.loc14_44.8(%.loc14_43) [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_44.9: init i32 = converted %.loc14_43, %int.convert_checked.loc14_44.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_44.10: ref i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_44.11: init i32 = initialize_from %.loc14_44.9 to %.loc14_44.10 [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_44.12: init %.1 = struct_init (%.loc14_44.6, %.loc14_44.11) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_45: init %.1 = converted %.loc14_44.1, %.loc14_44.12 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_45
// CHECK:STDOUT:   %x.ref: ref %.1 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_29.1: ref i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_29.2: i32 = bind_value %.loc15_29.1
// CHECK:STDOUT:   %.loc15_29.3: ref i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_29.4: init i32 = initialize_from %.loc15_29.2 to %.loc15_29.3
// CHECK:STDOUT:   %.loc15_29.5: ref i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_29.6: i32 = bind_value %.loc15_29.5
// CHECK:STDOUT:   %.loc15_29.7: ref i32 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_29.8: init i32 = initialize_from %.loc15_29.6 to %.loc15_29.7
// CHECK:STDOUT:   %.loc15_29.9: init %.1 = struct_init (%.loc15_29.4, %.loc15_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_30: init %.1 = converted %x.ref, %.loc15_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

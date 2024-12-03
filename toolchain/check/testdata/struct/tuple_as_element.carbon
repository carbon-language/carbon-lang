// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/tuple_as_element.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/tuple_as_element.carbon

var x: {.a: i32, .b: (i32,)} = {.a = 1, .b = (2,)};
var y: {.a: i32, .b: (i32,)} = x;

// CHECK:STDOUT: --- tuple_as_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: %i32, .b: %tuple.type.2} [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: Core.IntLiteral, .b: %tuple.type.3} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.2.2) [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%int.1.2, %tuple) [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32.loc11_13: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_13: init type = call constants.%Int(%int.32.loc11_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_signed.loc11_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_signed.loc11_13, %.loc11_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc11_23: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_23: init type = call constants.%Int(%int.32.loc11_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_23)
// CHECK:STDOUT:   %.loc11_27.2: type = value_of_initializer %int.make_type_signed.loc11_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_27.3: type = converted %int.make_type_signed.loc11_23, %.loc11_27.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_27.4: type = converted %.loc11_27.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc11_28: type = struct_type {.a: %i32, .b: %tuple.type.2} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT:   %int.32.loc12_13: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_13: init type = call constants.%Int(%int.32.loc12_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.1: type = value_of_initializer %int.make_type_signed.loc12_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.2: type = converted %int.make_type_signed.loc12_13, %.loc12_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc12_23: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_23: init type = call constants.%Int(%int.32.loc12_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_27.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12_23)
// CHECK:STDOUT:   %.loc12_27.2: type = value_of_initializer %int.make_type_signed.loc12_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_27.3: type = converted %int.make_type_signed.loc12_23, %.loc12_27.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_27.4: type = converted %.loc12_27.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc12_28: type = struct_type {.a: %i32, .b: %tuple.type.2} [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref %.1 = var y
// CHECK:STDOUT:   %y: ref %.1 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc11_49.1: %tuple.type.3 = tuple_literal (%int.2)
// CHECK:STDOUT:   %.loc11_50.1: %.3 = struct_literal (%int.1, %.loc11_49.1)
// CHECK:STDOUT:   %.loc11_50.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_50.3: <bound method> = bound_method %int.1, %.loc11_50.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc11_50.4: <specific function> = specific_function %.loc11_50.3, @Convert.2(constants.%int.32) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc11_50: init %i32 = call %.loc11_50.4(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_50.5: init %i32 = converted %int.1, %int.convert_checked.loc11_50 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_50.6: ref %i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_50.7: init %i32 = initialize_from %.loc11_50.5 to %.loc11_50.6 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_49.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_49.3: <bound method> = bound_method %int.2, %.loc11_49.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_49.4: <specific function> = specific_function %.loc11_49.3, @Convert.2(constants.%int.32) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc11_49: init %i32 = call %.loc11_49.4(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_49.5: init %i32 = converted %int.2, %int.convert_checked.loc11_49 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_50.8: ref %tuple.type.2 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_49.6: init %tuple.type.2 = tuple_init (%.loc11_49.5) to %.loc11_50.8 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_50.9: init %tuple.type.2 = converted %.loc11_49.1, %.loc11_49.6 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_50.10: init %tuple.type.2 = initialize_from %.loc11_50.9 to %.loc11_50.8 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_50.11: init %.1 = struct_init (%.loc11_50.7, %.loc11_50.10) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_51: init %.1 = converted %.loc11_50.1, %.loc11_50.11 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_51
// CHECK:STDOUT:   %x.ref: ref %.1 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_32.1: ref %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc12_32.2: %i32 = bind_value %.loc12_32.1
// CHECK:STDOUT:   %.loc12_32.3: ref %i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc12_32.4: init %i32 = initialize_from %.loc12_32.2 to %.loc12_32.3
// CHECK:STDOUT:   %.loc12_32.5: ref %tuple.type.2 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc12_32.6: ref %i32 = tuple_access %.loc12_32.5, element0
// CHECK:STDOUT:   %.loc12_32.7: %i32 = bind_value %.loc12_32.6
// CHECK:STDOUT:   %.loc12_32.8: ref %tuple.type.2 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc12_32.9: init %tuple.type.2 = tuple_init (%.loc12_32.7) to %.loc12_32.8
// CHECK:STDOUT:   %.loc12_32.10: init %tuple.type.2 = converted %.loc12_32.5, %.loc12_32.9
// CHECK:STDOUT:   %.loc12_32.11: init %tuple.type.2 = initialize_from %.loc12_32.10 to %.loc12_32.8
// CHECK:STDOUT:   %.loc12_32.12: init %.1 = struct_init (%.loc12_32.4, %.loc12_32.11) to file.%y.var
// CHECK:STDOUT:   %.loc12_33: init %.1 = converted %x.ref, %.loc12_32.12
// CHECK:STDOUT:   assign file.%y.var, %.loc12_33
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon

var a: [i32; 3] = (1, 2, 3);
// CHECK:STDERR: fail_out_of_bound_non_literal.carbon:[[@LINE+3]]:16: error: array index `3` is past the end of type `[i32; 3]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a[{.index = 3}.index];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
var b: i32 = a[{.index = 3}.index];

// CHECK:STDOUT: --- fail_out_of_bound_non_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.3.1, %i32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.1 = tuple_value (%int.1.2, %int.2.2, %int.3.2) [template]
// CHECK:STDOUT:   %struct_type.index: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int.3.1) [template]
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
// CHECK:STDOUT:   %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %int.3, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int.32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_signed.loc15, %.loc15_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %int.3.loc11: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type = tuple_literal (%int.1.loc11_20, %int.2.loc11_23, %int.3.loc11)
// CHECK:STDOUT:   %.loc11_27.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_27.3: <bound method> = bound_method %int.1.loc11_20, %.loc11_27.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_27.4: <specific function> = specific_function %.loc11_27.3, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %i32 = call %.loc11_27.4(%int.1.loc11_20) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = converted %int.1.loc11_20, %int.convert_checked.loc11_27.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc11_27.6: ref %i32 = array_index file.%a.var, %int.0
// CHECK:STDOUT:   %.loc11_27.7: init %i32 = initialize_from %.loc11_27.5 to %.loc11_27.6 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_27.8: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_27.9: <bound method> = bound_method %int.2.loc11_23, %.loc11_27.8 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_27.10: <specific function> = specific_function %.loc11_27.9, @Convert.2(constants.%int.32) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %i32 = call %.loc11_27.10(%int.2.loc11_23) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_27.11: init %i32 = converted %int.2.loc11_23, %int.convert_checked.loc11_27.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %int.1.loc11_27: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc11_27.12: ref %i32 = array_index file.%a.var, %int.1.loc11_27
// CHECK:STDOUT:   %.loc11_27.13: init %i32 = initialize_from %.loc11_27.11 to %.loc11_27.12 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_27.14: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_27.15: <bound method> = bound_method %int.3.loc11, %.loc11_27.14 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_27.16: <specific function> = specific_function %.loc11_27.15, @Convert.2(constants.%int.32) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.3: init %i32 = call %.loc11_27.16(%int.3.loc11) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_27.17: init %i32 = converted %int.3.loc11, %int.convert_checked.loc11_27.3 [template = constants.%int.3.2]
// CHECK:STDOUT:   %int.2.loc11_27: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc11_27.18: ref %i32 = array_index file.%a.var, %int.2.loc11_27
// CHECK:STDOUT:   %.loc11_27.19: init %i32 = initialize_from %.loc11_27.17 to %.loc11_27.18 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_27.20: init %.1 = array_init (%.loc11_27.7, %.loc11_27.13, %.loc11_27.19) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_28: init %.1 = converted %.loc11_27.1, %.loc11_27.20 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %.1 = name_ref a, file.%a
// CHECK:STDOUT:   %int.3.loc15: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc15_27.1: %struct_type.index = struct_literal (%int.3.loc15)
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int.3.loc15) [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_27.2: %struct_type.index = converted %.loc15_27.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_28.1: Core.IntLiteral = struct_access %.loc15_27.2, element0 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc15_28.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_28.3: <bound method> = bound_method %.loc15_28.1, %.loc15_28.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_28.4: <specific function> = specific_function %.loc15_28.3, @Convert.2(constants.%int.32) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %.loc15_28.4(%.loc15_28.1) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc15_28.5: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc15_28.6: %i32 = converted %.loc15_28.1, %.loc15_28.5 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc15_34.1: ref %i32 = array_index %a.ref, %.loc15_28.6 [template = <error>]
// CHECK:STDOUT:   %.loc15_34.2: %i32 = bind_value %.loc15_34.1
// CHECK:STDOUT:   assign file.%b.var, %.loc15_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

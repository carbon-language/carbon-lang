// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/nine_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/nine_elements.carbon

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: --- nine_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_9.1, %i32 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %int_7.1: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_8.1: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_4.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_6.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_7.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_8.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int_9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.2, %int_2.2, %int_3.2, %int_4.2, %int_5.2, %int_6.2, %int_7.2, %int_8.2, %int_9.2) [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_9, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_3.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_4.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %int_5.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_6.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %int_7.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%int_7.1]
// CHECK:STDOUT:   %int_8.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.1]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%int_1.loc11_20, %int_2.loc11_23, %int_3.loc11_26, %int_4.loc11_29, %int_5.loc11_32, %int_6.loc11_35, %int_7.loc11_38, %int_8.loc11_41, %int_9)
// CHECK:STDOUT:   %impl.elem0.loc11_45.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.1: <bound method> = bound_method %int_1.loc11_20, %impl.elem0.loc11_45.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.1: <specific function> = specific_function %Convert.bound.loc11_45.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %Convert.specific_fn.loc11_45.1(%int_1.loc11_20) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_45.2: init %i32 = converted %int_1.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_45.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_45.4: init %i32 = initialize_from %.loc11_45.2 to %.loc11_45.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.2: <bound method> = bound_method %int_2.loc11_23, %impl.elem0.loc11_45.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.2: <specific function> = specific_function %Convert.bound.loc11_45.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %Convert.specific_fn.loc11_45.2(%int_2.loc11_23) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = converted %int_2.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int_1.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc11_45.6: ref %i32 = array_index file.%a.var, %int_1.loc11_45
// CHECK:STDOUT:   %.loc11_45.7: init %i32 = initialize_from %.loc11_45.5 to %.loc11_45.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.3: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.3: <bound method> = bound_method %int_3.loc11_26, %impl.elem0.loc11_45.3 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.3: <specific function> = specific_function %Convert.bound.loc11_45.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %Convert.specific_fn.loc11_45.3(%int_3.loc11_26) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc11_45.8: init %i32 = converted %int_3.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %int_2.loc11_45: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc11_45.9: ref %i32 = array_index file.%a.var, %int_2.loc11_45
// CHECK:STDOUT:   %.loc11_45.10: init %i32 = initialize_from %.loc11_45.8 to %.loc11_45.9 [template = constants.%int_3.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.4: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.4: <bound method> = bound_method %int_4.loc11_29, %impl.elem0.loc11_45.4 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.4: <specific function> = specific_function %Convert.bound.loc11_45.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %Convert.specific_fn.loc11_45.4(%int_4.loc11_29) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = converted %int_4.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%int_4.2]
// CHECK:STDOUT:   %int_3.loc11_45: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc11_45.12: ref %i32 = array_index file.%a.var, %int_3.loc11_45
// CHECK:STDOUT:   %.loc11_45.13: init %i32 = initialize_from %.loc11_45.11 to %.loc11_45.12 [template = constants.%int_4.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.5: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.5: <bound method> = bound_method %int_5.loc11_32, %impl.elem0.loc11_45.5 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.5: <specific function> = specific_function %Convert.bound.loc11_45.5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %Convert.specific_fn.loc11_45.5(%int_5.loc11_32) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc11_45.14: init %i32 = converted %int_5.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %int_4.loc11_45: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %.loc11_45.15: ref %i32 = array_index file.%a.var, %int_4.loc11_45
// CHECK:STDOUT:   %.loc11_45.16: init %i32 = initialize_from %.loc11_45.14 to %.loc11_45.15 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.6: <bound method> = bound_method %int_6.loc11_35, %impl.elem0.loc11_45.6 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.6: <specific function> = specific_function %Convert.bound.loc11_45.6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %Convert.specific_fn.loc11_45.6(%int_6.loc11_35) [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_45.17: init %i32 = converted %int_6.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%int_6.2]
// CHECK:STDOUT:   %int_5.loc11_45: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc11_45.18: ref %i32 = array_index file.%a.var, %int_5.loc11_45
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = initialize_from %.loc11_45.17 to %.loc11_45.18 [template = constants.%int_6.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.7: <bound method> = bound_method %int_7.loc11_38, %impl.elem0.loc11_45.7 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.7: <specific function> = specific_function %Convert.bound.loc11_45.7, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %Convert.specific_fn.loc11_45.7(%int_7.loc11_38) [template = constants.%int_7.2]
// CHECK:STDOUT:   %.loc11_45.20: init %i32 = converted %int_7.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%int_7.2]
// CHECK:STDOUT:   %int_6.loc11_45: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc11_45.21: ref %i32 = array_index file.%a.var, %int_6.loc11_45
// CHECK:STDOUT:   %.loc11_45.22: init %i32 = initialize_from %.loc11_45.20 to %.loc11_45.21 [template = constants.%int_7.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.8: <bound method> = bound_method %int_8.loc11_41, %impl.elem0.loc11_45.8 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.8: <specific function> = specific_function %Convert.bound.loc11_45.8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %Convert.specific_fn.loc11_45.8(%int_8.loc11_41) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc11_45.23: init %i32 = converted %int_8.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%int_8.2]
// CHECK:STDOUT:   %int_7.loc11_45: Core.IntLiteral = int_value 7 [template = constants.%int_7.1]
// CHECK:STDOUT:   %.loc11_45.24: ref %i32 = array_index file.%a.var, %int_7.loc11_45
// CHECK:STDOUT:   %.loc11_45.25: init %i32 = initialize_from %.loc11_45.23 to %.loc11_45.24 [template = constants.%int_8.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.9: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.9: <bound method> = bound_method %int_9, %impl.elem0.loc11_45.9 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.9: <specific function> = specific_function %Convert.bound.loc11_45.9, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %Convert.specific_fn.loc11_45.9(%int_9) [template = constants.%int_9.2]
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = converted %int_9, %int.convert_checked.loc11_45.9 [template = constants.%int_9.2]
// CHECK:STDOUT:   %int_8.loc11_45: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %.loc11_45.27: ref %i32 = array_index file.%a.var, %int_8.loc11_45
// CHECK:STDOUT:   %.loc11_45.28: init %i32 = initialize_from %.loc11_45.26 to %.loc11_45.27 [template = constants.%int_9.2]
// CHECK:STDOUT:   %.loc11_45.29: init %array_type = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %array_type = converted %.loc11_45.1, %.loc11_45.29 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

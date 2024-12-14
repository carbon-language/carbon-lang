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
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_9, %i32 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3), @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.3(%int_3) [symbolic]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_4, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %iN = call %Convert.specific_fn.4(%int_4) [symbolic]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_5, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.5: init %iN = call %Convert.specific_fn.5(%int_5) [symbolic]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_6, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.6: init %iN = call %Convert.specific_fn.6(%int_6) [symbolic]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_7, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.7: init %iN = call %Convert.specific_fn.7(%int_7) [symbolic]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_8, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.8: init %iN = call %Convert.specific_fn.8(%int_8) [symbolic]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int_9, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.9: init %iN = call %Convert.specific_fn.9(%int_9) [symbolic]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int.convert_checked.1, %int.convert_checked.2, %int.convert_checked.3, %int.convert_checked.4, %int.convert_checked.5, %int.convert_checked.6, %int.convert_checked.7, %int.convert_checked.8, %int.convert_checked.9) [symbolic]
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
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_9, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_4.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %int_5.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_6.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %int_7.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%int_7]
// CHECK:STDOUT:   %int_8.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%int_1.loc11_20, %int_2.loc11_23, %int_3.loc11_26, %int_4.loc11_29, %int_5.loc11_32, %int_6.loc11_35, %int_7.loc11_38, %int_8.loc11_41, %int_9)
// CHECK:STDOUT:   %impl.elem0.loc11_45.1: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.1: <bound method> = bound_method %int_1.loc11_20, %impl.elem0.loc11_45.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.1: <specific function> = specific_function %Convert.bound.loc11_45.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %iN = call %Convert.specific_fn.loc11_45.1(%int_1.loc11_20) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_45.2: init %i32 = converted %int_1.loc11_20, %int.convert_checked.loc11_45.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_45.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_45.4: init %i32 = initialize_from %.loc11_45.2 to %.loc11_45.3 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc11_45.2: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.2: <bound method> = bound_method %int_2.loc11_23, %impl.elem0.loc11_45.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.2: <specific function> = specific_function %Convert.bound.loc11_45.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %iN = call %Convert.specific_fn.loc11_45.2(%int_2.loc11_23) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = converted %int_2.loc11_23, %int.convert_checked.loc11_45.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int_1.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc11_45.6: ref %i32 = array_index file.%a.var, %int_1.loc11_45
// CHECK:STDOUT:   %.loc11_45.7: init %i32 = initialize_from %.loc11_45.5 to %.loc11_45.6 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.3: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.3: <bound method> = bound_method %int_3.loc11_26, %impl.elem0.loc11_45.3 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.3: <specific function> = specific_function %Convert.bound.loc11_45.3, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %iN = call %Convert.specific_fn.loc11_45.3(%int_3.loc11_26) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc11_45.8: init %i32 = converted %int_3.loc11_26, %int.convert_checked.loc11_45.3 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int_2.loc11_45: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc11_45.9: ref %i32 = array_index file.%a.var, %int_2.loc11_45
// CHECK:STDOUT:   %.loc11_45.10: init %i32 = initialize_from %.loc11_45.8 to %.loc11_45.9 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %impl.elem0.loc11_45.4: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.4: <bound method> = bound_method %int_4.loc11_29, %impl.elem0.loc11_45.4 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.4: <specific function> = specific_function %Convert.bound.loc11_45.4, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %iN = call %Convert.specific_fn.loc11_45.4(%int_4.loc11_29) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = converted %int_4.loc11_29, %int.convert_checked.loc11_45.4 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int_3.loc11_45: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc11_45.12: ref %i32 = array_index file.%a.var, %int_3.loc11_45
// CHECK:STDOUT:   %.loc11_45.13: init %i32 = initialize_from %.loc11_45.11 to %.loc11_45.12 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %impl.elem0.loc11_45.5: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.5: <bound method> = bound_method %int_5.loc11_32, %impl.elem0.loc11_45.5 [symbolic = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.5: <specific function> = specific_function %Convert.bound.loc11_45.5, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %iN = call %Convert.specific_fn.loc11_45.5(%int_5.loc11_32) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc11_45.14: init %i32 = converted %int_5.loc11_32, %int.convert_checked.loc11_45.5 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int_4.loc11_45: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc11_45.15: ref %i32 = array_index file.%a.var, %int_4.loc11_45
// CHECK:STDOUT:   %.loc11_45.16: init %i32 = initialize_from %.loc11_45.14 to %.loc11_45.15 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %impl.elem0.loc11_45.6: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.6: <bound method> = bound_method %int_6.loc11_35, %impl.elem0.loc11_45.6 [symbolic = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.6: <specific function> = specific_function %Convert.bound.loc11_45.6, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %iN = call %Convert.specific_fn.loc11_45.6(%int_6.loc11_35) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc11_45.17: init %i32 = converted %int_6.loc11_35, %int.convert_checked.loc11_45.6 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %int_5.loc11_45: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc11_45.18: ref %i32 = array_index file.%a.var, %int_5.loc11_45
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = initialize_from %.loc11_45.17 to %.loc11_45.18 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %impl.elem0.loc11_45.7: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.7: <bound method> = bound_method %int_7.loc11_38, %impl.elem0.loc11_45.7 [symbolic = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.7: <specific function> = specific_function %Convert.bound.loc11_45.7, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %iN = call %Convert.specific_fn.loc11_45.7(%int_7.loc11_38) [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc11_45.20: init %i32 = converted %int_7.loc11_38, %int.convert_checked.loc11_45.7 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %int_6.loc11_45: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %.loc11_45.21: ref %i32 = array_index file.%a.var, %int_6.loc11_45
// CHECK:STDOUT:   %.loc11_45.22: init %i32 = initialize_from %.loc11_45.20 to %.loc11_45.21 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %impl.elem0.loc11_45.8: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.8: <bound method> = bound_method %int_8.loc11_41, %impl.elem0.loc11_45.8 [symbolic = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.8: <specific function> = specific_function %Convert.bound.loc11_45.8, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %iN = call %Convert.specific_fn.loc11_45.8(%int_8.loc11_41) [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc11_45.23: init %i32 = converted %int_8.loc11_41, %int.convert_checked.loc11_45.8 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %int_7.loc11_45: Core.IntLiteral = int_value 7 [template = constants.%int_7]
// CHECK:STDOUT:   %.loc11_45.24: ref %i32 = array_index file.%a.var, %int_7.loc11_45
// CHECK:STDOUT:   %.loc11_45.25: init %i32 = initialize_from %.loc11_45.23 to %.loc11_45.24 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %impl.elem0.loc11_45.9: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_45.9: <bound method> = bound_method %int_9, %impl.elem0.loc11_45.9 [symbolic = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.9: <specific function> = specific_function %Convert.bound.loc11_45.9, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %iN = call %Convert.specific_fn.loc11_45.9(%int_9) [symbolic = constants.%int.convert_checked.9]
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = converted %int_9, %int.convert_checked.loc11_45.9 [symbolic = constants.%int.convert_checked.9]
// CHECK:STDOUT:   %int_8.loc11_45: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %.loc11_45.27: ref %i32 = array_index file.%a.var, %int_8.loc11_45
// CHECK:STDOUT:   %.loc11_45.28: init %i32 = initialize_from %.loc11_45.26 to %.loc11_45.27 [symbolic = constants.%int.convert_checked.9]
// CHECK:STDOUT:   %.loc11_45.29: init %array_type = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [symbolic = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %array_type = converted %.loc11_45.1, %.loc11_45.29 [symbolic = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

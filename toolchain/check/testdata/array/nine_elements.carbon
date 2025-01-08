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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_9.988, %i32 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.0db: <bound method> = bound_method %int_3.1ba, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.456: <specific function> = specific_function %Convert.bound.0db, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25b: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.fe5: <bound method> = bound_method %int_4.0c1, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c6f: <specific function> = specific_function %Convert.bound.fe5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2ec: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.b33: <bound method> = bound_method %int_5.64b, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.89d: <specific function> = specific_function %Convert.bound.b33, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.70b: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.b1c: <bound method> = bound_method %int_6.462, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b26: <specific function> = specific_function %Convert.bound.b1c, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.b03: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.eba: <bound method> = bound_method %int_7.29f, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.785: <specific function> = specific_function %Convert.bound.eba, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2da: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.29e: <bound method> = bound_method %int_8.b85, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5a6: <specific function> = specific_function %Convert.bound.29e, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_8.f03: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %Convert.bound.cc8: <bound method> = bound_method %int_9.988, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b8b: <specific function> = specific_function %Convert.bound.cc8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.82c: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.c60, %int_2.166, %int_3.25b, %int_4.2ec, %int_5.70b, %int_6.b03, %int_7.2da, %int_8.f03, %int_9.82c) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_5.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b]
// CHECK:STDOUT:   %int_6.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%int_6.462]
// CHECK:STDOUT:   %int_7.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%int_7.29f]
// CHECK:STDOUT:   %int_8.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.988]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%int_1.loc11_20, %int_2.loc11_23, %int_3.loc11_26, %int_4.loc11_29, %int_5.loc11_32, %int_6.loc11_35, %int_7.loc11_38, %int_8.loc11_41, %int_9)
// CHECK:STDOUT:   %impl.elem0.loc11_45.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.1: <bound method> = bound_method %int_1.loc11_20, %impl.elem0.loc11_45.1 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.1: <specific function> = specific_function %Convert.bound.loc11_45.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %Convert.specific_fn.loc11_45.1(%int_1.loc11_20) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_45.2: init %i32 = converted %int_1.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_45.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_45.4: init %i32 = initialize_from %.loc11_45.2 to %.loc11_45.3 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc11_45.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.2: <bound method> = bound_method %int_2.loc11_23, %impl.elem0.loc11_45.2 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.2: <specific function> = specific_function %Convert.bound.loc11_45.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %Convert.specific_fn.loc11_45.2(%int_2.loc11_23) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = converted %int_2.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%int_2.166]
// CHECK:STDOUT:   %int_1.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_45.6: ref %i32 = array_index file.%a.var, %int_1.loc11_45
// CHECK:STDOUT:   %.loc11_45.7: init %i32 = initialize_from %.loc11_45.5 to %.loc11_45.6 [template = constants.%int_2.166]
// CHECK:STDOUT:   %impl.elem0.loc11_45.3: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.3: <bound method> = bound_method %int_3.loc11_26, %impl.elem0.loc11_45.3 [template = constants.%Convert.bound.0db]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.3: <specific function> = specific_function %Convert.bound.loc11_45.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.456]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %Convert.specific_fn.loc11_45.3(%int_3.loc11_26) [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc11_45.8: init %i32 = converted %int_3.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %int_2.loc11_45: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_45.9: ref %i32 = array_index file.%a.var, %int_2.loc11_45
// CHECK:STDOUT:   %.loc11_45.10: init %i32 = initialize_from %.loc11_45.8 to %.loc11_45.9 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %impl.elem0.loc11_45.4: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.4: <bound method> = bound_method %int_4.loc11_29, %impl.elem0.loc11_45.4 [template = constants.%Convert.bound.fe5]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.4: <specific function> = specific_function %Convert.bound.loc11_45.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c6f]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %Convert.specific_fn.loc11_45.4(%int_4.loc11_29) [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = converted %int_4.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %int_3.loc11_45: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc11_45.12: ref %i32 = array_index file.%a.var, %int_3.loc11_45
// CHECK:STDOUT:   %.loc11_45.13: init %i32 = initialize_from %.loc11_45.11 to %.loc11_45.12 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %impl.elem0.loc11_45.5: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.5: <bound method> = bound_method %int_5.loc11_32, %impl.elem0.loc11_45.5 [template = constants.%Convert.bound.b33]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.5: <specific function> = specific_function %Convert.bound.loc11_45.5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.89d]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %Convert.specific_fn.loc11_45.5(%int_5.loc11_32) [template = constants.%int_5.70b]
// CHECK:STDOUT:   %.loc11_45.14: init %i32 = converted %int_5.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%int_5.70b]
// CHECK:STDOUT:   %int_4.loc11_45: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc11_45.15: ref %i32 = array_index file.%a.var, %int_4.loc11_45
// CHECK:STDOUT:   %.loc11_45.16: init %i32 = initialize_from %.loc11_45.14 to %.loc11_45.15 [template = constants.%int_5.70b]
// CHECK:STDOUT:   %impl.elem0.loc11_45.6: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.6: <bound method> = bound_method %int_6.loc11_35, %impl.elem0.loc11_45.6 [template = constants.%Convert.bound.b1c]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.6: <specific function> = specific_function %Convert.bound.loc11_45.6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b26]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %Convert.specific_fn.loc11_45.6(%int_6.loc11_35) [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc11_45.17: init %i32 = converted %int_6.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %int_5.loc11_45: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc11_45.18: ref %i32 = array_index file.%a.var, %int_5.loc11_45
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = initialize_from %.loc11_45.17 to %.loc11_45.18 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %impl.elem0.loc11_45.7: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.7: <bound method> = bound_method %int_7.loc11_38, %impl.elem0.loc11_45.7 [template = constants.%Convert.bound.eba]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.7: <specific function> = specific_function %Convert.bound.loc11_45.7, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.785]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %Convert.specific_fn.loc11_45.7(%int_7.loc11_38) [template = constants.%int_7.2da]
// CHECK:STDOUT:   %.loc11_45.20: init %i32 = converted %int_7.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%int_7.2da]
// CHECK:STDOUT:   %int_6.loc11_45: Core.IntLiteral = int_value 6 [template = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_45.21: ref %i32 = array_index file.%a.var, %int_6.loc11_45
// CHECK:STDOUT:   %.loc11_45.22: init %i32 = initialize_from %.loc11_45.20 to %.loc11_45.21 [template = constants.%int_7.2da]
// CHECK:STDOUT:   %impl.elem0.loc11_45.8: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.8: <bound method> = bound_method %int_8.loc11_41, %impl.elem0.loc11_45.8 [template = constants.%Convert.bound.29e]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.8: <specific function> = specific_function %Convert.bound.loc11_45.8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a6]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %Convert.specific_fn.loc11_45.8(%int_8.loc11_41) [template = constants.%int_8.f03]
// CHECK:STDOUT:   %.loc11_45.23: init %i32 = converted %int_8.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %int_7.loc11_45: Core.IntLiteral = int_value 7 [template = constants.%int_7.29f]
// CHECK:STDOUT:   %.loc11_45.24: ref %i32 = array_index file.%a.var, %int_7.loc11_45
// CHECK:STDOUT:   %.loc11_45.25: init %i32 = initialize_from %.loc11_45.23 to %.loc11_45.24 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %impl.elem0.loc11_45.9: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_45.9: <bound method> = bound_method %int_9, %impl.elem0.loc11_45.9 [template = constants.%Convert.bound.cc8]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.9: <specific function> = specific_function %Convert.bound.loc11_45.9, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b8b]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %Convert.specific_fn.loc11_45.9(%int_9) [template = constants.%int_9.82c]
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = converted %int_9, %int.convert_checked.loc11_45.9 [template = constants.%int_9.82c]
// CHECK:STDOUT:   %int_8.loc11_45: Core.IntLiteral = int_value 8 [template = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc11_45.27: ref %i32 = array_index file.%a.var, %int_8.loc11_45
// CHECK:STDOUT:   %.loc11_45.28: init %i32 = initialize_from %.loc11_45.26 to %.loc11_45.27 [template = constants.%int_9.82c]
// CHECK:STDOUT:   %.loc11_45.29: init %array_type = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %array_type = converted %.loc11_45.1, %.loc11_45.29 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

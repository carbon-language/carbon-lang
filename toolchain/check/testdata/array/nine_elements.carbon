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
// CHECK:STDOUT:   %int_9.9884: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_9.9884, %i32 [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1baa: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4.0c1b: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_5.64b6: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_6.4624: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %int_7.29f7: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_8.b850: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c605: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1667: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.0db0: <bound method> = bound_method %int_3.1baa, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4567: <specific function> = specific_function %Convert.bound.0db0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25ba: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.fe5d: <bound method> = bound_method %int_4.0c1b, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c6f4: <specific function> = specific_function %Convert.bound.fe5d, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2ecd: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.b33e: <bound method> = bound_method %int_5.64b6, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.89dc: <specific function> = specific_function %Convert.bound.b33e, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.70b1: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.b1cb: <bound method> = bound_method %int_6.4624, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b269: <specific function> = specific_function %Convert.bound.b1cb, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.b036: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.ebac: <bound method> = bound_method %int_7.29f7, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7857: <specific function> = specific_function %Convert.bound.ebac, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2daa: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.29ec: <bound method> = bound_method %int_8.b850, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5a6c: <specific function> = specific_function %Convert.bound.29ec, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_8.f035: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %Convert.bound.cc8a: <bound method> = bound_method %int_9.9884, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b8bb: <specific function> = specific_function %Convert.bound.cc8a, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.82c7: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.c605, %int_2.1667, %int_3.25ba, %int_4.2ecd, %int_5.70b1, %int_6.b036, %int_7.2daa, %int_8.f035, %int_9.82c7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
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
// CHECK:STDOUT:   %int_1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %int_2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %int_3.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %int_4.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1b]
// CHECK:STDOUT:   %int_5.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b6]
// CHECK:STDOUT:   %int_6.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%int_6.4624]
// CHECK:STDOUT:   %int_7.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%int_7.29f7]
// CHECK:STDOUT:   %int_8.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%int_8.b850]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.9884]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%int_1.loc11_20, %int_2.loc11_23, %int_3.loc11_26, %int_4.loc11_29, %int_5.loc11_32, %int_6.loc11_35, %int_7.loc11_38, %int_8.loc11_41, %int_9)
// CHECK:STDOUT:   %impl.elem0.loc11_45.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.1: <bound method> = bound_method %int_1.loc11_20, %impl.elem0.loc11_45.1 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.1: <specific function> = specific_function %Convert.bound.loc11_45.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %Convert.specific_fn.loc11_45.1(%int_1.loc11_20) [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc11_45.2: init %i32 = converted %int_1.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_45.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_45.4: init %i32 = initialize_from %.loc11_45.2 to %.loc11_45.3 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %impl.elem0.loc11_45.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.2: <bound method> = bound_method %int_2.loc11_23, %impl.elem0.loc11_45.2 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.2: <specific function> = specific_function %Convert.bound.loc11_45.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %Convert.specific_fn.loc11_45.2(%int_2.loc11_23) [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = converted %int_2.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %int_1.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc11_45.6: ref %i32 = array_index file.%a.var, %int_1.loc11_45
// CHECK:STDOUT:   %.loc11_45.7: init %i32 = initialize_from %.loc11_45.5 to %.loc11_45.6 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %impl.elem0.loc11_45.3: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.3: <bound method> = bound_method %int_3.loc11_26, %impl.elem0.loc11_45.3 [template = constants.%Convert.bound.0db0]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.3: <specific function> = specific_function %Convert.bound.loc11_45.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4567]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %Convert.specific_fn.loc11_45.3(%int_3.loc11_26) [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc11_45.8: init %i32 = converted %int_3.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %int_2.loc11_45: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %.loc11_45.9: ref %i32 = array_index file.%a.var, %int_2.loc11_45
// CHECK:STDOUT:   %.loc11_45.10: init %i32 = initialize_from %.loc11_45.8 to %.loc11_45.9 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %impl.elem0.loc11_45.4: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.4: <bound method> = bound_method %int_4.loc11_29, %impl.elem0.loc11_45.4 [template = constants.%Convert.bound.fe5d]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.4: <specific function> = specific_function %Convert.bound.loc11_45.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c6f4]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %Convert.specific_fn.loc11_45.4(%int_4.loc11_29) [template = constants.%int_4.2ecd]
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = converted %int_4.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%int_4.2ecd]
// CHECK:STDOUT:   %int_3.loc11_45: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %.loc11_45.12: ref %i32 = array_index file.%a.var, %int_3.loc11_45
// CHECK:STDOUT:   %.loc11_45.13: init %i32 = initialize_from %.loc11_45.11 to %.loc11_45.12 [template = constants.%int_4.2ecd]
// CHECK:STDOUT:   %impl.elem0.loc11_45.5: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.5: <bound method> = bound_method %int_5.loc11_32, %impl.elem0.loc11_45.5 [template = constants.%Convert.bound.b33e]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.5: <specific function> = specific_function %Convert.bound.loc11_45.5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.89dc]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %Convert.specific_fn.loc11_45.5(%int_5.loc11_32) [template = constants.%int_5.70b1]
// CHECK:STDOUT:   %.loc11_45.14: init %i32 = converted %int_5.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%int_5.70b1]
// CHECK:STDOUT:   %int_4.loc11_45: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1b]
// CHECK:STDOUT:   %.loc11_45.15: ref %i32 = array_index file.%a.var, %int_4.loc11_45
// CHECK:STDOUT:   %.loc11_45.16: init %i32 = initialize_from %.loc11_45.14 to %.loc11_45.15 [template = constants.%int_5.70b1]
// CHECK:STDOUT:   %impl.elem0.loc11_45.6: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.6: <bound method> = bound_method %int_6.loc11_35, %impl.elem0.loc11_45.6 [template = constants.%Convert.bound.b1cb]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.6: <specific function> = specific_function %Convert.bound.loc11_45.6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b269]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %Convert.specific_fn.loc11_45.6(%int_6.loc11_35) [template = constants.%int_6.b036]
// CHECK:STDOUT:   %.loc11_45.17: init %i32 = converted %int_6.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%int_6.b036]
// CHECK:STDOUT:   %int_5.loc11_45: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b6]
// CHECK:STDOUT:   %.loc11_45.18: ref %i32 = array_index file.%a.var, %int_5.loc11_45
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = initialize_from %.loc11_45.17 to %.loc11_45.18 [template = constants.%int_6.b036]
// CHECK:STDOUT:   %impl.elem0.loc11_45.7: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.7: <bound method> = bound_method %int_7.loc11_38, %impl.elem0.loc11_45.7 [template = constants.%Convert.bound.ebac]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.7: <specific function> = specific_function %Convert.bound.loc11_45.7, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7857]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %Convert.specific_fn.loc11_45.7(%int_7.loc11_38) [template = constants.%int_7.2daa]
// CHECK:STDOUT:   %.loc11_45.20: init %i32 = converted %int_7.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%int_7.2daa]
// CHECK:STDOUT:   %int_6.loc11_45: Core.IntLiteral = int_value 6 [template = constants.%int_6.4624]
// CHECK:STDOUT:   %.loc11_45.21: ref %i32 = array_index file.%a.var, %int_6.loc11_45
// CHECK:STDOUT:   %.loc11_45.22: init %i32 = initialize_from %.loc11_45.20 to %.loc11_45.21 [template = constants.%int_7.2daa]
// CHECK:STDOUT:   %impl.elem0.loc11_45.8: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.8: <bound method> = bound_method %int_8.loc11_41, %impl.elem0.loc11_45.8 [template = constants.%Convert.bound.29ec]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.8: <specific function> = specific_function %Convert.bound.loc11_45.8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a6c]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %Convert.specific_fn.loc11_45.8(%int_8.loc11_41) [template = constants.%int_8.f035]
// CHECK:STDOUT:   %.loc11_45.23: init %i32 = converted %int_8.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%int_8.f035]
// CHECK:STDOUT:   %int_7.loc11_45: Core.IntLiteral = int_value 7 [template = constants.%int_7.29f7]
// CHECK:STDOUT:   %.loc11_45.24: ref %i32 = array_index file.%a.var, %int_7.loc11_45
// CHECK:STDOUT:   %.loc11_45.25: init %i32 = initialize_from %.loc11_45.23 to %.loc11_45.24 [template = constants.%int_8.f035]
// CHECK:STDOUT:   %impl.elem0.loc11_45.9: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_45.9: <bound method> = bound_method %int_9, %impl.elem0.loc11_45.9 [template = constants.%Convert.bound.cc8a]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.9: <specific function> = specific_function %Convert.bound.loc11_45.9, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b8bb]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %Convert.specific_fn.loc11_45.9(%int_9) [template = constants.%int_9.82c7]
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = converted %int_9, %int.convert_checked.loc11_45.9 [template = constants.%int_9.82c7]
// CHECK:STDOUT:   %int_8.loc11_45: Core.IntLiteral = int_value 8 [template = constants.%int_8.b850]
// CHECK:STDOUT:   %.loc11_45.27: ref %i32 = array_index file.%a.var, %int_8.loc11_45
// CHECK:STDOUT:   %.loc11_45.28: init %i32 = initialize_from %.loc11_45.26 to %.loc11_45.27 [template = constants.%int_9.82c7]
// CHECK:STDOUT:   %.loc11_45.29: init %array_type = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %array_type = converted %.loc11_45.1, %.loc11_45.29 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

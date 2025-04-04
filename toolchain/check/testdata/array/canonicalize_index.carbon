// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/canonicalize_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/canonicalize_index.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";
fn ConvertToU32(a: i32) -> u32 = "int.convert_checked";

var a: array(i32, Add(1, 2)) = (1, 2, 3);
let b: array(i32, 3)* = &a;
let c: array(i32, ConvertToU32(3))* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Add.type.b1f: type = fn_type @Add.1 [concrete]
// CHECK:STDOUT:   %Add: %Add.type.b1f = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %ConvertToU32.type: type = fn_type @ConvertToU32 [concrete]
// CHECK:STDOUT:   %ConvertToU32: %ConvertToU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b8: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.2b8 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ffe: %ImplicitAs.type.2fd = facet_value %i32, (%impl_witness.023) [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.ffe [concrete]
// CHECK:STDOUT:   %Convert.bound.2d6: <bound method> = bound_method %int_3.822, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c6f: <bound method> = bound_method %int_3.822, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %int_3.d14: %u32 = int_value 3 [concrete]
// CHECK:STDOUT:   %impl_witness.8da2: <witness> = impl_witness (imports.%Core.import_ref.823), @impl.750(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e06: type = fn_type @Convert.9, @impl.750(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.47f: %Convert.type.e06 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.bc2: %ImplicitAs.type.2fd = facet_value %u32, (%impl_witness.8da2) [concrete]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.bc2 [concrete]
// CHECK:STDOUT:   %Convert.bound.258: <bound method> = bound_method %int_3.d14, %Convert.47f [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.2eb: <specific function> = specific_function %Convert.47f, @Convert.9(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3a6: <bound method> = bound_method %int_3.d14, %Convert.specific_fn.2eb [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .ConvertToU32 = %ConvertToU32.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type.b1f = fn_decl @Add.1 [concrete = constants.%Add] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11_11: type = splice_block %i32.loc11_11 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc11_19: type = splice_block %i32.loc11_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertToU32.decl: %ConvertToU32.type = fn_decl @ConvertToU32 [concrete = constants.%ConvertToU32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc12_20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc14_1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %.loc14_28: type = splice_block %array_type.loc14 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Add.ref: %Add.type.b1f = name_ref Add, %Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc14_23: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc14_23.1: <bound method> = bound_method %int_1, %impl.elem0.loc14_23 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc14_23: <specific function> = specific_function %impl.elem0.loc14_23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc14_23.2: <bound method> = bound_method %int_1, %specific_fn.loc14_23 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc14_23: init %i32 = call %bound_method.loc14_23.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc14_23.1: %i32 = value_of_initializer %int.convert_checked.loc14_23 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc14_23.2: %i32 = converted %int_1, %.loc14_23.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc14_26: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc14_26.1: <bound method> = bound_method %int_2, %impl.elem0.loc14_26 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc14_26: <specific function> = specific_function %impl.elem0.loc14_26, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc14_26.2: <bound method> = bound_method %int_2, %specific_fn.loc14_26 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc14_26: init %i32 = call %bound_method.loc14_26.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc14_26.1: %i32 = value_of_initializer %int.convert_checked.loc14_26 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc14_26.2: %i32 = converted %int_2, %.loc14_26.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %int.sadd: init %i32 = call %Add.ref(%.loc14_23.2, %.loc14_26.2) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc14_27: %.4d8 = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:     %bound_method.loc14_27.1: <bound method> = bound_method %int.sadd, %impl.elem0.loc14_27 [concrete = constants.%Convert.bound.2d6]
// CHECK:STDOUT:     %specific_fn.loc14_27: <specific function> = specific_function %impl.elem0.loc14_27, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:     %bound_method.loc14_27.2: <bound method> = bound_method %int.sadd, %specific_fn.loc14_27 [concrete = constants.%bound_method.c6f]
// CHECK:STDOUT:     %.loc14_27.1: %i32 = value_of_initializer %int.sadd [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc14_27.2: %i32 = converted %int.sadd, %.loc14_27.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %int.convert_checked.loc14_27: init Core.IntLiteral = call %bound_method.loc14_27.2(%.loc14_27.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc14_27.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_27 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc14_27.4: Core.IntLiteral = converted %int.sadd, %.loc14_27.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc14: type = array_type %.loc14_27.4, %i32.loc14 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %ptr = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: type = splice_block %ptr.loc15 [concrete = constants.%ptr] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc15: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc15: type = array_type %int_3.loc15, %i32.loc15 [concrete = constants.%array_type]
// CHECK:STDOUT:     %ptr.loc15: type = ptr_type %array_type.loc15 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr = bind_name b, @__global_init.%addr.loc15
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %ptr = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_35: type = splice_block %ptr.loc16 [concrete = constants.%ptr] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ConvertToU32.ref: %ConvertToU32.type = name_ref ConvertToU32, %ConvertToU32.decl [concrete = constants.%ConvertToU32]
// CHECK:STDOUT:     %int_3.loc16: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0.loc16_32: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc16_32.1: <bound method> = bound_method %int_3.loc16, %impl.elem0.loc16_32 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:     %specific_fn.loc16_32: <specific function> = specific_function %impl.elem0.loc16_32, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc16_32.2: <bound method> = bound_method %int_3.loc16, %specific_fn.loc16_32 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:     %int.convert_checked.loc16_32: init %i32 = call %bound_method.loc16_32.2(%int_3.loc16) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc16_32.1: %i32 = value_of_initializer %int.convert_checked.loc16_32 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc16_32.2: %i32 = converted %int_3.loc16, %.loc16_32.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %int.convert_checked.loc16_33.1: init %u32 = call %ConvertToU32.ref(%.loc16_32.2) [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %impl.elem0.loc16_33: %.883 = impl_witness_access constants.%impl_witness.8da2, element0 [concrete = constants.%Convert.47f]
// CHECK:STDOUT:     %bound_method.loc16_33.1: <bound method> = bound_method %int.convert_checked.loc16_33.1, %impl.elem0.loc16_33 [concrete = constants.%Convert.bound.258]
// CHECK:STDOUT:     %specific_fn.loc16_33: <specific function> = specific_function %impl.elem0.loc16_33, @Convert.9(constants.%int_32) [concrete = constants.%Convert.specific_fn.2eb]
// CHECK:STDOUT:     %bound_method.loc16_33.2: <bound method> = bound_method %int.convert_checked.loc16_33.1, %specific_fn.loc16_33 [concrete = constants.%bound_method.3a6]
// CHECK:STDOUT:     %.loc16_33.1: %u32 = value_of_initializer %int.convert_checked.loc16_33.1 [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc16_33.2: %u32 = converted %int.convert_checked.loc16_33.1, %.loc16_33.1 [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %int.convert_checked.loc16_33.2: init Core.IntLiteral = call %bound_method.loc16_33.2(%.loc16_33.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc16_33.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc16_33.2 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc16_33.4: Core.IntLiteral = converted %int.convert_checked.loc16_33.1, %.loc16_33.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc16: type = array_type %.loc16_33.4, %i32.loc16 [concrete = constants.%array_type]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %array_type.loc16 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %ptr = bind_name c, @__global_init.%addr.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToU32(%a.param_patt: %i32) -> %u32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc14_33: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc14_36: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc14_40.1: %tuple.type = tuple_literal (%int_1.loc14_33, %int_2.loc14_36, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc14_40.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_40.1: <bound method> = bound_method %int_1.loc14_33, %impl.elem0.loc14_40.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_40.1: <specific function> = specific_function %impl.elem0.loc14_40.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:   %bound_method.loc14_40.2: <bound method> = bound_method %int_1.loc14_33, %specific_fn.loc14_40.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_40.1: init %i32 = call %bound_method.loc14_40.2(%int_1.loc14_33) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_40.2: init %i32 = converted %int_1.loc14_33, %int.convert_checked.loc14_40.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc14_40.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc14_40.4: init %i32 = initialize_from %.loc14_40.2 to %.loc14_40.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_40.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_40.3: <bound method> = bound_method %int_2.loc14_36, %impl.elem0.loc14_40.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_40.2: <specific function> = specific_function %impl.elem0.loc14_40.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:   %bound_method.loc14_40.4: <bound method> = bound_method %int_2.loc14_36, %specific_fn.loc14_40.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_40.2: init %i32 = call %bound_method.loc14_40.4(%int_2.loc14_36) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_40.5: init %i32 = converted %int_2.loc14_36, %int.convert_checked.loc14_40.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc14_40: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_40.6: ref %i32 = array_index file.%a.var, %int_1.loc14_40
// CHECK:STDOUT:   %.loc14_40.7: init %i32 = initialize_from %.loc14_40.5 to %.loc14_40.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc14_40.3: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_40.5: <bound method> = bound_method %int_3, %impl.elem0.loc14_40.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc14_40.3: <specific function> = specific_function %impl.elem0.loc14_40.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:   %bound_method.loc14_40.6: <bound method> = bound_method %int_3, %specific_fn.loc14_40.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc14_40.3: init %i32 = call %bound_method.loc14_40.6(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_40.8: init %i32 = converted %int_3, %int.convert_checked.loc14_40.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc14_40: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_40.9: ref %i32 = array_index file.%a.var, %int_2.loc14_40
// CHECK:STDOUT:   %.loc14_40.10: init %i32 = initialize_from %.loc14_40.8 to %.loc14_40.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_40.11: init %array_type = array_init (%.loc14_40.4, %.loc14_40.7, %.loc14_40.10) to file.%a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc14_1: init %array_type = converted %.loc14_40.1, %.loc14_40.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc14_1
// CHECK:STDOUT:   %a.ref.loc15: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %addr.loc15: %ptr = addr_of %a.ref.loc15
// CHECK:STDOUT:   %a.ref.loc16: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %addr.loc16: %ptr = addr_of %a.ref.loc16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

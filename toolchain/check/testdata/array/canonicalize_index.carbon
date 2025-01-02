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

var a: [i32; Add(1, 2)] = (1, 2, 3);
let b: [i32; 3]* = &a;
let c: [i32; ConvertToU32(3)]* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Add.type.1: type = fn_type @Add.1 [template]
// CHECK:STDOUT:   %Add: %Add.type.1 = struct_value () [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %ConvertToU32.type: type = fn_type @ConvertToU32 [template]
// CHECK:STDOUT:   %ConvertToU32: %ConvertToU32.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (imports.%import_ref.18), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_3.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.2, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_3.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.2, %int_2.2, %int_3.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.9
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .ConvertToU32 = %ConvertToU32.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type.1 = fn_decl @Add.1 [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_11: type = splice_block %i32.loc11_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc11_19: type = splice_block %i32.loc11_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertToU32.decl: %ConvertToU32.type = fn_decl @ConvertToU32 [template = constants.%ConvertToU32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc12_20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToU32(%a.param_patt: %i32) -> %u32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc14_28: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc14_31: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc14_35.1: %tuple.type = tuple_literal (%int_1.loc14_28, %int_2.loc14_31, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc14_35.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_35.1: <bound method> = bound_method %int_1.loc14_28, %impl.elem0.loc14_35.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_35.1: <specific function> = specific_function %Convert.bound.loc14_35.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_35.1: init %i32 = call %Convert.specific_fn.loc14_35.1(%int_1.loc14_28) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_35.2: init %i32 = converted %int_1.loc14_28, %int.convert_checked.loc14_35.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc14_35.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc14_35.4: init %i32 = initialize_from %.loc14_35.2 to %.loc14_35.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc14_35.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_35.2: <bound method> = bound_method %int_2.loc14_31, %impl.elem0.loc14_35.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_35.2: <specific function> = specific_function %Convert.bound.loc14_35.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_35.2: init %i32 = call %Convert.specific_fn.loc14_35.2(%int_2.loc14_31) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_35.5: init %i32 = converted %int_2.loc14_31, %int.convert_checked.loc14_35.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int_1.loc14_35: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_35.6: ref %i32 = array_index file.%a.var, %int_1.loc14_35
// CHECK:STDOUT:   %.loc14_35.7: init %i32 = initialize_from %.loc14_35.5 to %.loc14_35.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc14_35.3: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_35.3: <bound method> = bound_method %int_3, %impl.elem0.loc14_35.3 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_35.3: <specific function> = specific_function %Convert.bound.loc14_35.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc14_35.3: init %i32 = call %Convert.specific_fn.loc14_35.3(%int_3) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc14_35.8: init %i32 = converted %int_3, %int.convert_checked.loc14_35.3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_2.loc14_35: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc14_35.9: ref %i32 = array_index file.%a.var, %int_2.loc14_35
// CHECK:STDOUT:   %.loc14_35.10: init %i32 = initialize_from %.loc14_35.8 to %.loc14_35.9 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc14_35.11: init %array_type = array_init (%.loc14_35.4, %.loc14_35.7, %.loc14_35.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_36: init %array_type = converted %.loc14_35.1, %.loc14_35.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc14_36
// CHECK:STDOUT:   %a.ref.loc15: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %addr.loc15: %ptr = addr_of %a.ref.loc15
// CHECK:STDOUT:   %b: %ptr = bind_name b, %addr.loc15
// CHECK:STDOUT:   %a.ref.loc16: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %addr.loc16: %ptr = addr_of %a.ref.loc16
// CHECK:STDOUT:   %c: %ptr = bind_name c, %addr.loc16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

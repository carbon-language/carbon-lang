// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_bound_negative.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_bound_negative.carbon

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bound_negative.carbon:[[@LINE+4]]:8: error: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR: var a: array(i32, Negate(1));
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(i32, Negate(1));

// CHECK:STDOUT: --- fail_bound_negative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Negate.type.15b: type = fn_type @Negate.1 [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type.15b = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f7f: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.f7f [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_-1.251: %i32 = int_value -1 [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e25: %ImplicitAs.type.2fd = facet_value %i32, %impl_witness.023 [concrete]
// CHECK:STDOUT:   %.10e: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.e25 [concrete]
// CHECK:STDOUT:   %Convert.bound.75d: <bound method> = bound_method %int_-1.251, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.245: <bound method> = bound_method %int_-1.251, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type.15b = fn_decl @Negate.1 [concrete = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %i32.loc11_14 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %.loc17_1: <error> = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %.loc17_28: type = splice_block %array_type [concrete = <error>] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Negate.ref: %Negate.type.15b = name_ref Negate, %Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0.loc17_26: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc17_26.1: <bound method> = bound_method %int_1, %impl.elem0.loc17_26 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc17_26: <specific function> = specific_function %impl.elem0.loc17_26, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc17_26.2: <bound method> = bound_method %int_1, %specific_fn.loc17_26 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc17_26: init %i32 = call %bound_method.loc17_26.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc17_26.1: %i32 = value_of_initializer %int.convert_checked.loc17_26 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc17_26.2: %i32 = converted %int_1, %.loc17_26.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %int.snegate: init %i32 = call %Negate.ref(%.loc17_26.2) [concrete = constants.%int_-1.251]
// CHECK:STDOUT:     %impl.elem0.loc17_27: %.10e = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:     %bound_method.loc17_27.1: <bound method> = bound_method %int.snegate, %impl.elem0.loc17_27 [concrete = constants.%Convert.bound.75d]
// CHECK:STDOUT:     %specific_fn.loc17_27: <specific function> = specific_function %impl.elem0.loc17_27, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:     %bound_method.loc17_27.2: <bound method> = bound_method %int.snegate, %specific_fn.loc17_27 [concrete = constants.%bound_method.245]
// CHECK:STDOUT:     %.loc17_27.1: %i32 = value_of_initializer %int.snegate [concrete = constants.%int_-1.251]
// CHECK:STDOUT:     %.loc17_27.2: %i32 = converted %int.snegate, %.loc17_27.1 [concrete = constants.%int_-1.251]
// CHECK:STDOUT:     %int.convert_checked.loc17_27: init Core.IntLiteral = call %bound_method.loc17_27.2(%.loc17_27.2) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:     %.loc17_27.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc17_27 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:     %.loc17_27.4: Core.IntLiteral = converted %int.snegate, %.loc17_27.3 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:     %array_type: type = array_type %.loc17_27.4, %i32 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: <error> = bind_name a, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%n.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:

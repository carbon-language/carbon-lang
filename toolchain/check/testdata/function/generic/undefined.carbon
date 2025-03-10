// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/undefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/undefined.carbon

// --- call_defined.carbon

library "[[@TEST_NAME]]";

fn Defined[T:! type](x: T) -> T {
  return x;
}

fn CallDefined() -> i32 {
  return Defined(0 as i32);
}

// --- call_defined_late.carbon

library "[[@TEST_NAME]]";

fn Defined[T:! type](x: T) -> T;

fn CallDefined() -> i32 {
  return Defined(0 as i32);
}

fn Defined[T:! type](x: T) -> T {
  return x;
}

// --- fail_call_undefined.carbon

library "[[@TEST_NAME]]";

fn Undefined[T:! type](x: T) -> T;

fn CallUndefined() -> i32 {
  // CHECK:STDERR: fail_call_undefined.carbon:[[@LINE+7]]:10: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR:   return Undefined(0 as i32);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_call_undefined.carbon:[[@LINE-6]]:1: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR: fn Undefined[T:! type](x: T) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Undefined(0 as i32);
}

// CHECK:STDOUT: --- call_defined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Defined.type: type = fn_type @Defined [concrete]
// CHECK:STDOUT:   %Defined: %Defined.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CallDefined.type: type = fn_type @CallDefined [concrete]
// CHECK:STDOUT:   %CallDefined: %CallDefined.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Defined.specific_fn: <specific function> = specific_function %Defined, @Defined(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Defined = %Defined.decl
// CHECK:STDOUT:     .CallDefined = %CallDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Defined.decl: %Defined.type = fn_decl @Defined [concrete = constants.%Defined] {
// CHECK:STDOUT:     %T.patt.loc4_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Defined.%T.loc4_12.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Defined.%T.loc4_12.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Defined.%T.loc4_12.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Defined.%T.loc4_12.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_31: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_12.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Defined.%T.loc4_12.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_25: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %x: @Defined.%T.loc4_12.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Defined.%T.loc4_12.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Defined.%T.loc4_12.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallDefined.decl: %CallDefined.type = fn_decl @CallDefined [concrete = constants.%CallDefined] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Defined(%T.loc4_12.1: type) {
// CHECK:STDOUT:   %T.loc4_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_12.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Defined.%T.loc4_12.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_12.1: type](%x.param_patt: @Defined.%T.loc4_12.2 (%T)) -> @Defined.%T.loc4_12.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Defined.%T.loc4_12.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDefined() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Defined.ref: %Defined.type = name_ref Defined, file.%Defined.decl [concrete = constants.%Defined]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc9_20.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_20.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc9_20.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_20.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_20.2: %i32 = converted %int_0, %.loc9_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Defined.specific_fn: <specific function> = specific_function %Defined.ref, @Defined(constants.%i32) [concrete = constants.%Defined.specific_fn]
// CHECK:STDOUT:   %Defined.call: init %i32 = call %Defined.specific_fn(%.loc9_20.2)
// CHECK:STDOUT:   %.loc9_27.1: %i32 = value_of_initializer %Defined.call
// CHECK:STDOUT:   %.loc9_27.2: %i32 = converted %Defined.call, %.loc9_27.1
// CHECK:STDOUT:   return %.loc9_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(constants.%T) {
// CHECK:STDOUT:   %T.loc4_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_12.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_12.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_12.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_defined_late.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Defined.type: type = fn_type @Defined [concrete]
// CHECK:STDOUT:   %Defined: %Defined.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CallDefined.type: type = fn_type @CallDefined [concrete]
// CHECK:STDOUT:   %CallDefined: %CallDefined.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Defined.specific_fn: <specific function> = specific_function %Defined, @Defined(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Defined = %Defined.decl.loc4
// CHECK:STDOUT:     .CallDefined = %CallDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Defined.decl.loc4: %Defined.type = fn_decl @Defined [concrete = constants.%Defined] {
// CHECK:STDOUT:     %T.patt.loc10: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %x.patt: %T = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %T = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %T = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_31: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_12.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.param.loc4: @Defined.%T.loc4_12.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_25: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.loc4: @Defined.%T.loc4_12.2 (%T) = bind_name x, %x.param.loc4
// CHECK:STDOUT:     %return.param.loc4: ref @Defined.%T.loc4_12.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return.loc4: ref @Defined.%T.loc4_12.2 (%T) = return_slot %return.param.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallDefined.decl: %CallDefined.type = fn_decl @CallDefined [concrete = constants.%CallDefined] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Defined.decl.loc10: %Defined.type = fn_decl @Defined [concrete = constants.%Defined] {
// CHECK:STDOUT:     %T.patt.loc10: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %x.patt: %T = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %T = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %T = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10_31: type = name_ref T, %T.loc10 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T, 0 [symbolic = constants.%T]
// CHECK:STDOUT:     %x.param.loc10: %T = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc10_25: type = name_ref T, %T.loc10 [symbolic = constants.%T]
// CHECK:STDOUT:     %x.loc10: %T = bind_name x, %x.param.loc10
// CHECK:STDOUT:     %return.param.loc10: ref %T = out_param call_param1
// CHECK:STDOUT:     %return.loc10: ref %T = return_slot %return.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Defined(%T.loc4_12.1: type) {
// CHECK:STDOUT:   %T.loc4_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Defined.%T.loc4_12.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc10: type](%x.param_patt: %T) -> %T {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Defined.%T.loc4_12.2 (%T) = name_ref x, %x.loc10
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDefined() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Defined.ref: %Defined.type = name_ref Defined, file.%Defined.decl.loc4 [concrete = constants.%Defined]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc7_20.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_20.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc7_20.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_20.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_20.2: %i32 = converted %int_0, %.loc7_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Defined.specific_fn: <specific function> = specific_function %Defined.ref, @Defined(constants.%i32) [concrete = constants.%Defined.specific_fn]
// CHECK:STDOUT:   %Defined.call: init %i32 = call %Defined.specific_fn(%.loc7_20.2)
// CHECK:STDOUT:   %.loc7_27.1: %i32 = value_of_initializer %Defined.call
// CHECK:STDOUT:   %.loc7_27.2: %i32 = converted %Defined.call, %.loc7_27.1
// CHECK:STDOUT:   return %.loc7_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(constants.%T) {
// CHECK:STDOUT:   %T.loc4_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_12.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Undefined.type: type = fn_type @Undefined [concrete]
// CHECK:STDOUT:   %Undefined: %Undefined.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CallUndefined.type: type = fn_type @CallUndefined [concrete]
// CHECK:STDOUT:   %CallUndefined: %CallUndefined.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Undefined.specific_fn: <specific function> = specific_function %Undefined, @Undefined(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .CallUndefined = %CallUndefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Undefined.decl: %Undefined.type = fn_decl @Undefined [concrete = constants.%Undefined] {
// CHECK:STDOUT:     %T.patt.loc4_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Undefined.%T.loc4_14.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Undefined.%T.loc4_14.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Undefined.%T.loc4_14.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Undefined.%T.loc4_14.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_33: type = name_ref T, %T.loc4_14.1 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Undefined.%T.loc4_14.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_27: type = name_ref T, %T.loc4_14.1 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %x: @Undefined.%T.loc4_14.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Undefined.%T.loc4_14.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Undefined.%T.loc4_14.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallUndefined.decl: %CallUndefined.type = fn_decl @CallUndefined [concrete = constants.%CallUndefined] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Undefined(%T.loc4_14.1: type) {
// CHECK:STDOUT:   %T.loc4_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_14.1: type](%x.param_patt: @Undefined.%T.loc4_14.2 (%T)) -> @Undefined.%T.loc4_14.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallUndefined() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Undefined.ref: %Undefined.type = name_ref Undefined, file.%Undefined.decl [concrete = constants.%Undefined]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc14_22.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_22.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc14_22.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc14_22.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc14_22.2: %i32 = converted %int_0, %.loc14_22.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Undefined.specific_fn: <specific function> = specific_function %Undefined.ref, @Undefined(constants.%i32) [concrete = constants.%Undefined.specific_fn]
// CHECK:STDOUT:   %Undefined.call: init %i32 = call %Undefined.specific_fn(%.loc14_22.2)
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %Undefined.call
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %Undefined.call, %.loc14_29.1
// CHECK:STDOUT:   return %.loc14_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Undefined(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Undefined(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_14.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:

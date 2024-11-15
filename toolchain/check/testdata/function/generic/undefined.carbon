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
  return Defined(0);
}

// --- call_defined_late.carbon

library "[[@TEST_NAME]]";

fn Defined[T:! type](x: T) -> T;

fn CallDefined() -> i32 {
  return Defined(0);
}

fn Defined[T:! type](x: T) -> T {
  return x;
}

// --- fail_call_undefined.carbon

library "[[@TEST_NAME]]";

fn Undefined[T:! type](x: T) -> T;

fn CallUndefined() -> i32 {
  // CHECK:STDERR: fail_call_undefined.carbon:[[@LINE+6]]:10: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR:   return Undefined(0);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_call_undefined.carbon:[[@LINE-6]]:1: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR: fn Undefined[T:! type](x: T) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return Undefined(0);
}

// CHECK:STDOUT: --- call_defined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Defined.type: type = fn_type @Defined [template]
// CHECK:STDOUT:   %Defined: %Defined.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %CallDefined.type: type = fn_type @CallDefined [template]
// CHECK:STDOUT:   %CallDefined: %CallDefined.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %Defined, @Defined(i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Defined = %Defined.decl
// CHECK:STDOUT:     .CallDefined = %CallDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Defined.decl: %Defined.type = fn_decl @Defined [template = constants.%Defined] {
// CHECK:STDOUT:     %T.patt.loc4_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_12.1, runtime_param<invalid> [symbolic = %T.patt.loc4_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Defined.%T.loc4_12.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Defined.%T.loc4_12.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Defined.%T.loc4_12.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Defined.%T.loc4_12.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_25: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_31: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_12.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Defined.%T.loc4_12.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Defined.%T.loc4_12.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Defined.%T.loc4_12.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Defined.%T.loc4_12.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallDefined.decl: %CallDefined.type = fn_decl @CallDefined [template = constants.%CallDefined] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_21.2: type = converted %int.make_type_32, %.loc8_21.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Defined(%T.loc4_12.1: type) {
// CHECK:STDOUT:   %T.loc4_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_12.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @Defined.%T.loc4_12.2 (%T)) -> @Defined.%T.loc4_12.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Defined.%T.loc4_12.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDefined() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Defined.ref: %Defined.type = name_ref Defined, file.%Defined.decl [template = constants.%Defined]
// CHECK:STDOUT:   %.loc9_18: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_10: <specific function> = specific_function %Defined.ref, @Defined(i32) [template = constants.%.2]
// CHECK:STDOUT:   %Defined.call: init i32 = call %.loc9_10(%.loc9_18)
// CHECK:STDOUT:   %.loc9_20.1: i32 = value_of_initializer %Defined.call
// CHECK:STDOUT:   %.loc9_20.2: i32 = converted %Defined.call, %.loc9_20.1
// CHECK:STDOUT:   return %.loc9_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(constants.%T) {
// CHECK:STDOUT:   %T.loc4_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_12.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(i32) {
// CHECK:STDOUT:   %T.loc4_12.2 => i32
// CHECK:STDOUT:   %T.patt.loc4_12.2 => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_defined_late.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Defined.type: type = fn_type @Defined [template]
// CHECK:STDOUT:   %Defined: %Defined.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %CallDefined.type: type = fn_type @CallDefined [template]
// CHECK:STDOUT:   %CallDefined: %CallDefined.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %Defined, @Defined(i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Defined = %Defined.decl.loc4
// CHECK:STDOUT:     .CallDefined = %CallDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Defined.decl.loc4: %Defined.type = fn_decl @Defined [template = constants.%Defined] {
// CHECK:STDOUT:     %T.patt.loc10: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10, runtime_param<invalid> [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %x.patt: %T = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %T = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %T = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_25: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_31: type = name_ref T, %T.loc4_12.1 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %T.param.loc4: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_12.1: type = bind_symbolic_name T, 0, %T.param.loc4 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:     %x.param.loc4: @Defined.%T.loc4_12.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x.loc4: @Defined.%T.loc4_12.2 (%T) = bind_name x, %x.param.loc4
// CHECK:STDOUT:     %return.param.loc4: ref @Defined.%T.loc4_12.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %.loc4: ref @Defined.%T.loc4_12.2 (%T) = return_slot %return.param.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallDefined.decl: %CallDefined.type = fn_decl @CallDefined [template = constants.%CallDefined] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_21.2: type = converted %int.make_type_32, %.loc6_21.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Defined.decl.loc10: %Defined.type = fn_decl @Defined [template = constants.%Defined] {
// CHECK:STDOUT:     %T.patt.loc10: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10, runtime_param<invalid> [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %x.patt: %T = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %T = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %T = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10_25: type = name_ref T, %T.loc10 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc10_31: type = name_ref T, %T.loc10 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.param.loc10: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T, 0, %T.param.loc10 [symbolic = constants.%T]
// CHECK:STDOUT:     %x.param.loc10: %T = value_param runtime_param0
// CHECK:STDOUT:     %x.loc10: %T = bind_name x, %x.param.loc10
// CHECK:STDOUT:     %return.param.loc10: ref %T = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %T = return_slot %return.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Defined(%T.loc4_12.1: type) {
// CHECK:STDOUT:   %T.loc4_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: %T) -> %T {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Defined.%T.loc4_12.2 (%T) = name_ref x, %x.loc10
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDefined() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Defined.ref: %Defined.type = name_ref Defined, file.%Defined.decl.loc4 [template = constants.%Defined]
// CHECK:STDOUT:   %.loc7_18: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %Defined.ref, @Defined(i32) [template = constants.%.2]
// CHECK:STDOUT:   %Defined.call: init i32 = call %.loc7_10(%.loc7_18)
// CHECK:STDOUT:   %.loc7_20.1: i32 = value_of_initializer %Defined.call
// CHECK:STDOUT:   %.loc7_20.2: i32 = converted %Defined.call, %.loc7_20.1
// CHECK:STDOUT:   return %.loc7_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(constants.%T) {
// CHECK:STDOUT:   %T.loc4_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Defined(i32) {
// CHECK:STDOUT:   %T.loc4_12.2 => i32
// CHECK:STDOUT:   %T.patt.loc4 => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Undefined.type: type = fn_type @Undefined [template]
// CHECK:STDOUT:   %Undefined: %Undefined.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %CallUndefined.type: type = fn_type @CallUndefined [template]
// CHECK:STDOUT:   %CallUndefined: %CallUndefined.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %Undefined, @Undefined(i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .CallUndefined = %CallUndefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Undefined.decl: %Undefined.type = fn_decl @Undefined [template = constants.%Undefined] {
// CHECK:STDOUT:     %T.patt.loc4_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_14.1, runtime_param<invalid> [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Undefined.%T.loc4_14.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Undefined.%T.loc4_14.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Undefined.%T.loc4_14.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Undefined.%T.loc4_14.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_27: type = name_ref T, %T.loc4_14.1 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_33: type = name_ref T, %T.loc4_14.1 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Undefined.%T.loc4_14.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Undefined.%T.loc4_14.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Undefined.%T.loc4_14.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Undefined.%T.loc4_14.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallUndefined.decl: %CallUndefined.type = fn_decl @CallUndefined [template = constants.%CallUndefined] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_23.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_23.2: type = converted %int.make_type_32, %.loc6_23.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Undefined(%T.loc4_14.1: type) {
// CHECK:STDOUT:   %T.loc4_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @Undefined.%T.loc4_14.2 (%T)) -> @Undefined.%T.loc4_14.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallUndefined() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Undefined.ref: %Undefined.type = name_ref Undefined, file.%Undefined.decl [template = constants.%Undefined]
// CHECK:STDOUT:   %.loc13_20: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_10: <specific function> = specific_function %Undefined.ref, @Undefined(i32) [template = constants.%.2]
// CHECK:STDOUT:   %Undefined.call: init i32 = call %.loc13_10(%.loc13_20)
// CHECK:STDOUT:   %.loc13_22.1: i32 = value_of_initializer %Undefined.call
// CHECK:STDOUT:   %.loc13_22.2: i32 = converted %Undefined.call, %.loc13_22.1
// CHECK:STDOUT:   return %.loc13_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Undefined(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Undefined(i32) {
// CHECK:STDOUT:   %T.loc4_14.2 => i32
// CHECK:STDOUT:   %T.patt.loc4_14.2 => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:

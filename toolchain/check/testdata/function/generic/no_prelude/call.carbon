// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/call.carbon

// --- explicit.carbon

library "[[@TEST_NAME]]";

fn Function(T:! type, x: T) -> T {
  return x;
}

fn CallGeneric(T:! type, x: T) -> T {
  return Function(T, x);
}

fn CallGenericPtr(T:! type, x: T*) -> T* {
  return Function(T*, x);
}

class C {}

fn CallSpecific(x: C) -> C {
  return Function(C, x);
}

// --- deduced.carbon

library "[[@TEST_NAME]]";

fn Function[T:! type](x: T) -> T {
  return x;
}

fn CallGeneric(T:! type, x: T) -> T {
  return Function(x);
}

fn CallGenericPtr(T:! type, x: T*) -> T* {
  return Function(x);
}

class C {}

fn CallSpecific(x: C) -> C {
  return Function(x);
}

// CHECK:STDOUT: --- explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [template]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [template]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [template]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [template]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [template = constants.%Function] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @Function.%T.loc4: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Function.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, @Function.%T.loc4 [symbolic = @Function.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc4_23.1: @Function.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @Function.%x: @Function.%T.1 (%T) = bind_name x, %x.loc4_23.1
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, @Function.%T.loc4 [symbolic = @Function.%T.1 (constants.%T)]
// CHECK:STDOUT:     @Function.%return: ref @Function.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [template = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.loc8_16.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @CallGeneric.%T.loc8: type = bind_symbolic_name T 0, %T.loc8_16.1 [symbolic = @CallGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, @CallGeneric.%T.loc8 [symbolic = @CallGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc8_26.1: @CallGeneric.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @CallGeneric.%x: @CallGeneric.%T.1 (%T) = bind_name x, %x.loc8_26.1
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, @CallGeneric.%T.loc8 [symbolic = @CallGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     @CallGeneric.%return: ref @CallGeneric.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [template = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.loc12_19.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @CallGenericPtr.%T.loc12: type = bind_symbolic_name T 0, %T.loc12_19.1 [symbolic = @CallGenericPtr.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc12_32: type = name_ref T, @CallGenericPtr.%T.loc12 [symbolic = @CallGenericPtr.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12_33: type = ptr_type %T [symbolic = @CallGenericPtr.%.1 (constants.%.2)]
// CHECK:STDOUT:     %x.loc12_29.1: @CallGenericPtr.%.1 (%.2) = param x, runtime_param0
// CHECK:STDOUT:     @CallGenericPtr.%x: @CallGenericPtr.%.1 (%.2) = bind_name x, %x.loc12_29.1
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, @CallGenericPtr.%T.loc12 [symbolic = @CallGenericPtr.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12_40: type = ptr_type %T [symbolic = @CallGenericPtr.%.1 (constants.%.2)]
// CHECK:STDOUT:     @CallGenericPtr.%return: ref @CallGenericPtr.%.1 (%.2) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [template = constants.%CallSpecific] {
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.loc18_17.1: %C = param x, runtime_param0
// CHECK:STDOUT:     @CallSpecific.%x: %C = bind_name x, %x.loc18_17.1
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @CallSpecific.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc4: type, %x: @Function.%T.1 (%T)) -> @Function.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: type, %x: @CallGeneric.%T.1 (%T)) -> @CallGeneric.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.1 (%T) = call %Function.ref(%x.ref)
// CHECK:STDOUT:     %.loc9_24.1: @CallGeneric.%T.1 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_24.2: @CallGeneric.%T.1 (%T) = converted %Function.call, %.loc9_24.1
// CHECK:STDOUT:     return %.loc9_24.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %.1: type = ptr_type @CallGenericPtr.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc12: type, %x: @CallGenericPtr.%.1 (%.2)) -> @CallGenericPtr.%.1 (%.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13_20: type = ptr_type %T [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%.1 (%.2) = name_ref x, %x
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%.1 (%.2) = call %Function.ref(%x.ref)
// CHECK:STDOUT:     %.loc13_25.1: @CallGenericPtr.%.1 (%.2) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_25.2: @CallGenericPtr.%.1 (%.2) = converted %Function.call, %.loc13_25.1
// CHECK:STDOUT:     return %.loc13_25.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %.loc18: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %Function.ref(%x.ref) to %.loc18
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%.2) {
// CHECK:STDOUT:   %T.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [template]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [template]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [template]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [template]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [template = constants.%Function] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @Function.%T.loc4: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Function.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, @Function.%T.loc4 [symbolic = @Function.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc4_23.1: @Function.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @Function.%x: @Function.%T.1 (%T) = bind_name x, %x.loc4_23.1
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, @Function.%T.loc4 [symbolic = @Function.%T.1 (constants.%T)]
// CHECK:STDOUT:     @Function.%return: ref @Function.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [template = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.loc8_16.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @CallGeneric.%T.loc8: type = bind_symbolic_name T 0, %T.loc8_16.1 [symbolic = @CallGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, @CallGeneric.%T.loc8 [symbolic = @CallGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc8_26.1: @CallGeneric.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @CallGeneric.%x: @CallGeneric.%T.1 (%T) = bind_name x, %x.loc8_26.1
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, @CallGeneric.%T.loc8 [symbolic = @CallGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     @CallGeneric.%return: ref @CallGeneric.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [template = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.loc12_19.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @CallGenericPtr.%T.loc12: type = bind_symbolic_name T 0, %T.loc12_19.1 [symbolic = @CallGenericPtr.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc12_32: type = name_ref T, @CallGenericPtr.%T.loc12 [symbolic = @CallGenericPtr.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12_33: type = ptr_type %T [symbolic = @CallGenericPtr.%.1 (constants.%.2)]
// CHECK:STDOUT:     %x.loc12_29.1: @CallGenericPtr.%.1 (%.2) = param x, runtime_param0
// CHECK:STDOUT:     @CallGenericPtr.%x: @CallGenericPtr.%.1 (%.2) = bind_name x, %x.loc12_29.1
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, @CallGenericPtr.%T.loc12 [symbolic = @CallGenericPtr.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12_40: type = ptr_type %T [symbolic = @CallGenericPtr.%.1 (constants.%.2)]
// CHECK:STDOUT:     @CallGenericPtr.%return: ref @CallGenericPtr.%.1 (%.2) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [template = constants.%CallSpecific] {
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.loc18_17.1: %C = param x, runtime_param0
// CHECK:STDOUT:     @CallSpecific.%x: %C = bind_name x, %x.loc18_17.1
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @CallSpecific.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.loc4: type](%x: @Function.%T.1 (%T)) -> @Function.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: type, %x: @CallGeneric.%T.1 (%T)) -> @CallGeneric.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.1 (%T) = call %Function.ref(%x.ref)
// CHECK:STDOUT:     %.loc9_21.1: @CallGeneric.%T.1 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_21.2: @CallGeneric.%T.1 (%T) = converted %Function.call, %.loc9_21.1
// CHECK:STDOUT:     return %.loc9_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %.1: type = ptr_type @CallGenericPtr.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc12: type, %x: @CallGenericPtr.%.1 (%.2)) -> @CallGenericPtr.%.1 (%.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%.1 (%.2) = name_ref x, %x
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%.1 (%.2) = call %Function.ref(%x.ref)
// CHECK:STDOUT:     %.loc13_21.1: @CallGenericPtr.%.1 (%.2) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_21.2: @CallGenericPtr.%.1 (%.2) = converted %Function.call, %.loc13_21.1
// CHECK:STDOUT:     return %.loc13_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %.loc18: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %Function.ref(%x.ref) to %.loc18
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%.2) {
// CHECK:STDOUT:   %T.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:

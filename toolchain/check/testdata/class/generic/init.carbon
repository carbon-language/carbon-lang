// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- from_struct.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  var k: T;
}

fn InitFromStructGeneric(T:! type, x: T) -> T {
  //@dump-sem-ir-begin
  var v: Class(T) = {.k = x};
  //@dump-sem-ir-end
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  //@dump-sem-ir-begin
  var v: Class(i32) = {.k = x};
  //@dump-sem-ir-end
  return v.k;
}

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T:! type) {
  adapt T;
}

fn InitFromAdaptedGeneric(T:! type, x: T) -> T {
  //@dump-sem-ir-begin
  return (x as Adapt(T)) as T;
  //@dump-sem-ir-end
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  //@dump-sem-ir-begin
  return (x as Adapt(i32)) as i32;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.k.b21: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %Class.247 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc8_26.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.loc10_17.2: type = class_type @Class, @Class(%T.loc8_26.2) [symbolic = %Class.loc10_17.2 (constants.%Class.fe1)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class.loc10_17.2 [symbolic = %require_complete.loc10 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %pattern_type.loc10: type = pattern_type %Class.loc10_17.2 [symbolic = %pattern_type.loc10 (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.loc8_26.2 (%T)} [symbolic = %struct_type.k (constants.%struct_type.k.b21)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.loc8_26.2 (%T)) -> @InitFromStructGeneric.%T.loc8_26.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.3c1) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.3c1) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.fe1) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.loc8_26.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.b21) = struct_literal (%x.ref)
// CHECK:STDOUT:     %.loc10_28.2: ref @InitFromStructGeneric.%T.loc8_26.2 (%T) = class_element_access %v.var, element0
// CHECK:STDOUT:     %.loc10_28.3: init @InitFromStructGeneric.%T.loc8_26.2 (%T) = initialize_from %x.ref to %.loc10_28.2
// CHECK:STDOUT:     %.loc10_28.4: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.fe1) = class_init (%.loc10_28.3), %v.var
// CHECK:STDOUT:     %.loc10_3: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.fe1) = converted %.loc10_28.1, %.loc10_28.4
// CHECK:STDOUT:     assign %v.var, %.loc10_3
// CHECK:STDOUT:     %.loc10_17: type = splice_block %Class.loc10_17.1 [symbolic = %Class.loc10_17.2 (constants.%Class.fe1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc10: type = name_ref T, %T.loc8_26.1 [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:       %Class.loc10_17.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc10_17.2 (constants.%Class.fe1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.fe1) = bind_name v, %v.var
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0fa = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0fa = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.247 = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc17_30.1: %struct_type.k.0bf = struct_literal (%x.ref)
// CHECK:STDOUT:   %.loc17_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc17_30.3: init %i32 = initialize_from %x.ref to %.loc17_30.2
// CHECK:STDOUT:   %.loc17_30.4: init %Class.247 = class_init (%.loc17_30.3), %v.var
// CHECK:STDOUT:   %.loc17_3: init %Class.247 = converted %.loc17_30.1, %.loc17_30.4
// CHECK:STDOUT:   assign %v.var, %.loc17_3
// CHECK:STDOUT:   %.loc17_19: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.247 = bind_name v, %v.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_26.2 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc8 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.2e4: type = class_type @Adapt, @Adapt(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.26c: <witness> = require_complete_type %Adapt.2e4 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapt.526: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc8_27.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Adapt.loc10_23.2: type = class_type @Adapt, @Adapt(%T.loc8_27.2) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.2e4)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Adapt.loc10_23.2 [symbolic = %require_complete.loc10 (constants.%require_complete.26c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T)) -> @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc10_22: type = name_ref T, %T.loc8_27.1 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %Adapt.loc10_23.1: type = class_type @Adapt, @Adapt(constants.%T) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.2e4)]
// CHECK:STDOUT:     %.loc10_13.1: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.2e4) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc10_13.2: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.2e4) = converted %x.ref, %.loc10_13.1
// CHECK:STDOUT:     %T.ref.loc10_29: type = name_ref T, %T.loc8_27.1 [symbolic = %T.loc8_27.2 (constants.%T)]
// CHECK:STDOUT:     %.loc10_26.1: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = as_compatible %.loc10_13.2
// CHECK:STDOUT:     %.loc10_26.2: @InitFromAdaptedGeneric.%T.loc8_27.2 (%T) = converted %.loc10_13.2, %.loc10_26.1
// CHECK:STDOUT:     return %.loc10_26.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %int_32.loc16_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.526]
// CHECK:STDOUT:   %.loc16_13.1: %Adapt.526 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc16_13.2: %Adapt.526 = converted %x.ref, %.loc16_13.1
// CHECK:STDOUT:   %int_32.loc16_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16_28.1: %i32 = as_compatible %.loc16_13.2
// CHECK:STDOUT:   %.loc16_28.2: %i32 = converted %.loc16_13.2, %.loc16_28.1
// CHECK:STDOUT:   return %.loc16_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_27.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:

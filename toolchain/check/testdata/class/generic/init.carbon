// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  var v: Class(T) = {.k = x};
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  var v: Class(i32) = {.k = x};
  return v.k;
}

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T:! type) {
  adapt T;
}

fn InitFromAdaptedGeneric(T:! type, x: T) -> T {
  return (x as Adapt(T)) as T;
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  return (x as Adapt(i32)) as i32;
}

// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [template]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [template]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.7 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .InitFromStructGeneric = %InitFromStructGeneric.decl
// CHECK:STDOUT:     .InitFromStructSpecific = %InitFromStructSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [template = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.loc8_26.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @InitFromStructGeneric.%T.loc8: type = bind_symbolic_name T 0, %T.loc8_26.1 [symbolic = @InitFromStructGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_39: type = name_ref T, @InitFromStructGeneric.%T.loc8 [symbolic = @InitFromStructGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc8_36.1: @InitFromStructGeneric.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @InitFromStructGeneric.%x: @InitFromStructGeneric.%T.1 (%T) = bind_name x, %x.loc8_36.1
// CHECK:STDOUT:     %T.ref.loc8_45: type = name_ref T, @InitFromStructGeneric.%T.loc8 [symbolic = @InitFromStructGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     @InitFromStructGeneric.%return: ref @InitFromStructGeneric.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [template = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %int.make_type_32.loc13_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_30.1: type = value_of_initializer %int.make_type_32.loc13_30 [template = i32]
// CHECK:STDOUT:     %.loc13_30.2: type = converted %int.make_type_32.loc13_30, %.loc13_30.1 [template = i32]
// CHECK:STDOUT:     %x.loc13_27.1: i32 = param x, runtime_param0
// CHECK:STDOUT:     @InitFromStructSpecific.%x: i32 = bind_name x, %x.loc13_27.1
// CHECK:STDOUT:     %int.make_type_32.loc13_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_38.1: type = value_of_initializer %int.make_type_32.loc13_38 [template = i32]
// CHECK:STDOUT:     %.loc13_38.2: type = converted %int.make_type_32.loc13_38, %.loc13_38.1 [template = i32]
// CHECK:STDOUT:     @InitFromStructSpecific.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: type = struct_type {.k: @Class.%T (%T)} [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness @Class.%.2 (%.3) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Class.%.1 (%.2) = field_decl k, element0 [template]
// CHECK:STDOUT:     %.loc6: <witness> = complete_type_witness %.3 [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .k = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.1: type = class_type @Class, @Class(%T.1) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = struct_type {.k: @InitFromStructGeneric.%T.1 (%T)} [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: type = unbound_element_type @InitFromStructGeneric.%Class.1 (%Class.2), @InitFromStructGeneric.%T.1 (%T) [symbolic = %.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: type, %x: @InitFromStructGeneric.%T.1 (%T)) -> @InitFromStructGeneric.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc9: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.1 (%Class.2) = var v
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.1 (%Class.2) = bind_name v, %v.var
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc9_28.1: @InitFromStructGeneric.%.1 (%.3) = struct_literal (%x.ref)
// CHECK:STDOUT:     %.loc9_28.2: ref @InitFromStructGeneric.%T.1 (%T) = class_element_access %v.var, element0
// CHECK:STDOUT:     %.loc9_28.3: init @InitFromStructGeneric.%T.1 (%T) = initialize_from %x.ref to %.loc9_28.2
// CHECK:STDOUT:     %.loc9_28.4: init @InitFromStructGeneric.%Class.1 (%Class.2) = class_init (%.loc9_28.3), %v.var
// CHECK:STDOUT:     %.loc9_29: init @InitFromStructGeneric.%Class.1 (%Class.2) = converted %.loc9_28.1, %.loc9_28.4
// CHECK:STDOUT:     assign %v.var, %.loc9_29
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.1 (%Class.2) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%.2 (%.2) = name_ref k, @Class.%.loc5 [template = @Class.%.loc5]
// CHECK:STDOUT:     %.loc10_11.1: ref @InitFromStructGeneric.%T.1 (%T) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc10_11.2: @InitFromStructGeneric.%T.1 (%T) = bind_value %.loc10_11.1
// CHECK:STDOUT:     return %.loc10_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %int.make_type_32, %.loc14_15.1 [template = i32]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(i32) [template = constants.%Class.3]
// CHECK:STDOUT:   %v.var: ref %Class.3 = var v
// CHECK:STDOUT:   %v: ref %Class.3 = bind_name v, %v.var
// CHECK:STDOUT:   %x.ref: i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc14_30.1: %.7 = struct_literal (%x.ref)
// CHECK:STDOUT:   %.loc14_30.2: ref i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc14_30.3: init i32 = initialize_from %x.ref to %.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.4: init %Class.3 = class_init (%.loc14_30.3), %v.var
// CHECK:STDOUT:   %.loc14_31: init %Class.3 = converted %.loc14_30.1, %.loc14_30.4
// CHECK:STDOUT:   assign %v.var, %.loc14_31
// CHECK:STDOUT:   %v.ref: ref %Class.3 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %.6 = name_ref k, @Class.%.loc5 [template = @Class.%.loc5]
// CHECK:STDOUT:   %.loc15_11.1: ref i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc15_11.2: i32 = bind_value %.loc15_11.1
// CHECK:STDOUT:   return %.loc15_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT:   %.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@InitFromStructGeneric.%T.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Adapt.1: %Adapt.type = struct_value () [template]
// CHECK:STDOUT:   %Adapt.2: type = class_type @Adapt, @Adapt(%T) [symbolic]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [template]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [template]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %Adapt.3: type = class_type @Adapt, @Adapt(i32) [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Adapt = %Adapt.decl
// CHECK:STDOUT:     .InitFromAdaptedGeneric = %InitFromAdaptedGeneric.decl
// CHECK:STDOUT:     .InitFromAdaptedSpecific = %InitFromAdaptedSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapt.decl: %Adapt.type = class_decl @Adapt [template = constants.%Adapt.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Adapt.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [template = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.loc8_27.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @InitFromAdaptedGeneric.%T.loc8: type = bind_symbolic_name T 0, %T.loc8_27.1 [symbolic = @InitFromAdaptedGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_40: type = name_ref T, @InitFromAdaptedGeneric.%T.loc8 [symbolic = @InitFromAdaptedGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc8_37.1: @InitFromAdaptedGeneric.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @InitFromAdaptedGeneric.%x: @InitFromAdaptedGeneric.%T.1 (%T) = bind_name x, %x.loc8_37.1
// CHECK:STDOUT:     %T.ref.loc8_46: type = name_ref T, @InitFromAdaptedGeneric.%T.loc8 [symbolic = @InitFromAdaptedGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     @InitFromAdaptedGeneric.%return: ref @InitFromAdaptedGeneric.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [template = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %int.make_type_32.loc12_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_31.1: type = value_of_initializer %int.make_type_32.loc12_31 [template = i32]
// CHECK:STDOUT:     %.loc12_31.2: type = converted %int.make_type_32.loc12_31, %.loc12_31.1 [template = i32]
// CHECK:STDOUT:     %x.loc12_28.1: i32 = param x, runtime_param0
// CHECK:STDOUT:     @InitFromAdaptedSpecific.%x: i32 = bind_name x, %x.loc12_28.1
// CHECK:STDOUT:     %int.make_type_32.loc12_39: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_39.1: type = value_of_initializer %int.make_type_32.loc12_39 [template = i32]
// CHECK:STDOUT:     %.loc12_39.2: type = converted %int.make_type_32.loc12_39, %.loc12_39.1 [template = i32]
// CHECK:STDOUT:     @InitFromAdaptedSpecific.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapt(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: <witness> = complete_type_witness @Adapt.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     adapt_decl %T
// CHECK:STDOUT:     %.loc6: <witness> = complete_type_witness %T [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapt.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Adapt.1: type = class_type @Adapt, @Adapt(%T.1) [symbolic = %Adapt.1 (constants.%Adapt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: type, %x: @InitFromAdaptedGeneric.%T.1 (%T)) -> @InitFromAdaptedGeneric.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [template = constants.%Adapt.1]
// CHECK:STDOUT:     %T.ref.loc9_22: type = name_ref T, %T.loc8 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %Adapt.loc9: type = class_type @Adapt, @Adapt(constants.%T) [symbolic = %Adapt.1 (constants.%Adapt.2)]
// CHECK:STDOUT:     %.loc9_13.1: @InitFromAdaptedGeneric.%Adapt.1 (%Adapt.2) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc9_13.2: @InitFromAdaptedGeneric.%Adapt.1 (%Adapt.2) = converted %x.ref, %.loc9_13.1
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc8 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc9_26.1: @InitFromAdaptedGeneric.%T.1 (%T) = as_compatible %.loc9_13.2
// CHECK:STDOUT:     %.loc9_26.2: @InitFromAdaptedGeneric.%T.1 (%T) = converted %.loc9_13.2, %.loc9_26.1
// CHECK:STDOUT:     return %.loc9_26.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [template = constants.%Adapt.1]
// CHECK:STDOUT:   %int.make_type_32.loc13_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_32.loc13_22 [template = i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_32.loc13_22, %.loc13_21.1 [template = i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(i32) [template = constants.%Adapt.3]
// CHECK:STDOUT:   %.loc13_13.1: %Adapt.3 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc13_13.2: %Adapt.3 = converted %x.ref, %.loc13_13.1
// CHECK:STDOUT:   %int.make_type_32.loc13_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_31.1: type = value_of_initializer %int.make_type_32.loc13_31 [template = i32]
// CHECK:STDOUT:   %.loc13_31.2: type = converted %int.make_type_32.loc13_31, %.loc13_31.1 [template = i32]
// CHECK:STDOUT:   %.loc13_28.1: i32 = as_compatible %.loc13_13.2
// CHECK:STDOUT:   %.loc13_28.2: i32 = converted %.loc13_13.2, %.loc13_28.1
// CHECK:STDOUT:   return %.loc13_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(@InitFromAdaptedGeneric.%T.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- init.carbon

class Class(T:! type) {
  var k: T;
}

fn InitFromStructGeneric(T:! type, x: T) -> T {
  var v: Class(T) = {.k = x};
  return v.k;
}

// --- fail_todo_init.carbon

library "fail_todo_init";

class Class(T:! type) {
  var k: T;
}

// TODO: The following should work.
fn InitFromStructSpecific() -> i32 {
  // CHECK:STDERR: fail_todo_init.carbon:[[@LINE+6]]:23: ERROR: Cannot implicitly convert from `i32` to `T`.
  // CHECK:STDERR:   var v: Class(i32) = {.k = 0};
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR: fail_todo_init.carbon:[[@LINE+3]]:23: Type `i32` does not implement interface `ImplicitAs`.
  // CHECK:STDERR:   var v: Class(i32) = {.k = 0};
  // CHECK:STDERR:                       ^~~~~~~~
  var v: Class(i32) = {.k = 0};
  return v.k;
}

// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [template]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .InitFromStructGeneric = %InitFromStructGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc2_13.1: type = param T
// CHECK:STDOUT:     %T.loc2_13.2: type = bind_symbolic_name T 0, %T.loc2_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [template = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.loc6_26.1: type = param T
// CHECK:STDOUT:     @InitFromStructGeneric.%T.loc6: type = bind_symbolic_name T 0, %T.loc6_26.1 [symbolic = @InitFromStructGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc6_39: type = name_ref T, @InitFromStructGeneric.%T.loc6 [symbolic = @InitFromStructGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc6_36.1: @InitFromStructGeneric.%T.1 (%T) = param x
// CHECK:STDOUT:     @InitFromStructGeneric.%x: @InitFromStructGeneric.%T.1 (%T) = bind_name x, %x.loc6_36.1
// CHECK:STDOUT:     %T.ref.loc6_45: type = name_ref T, @InitFromStructGeneric.%T.loc6 [symbolic = @InitFromStructGeneric.%T.1 (constants.%T)]
// CHECK:STDOUT:     @InitFromStructGeneric.%return: ref @InitFromStructGeneric.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc2_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc3: @Class.%.1 (%.2) = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .k = %.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc6: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = struct_type {.k: @InitFromStructGeneric.%T.1 (%T)} [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: type = unbound_element_type @InitFromStructGeneric.%Class (%Class.2), @InitFromStructGeneric.%T.1 (%T) [symbolic = %.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc6: type, %x: @InitFromStructGeneric.%T.1 (%T)) -> @InitFromStructGeneric.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_15: init type = call %Class.ref(%T.ref) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:     %.loc7_17.1: type = value_of_initializer %.loc7_15 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:     %.loc7_17.2: type = converted %.loc7_15, %.loc7_17.1 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class (%Class.2) = var v
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class (%Class.2) = bind_name v, %v.var
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc7_28.1: @InitFromStructGeneric.%.1 (%.3) = struct_literal (%x.ref)
// CHECK:STDOUT:     %.loc7_28.2: ref @InitFromStructGeneric.%T.1 (%T) = class_element_access %v.var, element0
// CHECK:STDOUT:     %.loc7_28.3: init @InitFromStructGeneric.%T.1 (%T) = initialize_from %x.ref to %.loc7_28.2
// CHECK:STDOUT:     %.loc7_28.4: init @InitFromStructGeneric.%Class (%Class.2) = class_init (%.loc7_28.3), %v.var
// CHECK:STDOUT:     %.loc7_29: init @InitFromStructGeneric.%Class (%Class.2) = converted %.loc7_28.1, %.loc7_28.4
// CHECK:STDOUT:     assign %v.var, %.loc7_29
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class (%Class.2) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%.2 (%.2) = name_ref k, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:     %.loc8_11.1: ref @InitFromStructGeneric.%T.1 (%T) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc8_11.2: @InitFromStructGeneric.%T.1 (%T) = bind_value %.loc8_11.1
// CHECK:STDOUT:     return %.loc8_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
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
// CHECK:STDOUT: --- fail_todo_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [template]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.8: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.8) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.8 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.8, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.11: type = interface_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %.11, %Convert.type.2 [symbolic]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.14: %.9 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.2 (%.9) = import_ref Core//prelude/operators/as, inst+59, loaded [symbolic = @ImplicitAs.%.3 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .InitFromStructSpecific = %InitFromStructSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [template = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_32.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc9_32.2: type = converted %int.make_type_32, %.loc9_32.1 [template = i32]
// CHECK:STDOUT:     @InitFromStructSpecific.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.8), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.9) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.3 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Class.%.1 (%.2) = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .k = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_15.2: type = converted %int.make_type_32, %.loc16_15.1 [template = i32]
// CHECK:STDOUT:   %.loc16_15.3: init type = call %Class.ref(%.loc16_15.2) [template = constants.%Class.3]
// CHECK:STDOUT:   %.loc16_19.1: type = value_of_initializer %.loc16_15.3 [template = constants.%Class.3]
// CHECK:STDOUT:   %.loc16_19.2: type = converted %.loc16_15.3, %.loc16_19.1 [template = constants.%Class.3]
// CHECK:STDOUT:   %v.var: ref %Class.3 = var v
// CHECK:STDOUT:   %v: ref %Class.3 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc16_29: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_30.1: %.7 = struct_literal (%.loc16_29)
// CHECK:STDOUT:   %.loc16_30.2: init type = call constants.%ImplicitAs(constants.%T) [symbolic = constants.%.11]
// CHECK:STDOUT:   %.loc16_30.3: %.12 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%T) [symbolic = constants.%.13]
// CHECK:STDOUT:   %Convert.ref: %.12 = name_ref Convert, %.loc16_30.3 [symbolic = constants.%.13]
// CHECK:STDOUT:   %.loc16_30.4: %T = converted %.loc16_29, <error> [template = <error>]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %v.ref: ref %Class.3 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %.4 = name_ref k, @Class.%.loc5 [template = @Class.%.loc5]
// CHECK:STDOUT:   %.loc17_11.1: ref i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc17_11.2: i32 = bind_value %.loc17_11.1
// CHECK:STDOUT:   return %.loc17_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%.1 (%.8)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %Dest => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT:   %.3 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:

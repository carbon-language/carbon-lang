// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/field.carbon

class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  return c.x;
}

fn G(T:! type, c: Class(T)) -> T {
  return c.x;
}

fn H(U:! type, c: Class(U)) -> U {
  return c.x;
}

// CHECK:STDOUT: --- field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Class.4, %U [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Class.ref.loc15: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc15_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_14.1: type = value_of_initializer %int.make_type_32.loc15_15 [template = i32]
// CHECK:STDOUT:     %.loc15_14.2: type = converted %int.make_type_32.loc15_15, %.loc15_14.1 [template = i32]
// CHECK:STDOUT:     %.loc15_14.3: init type = call %Class.ref.loc15(%.loc15_14.2) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc15_18.1: type = value_of_initializer %.loc15_14.3 [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc15_18.2: type = converted %.loc15_14.3, %.loc15_18.1 [template = constants.%Class.3]
// CHECK:STDOUT:     %c.loc15_6.1: %Class.3 = param c
// CHECK:STDOUT:     @F.%c: %Class.3 = bind_name c, %c.loc15_6.1
// CHECK:STDOUT:     %int.make_type_32.loc15_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_24.1: type = value_of_initializer %int.make_type_32.loc15_24 [template = i32]
// CHECK:STDOUT:     %.loc15_24.2: type = converted %int.make_type_32.loc15_24, %.loc15_24.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.loc19_6.1: type = param T
// CHECK:STDOUT:     @G.%T.loc19: type = bind_symbolic_name T 0, %T.loc19_6.1 [symbolic = @G.%T.1 (constants.%T)]
// CHECK:STDOUT:     %Class.ref.loc19: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref.loc19_25: type = name_ref T, @G.%T.loc19 [symbolic = @G.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc19_24: init type = call %Class.ref.loc19(%T.ref.loc19_25) [symbolic = @G.%Class (constants.%Class.2)]
// CHECK:STDOUT:     %.loc19_26.1: type = value_of_initializer %.loc19_24 [symbolic = @G.%Class (constants.%Class.2)]
// CHECK:STDOUT:     %.loc19_26.2: type = converted %.loc19_24, %.loc19_26.1 [symbolic = @G.%Class (constants.%Class.2)]
// CHECK:STDOUT:     %c.loc19_16.1: @G.%Class (%Class.2) = param c
// CHECK:STDOUT:     @G.%c: @G.%Class (%Class.2) = bind_name c, %c.loc19_16.1
// CHECK:STDOUT:     %T.ref.loc19_32: type = name_ref T, @G.%T.loc19 [symbolic = @G.%T.1 (constants.%T)]
// CHECK:STDOUT:     @G.%return: ref @G.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %U.loc23_6.1: type = param U
// CHECK:STDOUT:     @H.%U.loc23: type = bind_symbolic_name U 0, %U.loc23_6.1 [symbolic = @H.%U.1 (constants.%U)]
// CHECK:STDOUT:     %Class.ref.loc23: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %U.ref.loc23_25: type = name_ref U, @H.%U.loc23 [symbolic = @H.%U.1 (constants.%U)]
// CHECK:STDOUT:     %.loc23_24: init type = call %Class.ref.loc23(%U.ref.loc23_25) [symbolic = @H.%Class (constants.%Class.4)]
// CHECK:STDOUT:     %.loc23_26.1: type = value_of_initializer %.loc23_24 [symbolic = @H.%Class (constants.%Class.4)]
// CHECK:STDOUT:     %.loc23_26.2: type = converted %.loc23_24, %.loc23_26.1 [symbolic = @H.%Class (constants.%Class.4)]
// CHECK:STDOUT:     %c.loc23_16.1: @H.%Class (%Class.4) = param c
// CHECK:STDOUT:     @H.%c: @H.%Class (%Class.4) = bind_name c, %c.loc23_16.1
// CHECK:STDOUT:     %U.ref.loc23_32: type = name_ref U, @H.%U.loc23 [symbolic = @H.%U.1 (constants.%U)]
// CHECK:STDOUT:     @H.%return: ref @H.%U.1 (%U) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc12: @Class.%.1 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .x = %.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %.4 = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc16_11.1: ref i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: i32 = bind_value %.loc16_11.1
// CHECK:STDOUT:   return %.loc16_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc19: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @G.%Class (%Class.2), @G.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc19: type, %c: @G.%Class (%Class.2)) -> @G.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class (%Class.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%.1 (%.2) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:     %.loc20_11.1: ref @G.%T.1 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc20_11.2: @G.%T.1 (%T) = bind_value %.loc20_11.1
// CHECK:STDOUT:     return %.loc20_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc23: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%U.1) [symbolic = %Class (constants.%Class.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @H.%Class (%Class.4), @H.%U.1 (%U) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc23: type, %c: @H.%Class (%Class.4)) -> @H.%U.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class (%Class.4) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%.1 (%.6) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:     %.loc24_11.1: ref @H.%U.1 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc24_11.2: @H.%U.1 (%U) = bind_value %.loc24_11.1
// CHECK:STDOUT:     return %.loc24_11.2
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
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.4
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@H.%U.1) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %Class => constants.%Class.4
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type %Class.4, %U [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.x: %U} [symbolic]
// CHECK:STDOUT:   %.12: <witness> = complete_type_witness %.11 [symbolic]
// CHECK:STDOUT:   %.13: type = ptr_type %.11 [symbolic]
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
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: %Class.3 = binding_pattern c
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc15_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_14.1: type = value_of_initializer %int.make_type_32.loc15_15 [template = i32]
// CHECK:STDOUT:     %.loc15_14.2: type = converted %int.make_type_32.loc15_15, %.loc15_14.1 [template = i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(i32) [template = constants.%Class.3]
// CHECK:STDOUT:     %c.param: %Class.3 = param c, runtime_param0
// CHECK:STDOUT:     %c: %Class.3 = bind_name c, %c.param
// CHECK:STDOUT:     %int.make_type_32.loc15_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_24.1: type = value_of_initializer %int.make_type_32.loc15_24 [template = i32]
// CHECK:STDOUT:     %.loc15_24.2: type = converted %int.make_type_32.loc15_24, %.loc15_24.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %c.patt: @G.%Class.loc19_24.2 (%Class.2) = binding_pattern c
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref.loc19_25: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc19_24.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc19_24.2 (constants.%Class.2)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc19_24.2 (%Class.2) = param c, runtime_param0
// CHECK:STDOUT:     %c: @G.%Class.loc19_24.2 (%Class.2) = bind_name c, %c.param
// CHECK:STDOUT:     %T.ref.loc19_32: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %return: ref @G.%T.loc19_6.2 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %U.patt: type = symbolic_binding_pattern U, 0
// CHECK:STDOUT:     %c.patt: @H.%Class.loc23_24.2 (%Class.4) = binding_pattern c
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc23_6.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %U.ref.loc23_25: type = name_ref U, %U.loc23_6.1 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %Class.loc23_24.1: type = class_type @Class, @Class(constants.%U) [symbolic = %Class.loc23_24.2 (constants.%Class.4)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc23_24.2 (%Class.4) = param c, runtime_param0
// CHECK:STDOUT:     %c: @H.%Class.loc23_24.2 (%Class.4) = bind_name c, %c.param
// CHECK:STDOUT:     %U.ref.loc23_32: type = name_ref U, %U.loc23_6.1 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:     %return: ref @H.%U.loc23_6.2 (%U) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc12_8.2: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.loc11_13.2 (%T) [symbolic = %.loc12_8.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc13_1.2: type = struct_type {.x: @Class.%T.loc11_13.2 (%T)} [symbolic = %.loc13_1.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc13_1.3: <witness> = complete_type_witness @Class.%.loc13_1.2 (%.3) [symbolic = %.loc13_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_8.1: @Class.%.loc12_8.2 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:     %.loc13_1.1: <witness> = complete_type_witness %.3 [symbolic = %.loc13_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .x = %.loc12_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %.5 = name_ref x, @Class.%.loc12_8.1 [template = @Class.%.loc12_8.1]
// CHECK:STDOUT:   %.loc16_11.1: ref i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: i32 = bind_value %.loc16_11.1
// CHECK:STDOUT:   return %.loc16_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc19_6.1: type) {
// CHECK:STDOUT:   %T.loc19_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:   %Class.loc19_24.2: type = class_type @Class, @Class(%T.loc19_6.2) [symbolic = %Class.loc19_24.2 (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc20_11.3: type = unbound_element_type @G.%Class.loc19_24.2 (%Class.2), @G.%T.loc19_6.2 (%T) [symbolic = %.loc20_11.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc19_6.1: type, %c: @G.%Class.loc19_24.2 (%Class.2)) -> @G.%T.loc19_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc19_24.2 (%Class.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%.loc20_11.3 (%.2) = name_ref x, @Class.%.loc12_8.1 [template = @Class.%.loc12_8.1]
// CHECK:STDOUT:     %.loc20_11.1: ref @G.%T.loc19_6.2 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc20_11.2: @G.%T.loc19_6.2 (%T) = bind_value %.loc20_11.1
// CHECK:STDOUT:     return %.loc20_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc23_6.1: type) {
// CHECK:STDOUT:   %U.loc23_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc23_6.2 (constants.%U)]
// CHECK:STDOUT:   %Class.loc23_24.2: type = class_type @Class, @Class(%U.loc23_6.2) [symbolic = %Class.loc23_24.2 (constants.%Class.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc24_11.3: type = unbound_element_type @H.%Class.loc23_24.2 (%Class.4), @H.%U.loc23_6.2 (%U) [symbolic = %.loc24_11.3 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc23_6.1: type, %c: @H.%Class.loc23_24.2 (%Class.4)) -> @H.%U.loc23_6.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc23_24.2 (%Class.4) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%.loc24_11.3 (%.10) = name_ref x, @Class.%.loc12_8.1 [template = @Class.%.loc12_8.1]
// CHECK:STDOUT:     %.loc24_11.1: ref @H.%U.loc23_6.2 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc24_11.2: @H.%U.loc23_6.2 (%U) = bind_value %.loc24_11.1
// CHECK:STDOUT:     return %.loc24_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.loc12_8.2 => constants.%.2
// CHECK:STDOUT:   %.loc13_1.2 => constants.%.3
// CHECK:STDOUT:   %.loc13_1.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T.loc11_13.2 => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.loc12_8.2 => constants.%.5
// CHECK:STDOUT:   %.loc13_1.2 => constants.%.6
// CHECK:STDOUT:   %.loc13_1.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T.loc19_6.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%T
// CHECK:STDOUT:   %Class.loc19_24.2 => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.4
// CHECK:STDOUT:   %.loc12_8.2 => constants.%.10
// CHECK:STDOUT:   %.loc13_1.2 => constants.%.11
// CHECK:STDOUT:   %.loc13_1.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@H.%U.loc23_6.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.loc23_6.2 => constants.%U
// CHECK:STDOUT:   %Class.loc23_24.2 => constants.%Class.4
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %Class.2: type = class_type @Class, file.%Class.decl(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, file.%Class.decl(i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, file.%Class.decl(%U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Class.4, %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref ir3, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref ir3, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
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
// CHECK:STDOUT:     @G.%T: type = bind_symbolic_name T 0, %T.loc19_6.1 [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:     %Class.ref.loc19: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref.loc19_25: type = name_ref T, @G.%T [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:     %.loc19_24: init type = call %Class.ref.loc19(%T.ref.loc19_25) [symbolic = %.loc19_24 (constants.%Class.2)]
// CHECK:STDOUT:     %.loc19_26.1: type = value_of_initializer %.loc19_24 [symbolic = %.loc19_24 (constants.%Class.2)]
// CHECK:STDOUT:     %.loc19_26.2: type = converted %.loc19_24, %.loc19_26.1 [symbolic = %.loc19_24 (constants.%Class.2)]
// CHECK:STDOUT:     %c.loc19_16.1: file.%.loc19_24 (%Class.2) = param c
// CHECK:STDOUT:     @G.%c: file.%.loc19_24 (%Class.2) = bind_name c, %c.loc19_16.1
// CHECK:STDOUT:     %T.ref.loc19_32: type = name_ref T, @G.%T [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:     @G.%return: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %U.loc23_6.1: type = param U
// CHECK:STDOUT:     @H.%U: type = bind_symbolic_name U 0, %U.loc23_6.1 [symbolic = @H.%U (constants.%U)]
// CHECK:STDOUT:     %Class.ref.loc23: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %U.ref.loc23_25: type = name_ref U, @H.%U [symbolic = @H.%U (constants.%U)]
// CHECK:STDOUT:     %.loc23_24: init type = call %Class.ref.loc23(%U.ref.loc23_25) [symbolic = %.loc23_24 (constants.%Class.4)]
// CHECK:STDOUT:     %.loc23_26.1: type = value_of_initializer %.loc23_24 [symbolic = %.loc23_24 (constants.%Class.4)]
// CHECK:STDOUT:     %.loc23_26.2: type = converted %.loc23_24, %.loc23_26.1 [symbolic = %.loc23_24 (constants.%Class.4)]
// CHECK:STDOUT:     %c.loc23_16.1: file.%.loc23_24 (%Class.4) = param c
// CHECK:STDOUT:     @H.%c: file.%.loc23_24 (%Class.4) = bind_name c, %c.loc23_16.1
// CHECK:STDOUT:     %U.ref.loc23_32: type = name_ref U, @H.%U [symbolic = @H.%U (constants.%U)]
// CHECK:STDOUT:     @H.%return: ref %U = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT:   %T.ref: type = name_ref T, file.%T.loc11_13.2 [symbolic = file.%T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %.loc12: <unexpected>.inst+18 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .x = %.loc12
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
// CHECK:STDOUT: fn @G(%T: type, %c: file.%.loc19_24 (%Class.2)) -> %T
// CHECK:STDOUT:     generic [%T: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: file.%.loc19_24 (%Class.2) = name_ref c, %c
// CHECK:STDOUT:   %x.ref: <unexpected>.inst+68 (%.2) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc20_11.1: ref @G.%T (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc20_11.2: @G.%T (%T) = bind_value %.loc20_11.1
// CHECK:STDOUT:   return %.loc20_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%U: type, %c: file.%.loc23_24 (%Class.4)) -> %U
// CHECK:STDOUT:     generic [%U: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: file.%.loc23_24 (%Class.4) = name_ref c, %c
// CHECK:STDOUT:   %x.ref: <unexpected>.inst+91 (%.6) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc24_11.1: ref @H.%U (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc24_11.2: @H.%U (%U) = bind_value %.loc24_11.1
// CHECK:STDOUT:   return %.loc24_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: definition:
// CHECK:STDOUT:   <unexpected>.inst+17 => constants.%Class.2
// CHECK:STDOUT:   <unexpected>.inst+18 => constants.%.2
// CHECK:STDOUT:   <unexpected>.inst+14.loc12_8 => <unexpected>.inst+15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(file.%T.loc11_13.2) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(i32) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => i32
// CHECK:STDOUT:
// CHECK:STDOUT: definition:
// CHECK:STDOUT:   <unexpected>.inst+17 => constants.%Class.3
// CHECK:STDOUT:   <unexpected>.inst+18 => constants.%.4
// CHECK:STDOUT:   <unexpected>.inst+14.loc12_8 => <unexpected>.inst+15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%G.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @G.%T => constants.%T
// CHECK:STDOUT:   file.%.loc19_24 => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(constants.%U) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: definition:
// CHECK:STDOUT:   <unexpected>.inst+17 => constants.%Class.4
// CHECK:STDOUT:   <unexpected>.inst+18 => constants.%.6
// CHECK:STDOUT:   <unexpected>.inst+14.loc12_8 => <unexpected>.inst+15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%H.decl(constants.%U) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @H.%U => constants.%U
// CHECK:STDOUT:   file.%.loc23_24 => constants.%Class.4
// CHECK:STDOUT: }
// CHECK:STDOUT:

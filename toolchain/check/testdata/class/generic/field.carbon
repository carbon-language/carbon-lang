// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/field.carbon
// CHECK:STDERR: fail_todo_field.carbon: ERROR: Main//default previously provided by `field.carbon`.
// CHECK:STDERR:

// --- field.carbon

class Class(T:! type) {
  var x: T;
}

// TODO: This case only works because the parameters of Class and G have the
// same name and index, so canonicalize to the same type.
fn G(T:! type, c: Class(T)) -> T {
  return c.x;
}

// --- fail_todo_field.carbon

class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  // CHECK:STDERR: fail_todo_field.carbon:[[@LINE+8]]:3: ERROR: Cannot implicitly convert from `T` to `i32`.
  // CHECK:STDERR:   return c.x;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_field.carbon:[[@LINE+4]]:10: ERROR: Cannot implicitly convert from `Class` to `Class`.
  // CHECK:STDERR:   return c.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return c.x;
}

fn H(U:! type, c: Class(U)) -> U {
  // CHECK:STDERR: fail_todo_field.carbon:[[@LINE+7]]:3: ERROR: Cannot implicitly convert from `T` to `U`.
  // CHECK:STDERR:   return c.x;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_field.carbon:[[@LINE+3]]:10: ERROR: Cannot implicitly convert from `Class` to `Class`.
  // CHECK:STDERR:   return c.x;
  // CHECK:STDERR:          ^~~
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
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc2_13.1: type = param T
// CHECK:STDOUT:     %T.loc2_13.2: type = bind_symbolic_name T 0, %T.loc2_13.1 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.loc8_6.1: type = param T
// CHECK:STDOUT:     @G.%T: type = bind_symbolic_name T 0, %T.loc8_6.1 [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref.loc8_25: type = name_ref T, @G.%T [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:     %.loc8_24: init type = call %Class.ref(%T.ref.loc8_25) [symbolic = %.loc8_24 (constants.%Class.2)]
// CHECK:STDOUT:     %.loc8_26.1: type = value_of_initializer %.loc8_24 [symbolic = %.loc8_24 (constants.%Class.2)]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_24, %.loc8_26.1 [symbolic = %.loc8_24 (constants.%Class.2)]
// CHECK:STDOUT:     %c.loc8_16.1: file.%.loc8_24 (%Class.2) = param c
// CHECK:STDOUT:     @G.%c: file.%.loc8_24 (%Class.2) = bind_name c, %c.loc8_16.1
// CHECK:STDOUT:     %T.ref.loc8_32: type = name_ref T, @G.%T [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:     @G.%return: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc2_13.2: type] {
// CHECK:STDOUT:   %T.ref: type = name_ref T, file.%T.loc2_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc3: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .x = %.loc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%T: type, %c: file.%.loc8_24 (%Class.2)) -> %T
// CHECK:STDOUT:     generic [%T: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.2 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %.2 = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:   %.loc9_11.1: ref %T = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc9_11.2: %T = bind_value %.loc9_11.1
// CHECK:STDOUT:   return %.loc9_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%Class.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%G.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @G.%T => constants.%T
// CHECK:STDOUT:   file.%.loc8_24 => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_field.carbon
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
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, file.%Class.decl(%U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
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
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc2_13.1: type = param T
// CHECK:STDOUT:     %T.loc2_13.2: type = bind_symbolic_name T 0, %T.loc2_13.1 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc6_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_14.1: type = value_of_initializer %int.make_type_32.loc6_15 [template = i32]
// CHECK:STDOUT:     %.loc6_14.2: type = converted %int.make_type_32.loc6_15, %.loc6_14.1 [template = i32]
// CHECK:STDOUT:     %.loc6_14.3: init type = call %Class.ref.loc6(%.loc6_14.2) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc6_18.1: type = value_of_initializer %.loc6_14.3 [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc6_18.2: type = converted %.loc6_14.3, %.loc6_18.1 [template = constants.%Class.3]
// CHECK:STDOUT:     %c.loc6_6.1: %Class.3 = param c
// CHECK:STDOUT:     @F.%c: %Class.3 = bind_name c, %c.loc6_6.1
// CHECK:STDOUT:     %int.make_type_32.loc6_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_24.1: type = value_of_initializer %int.make_type_32.loc6_24 [template = i32]
// CHECK:STDOUT:     %.loc6_24.2: type = converted %int.make_type_32.loc6_24, %.loc6_24.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %U.loc18_6.1: type = param U
// CHECK:STDOUT:     @H.%U: type = bind_symbolic_name U 0, %U.loc18_6.1 [symbolic = @H.%U (constants.%U)]
// CHECK:STDOUT:     %Class.ref.loc18: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %U.ref.loc18_25: type = name_ref U, @H.%U [symbolic = @H.%U (constants.%U)]
// CHECK:STDOUT:     %.loc18_24: init type = call %Class.ref.loc18(%U.ref.loc18_25) [symbolic = %.loc18_24 (constants.%Class.4)]
// CHECK:STDOUT:     %.loc18_26.1: type = value_of_initializer %.loc18_24 [symbolic = %.loc18_24 (constants.%Class.4)]
// CHECK:STDOUT:     %.loc18_26.2: type = converted %.loc18_24, %.loc18_26.1 [symbolic = %.loc18_24 (constants.%Class.4)]
// CHECK:STDOUT:     %c.loc18_16.1: file.%.loc18_24 (%Class.4) = param c
// CHECK:STDOUT:     @H.%c: file.%.loc18_24 (%Class.4) = bind_name c, %c.loc18_16.1
// CHECK:STDOUT:     %U.ref.loc18_32: type = name_ref U, @H.%U [symbolic = @H.%U (constants.%U)]
// CHECK:STDOUT:     @H.%return: ref %U = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc2_13.2: type] {
// CHECK:STDOUT:   %T.ref: type = name_ref T, file.%T.loc2_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc3: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .x = %.loc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %.2 = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:   %.loc15: %T = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%U: type, %c: file.%.loc18_24 (%Class.4)) -> %U
// CHECK:STDOUT:     generic [%U: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.4 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %.2 = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:   %.loc26: %T = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%Class.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%Class.decl(i32) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc2_13.2 => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%Class.decl(constants.%U) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc2_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%H.decl(constants.%U) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @H.%U => constants.%U
// CHECK:STDOUT:   file.%.loc18_24 => constants.%Class.4
// CHECK:STDOUT: }
// CHECK:STDOUT:

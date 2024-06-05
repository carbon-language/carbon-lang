// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_function_unqualified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_function_unqualified.carbon

base class Base {
  fn F();
}

class Derived {
  extend base: Base;

  fn G() { F(); }
  fn H();
}

fn Derived.H() {
  F();
}

// CHECK:STDOUT: --- base_function_unqualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: <type of G> = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: <type of H> = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: Base} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %H.decl: <type of H> = fn_decl @H [template = constants.%H] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc16: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %G.decl: <type of G> = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT:   %H.decl: <type of H> = fn_decl @H [template = constants.%H] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <type of F> = name_ref F, @Base.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <type of F> = name_ref F, @Base.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

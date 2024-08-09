// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/generic_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/generic_import.carbon

// --- a.carbon

library "a";

interface AddWith(T:! type) {
  fn F();
}

// --- b.carbon

library "b";
import library "a";

class C {}

impl C as AddWith(C) {
  fn F() {}
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type: type = generic_interface_type @AddWith [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %AddWith: %AddWith.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, %F.type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @AddWith.%F.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type = interface_decl @AddWith [template = constants.%AddWith] {
// CHECK:STDOUT:     %T.loc4_19.1: type = param T
// CHECK:STDOUT:     %T.loc4_19.2: type = bind_symbolic_name T 0, %T.loc4_19.1 [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(file.%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @AddWith, @AddWith(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @AddWith(%T) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @AddWith.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @AddWith.%.1 (%.2), @AddWith.%F.type (%F.type) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: @AddWith.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @AddWith.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @AddWith.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {}
// CHECK:STDOUT:     %.loc5: @AddWith.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc5
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(file.%T.loc4_19.2: type, @AddWith.%Self.1: @AddWith.%.1 (%.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(@AddWith.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %AddWith.type: type = generic_interface_type @AddWith [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %AddWith: %AddWith.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @AddWith.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.3, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @AddWith, @AddWith(%C) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @AddWith(%C) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %F.type.3 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %AddWith.type = import_ref Main//a, inst+4, loaded [template = constants.%AddWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//a, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//a, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.4: @AddWith.%F.type (%F.type.1) = import_ref Main//a, inst+12, loaded [symbolic = @AddWith.%F (constants.%F.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+12, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AddWith = imports.%import_ref.1
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %.loc7_20.1: type = value_of_initializer %.loc7_18 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_20.2: type = converted %.loc7_18, %.loc7_20.1 [template = constants.%.6]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc7_6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type = name_ref AddWith, imports.%import_ref.1 [template = constants.%AddWith]
// CHECK:STDOUT:     %C.ref.loc7_19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc7_18: init type = call %AddWith.ref(%C.ref.loc7_19) [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @AddWith, @AddWith(%T) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @AddWith(%T) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @AddWith.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @AddWith.%.1 (%.3), @AddWith.%F.type (%F.type.1) [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %.3: @AddWith.%.2 (%.4) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .F = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.6 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self.1: @AddWith.%.1 (%.3)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(@AddWith.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%C) {}
// CHECK:STDOUT:

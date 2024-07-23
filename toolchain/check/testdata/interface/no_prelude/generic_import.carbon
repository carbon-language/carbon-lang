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
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @AddWith, %F.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @AddWith.%F.decl [template]
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
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @AddWith.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:     %.loc5: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0, <unexpected>.inst+14 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @AddWith, invalid(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @AddWith, invalid(%C) [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %AddWith.type = import_ref Main//a, inst+4, loaded [template = constants.%AddWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//a, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//a, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.4: %F.type.2 = import_ref Main//a, inst+12, loaded [template = constants.%F.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AddWith = imports.%import_ref.1
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %.loc7_20.1: type = value_of_initializer %.loc7_18 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_20.2: type = converted %.loc7_18, %.loc7_20.1 [template = constants.%.4]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc7_6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type = name_ref AddWith, imports.%import_ref.1 [template = constants.%AddWith]
// CHECK:STDOUT:     %C.ref.loc7_19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc7_18: init type = call %AddWith.ref(%C.ref.loc7_19) [template = constants.%.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddWith {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .F = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.4 {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.5]
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
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific invalid(constants.%T);
// CHECK:STDOUT:
// CHECK:STDOUT: specific invalid(constants.%C);
// CHECK:STDOUT:

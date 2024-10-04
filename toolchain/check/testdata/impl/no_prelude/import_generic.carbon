// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_generic.carbon

// --- import_generic.carbon

library "[[@TEST_NAME]]";

class C {}
interface I(T:! type) {}

impl forall [T:! type] C as I(T);
impl forall [T:! type] C as I(T*) {}

// --- import_generic.impl.carbon

impl library "[[@TEST_NAME]]";

impl forall [T:! type] C as I(T) {}

// --- todo_fail_import_generic.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: this should fail because it's redefining this impl.
impl forall [T:! type] C as I(T*) {}

// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @I, @I(%.5) [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.patt.loc5: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc5, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc5: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc7: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc7, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7: type = interface_type @I, @I(constants.%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc8: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc8, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_32: type = ptr_type %T [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:     %.loc8_30: type = interface_type @I, @I(constants.%.5) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%.1 (%.4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc7: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc7;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.3)]
// CHECK:STDOUT:   %.1: type = ptr_type @impl.2.%T.1 (%T) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%.1) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc8_30 {
// CHECK:STDOUT:     %.loc8_35: <witness> = interface_witness () [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc8_35
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.5) {
// CHECK:STDOUT:   %T.1 => constants.%.5
// CHECK:STDOUT:   %T.patt.1 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%.1) {
// CHECK:STDOUT:   %T.1 => constants.%.5
// CHECK:STDOUT:   %T.patt.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @I.%.1 (%.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @I, @I(%.5) [symbolic]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//import_generic, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %I.type = import_ref Main//import_generic, inst+12, loaded [template = constants.%I]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//import_generic, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//import_generic, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.5: type = import_ref Main//import_generic, inst+28, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.6: type = import_ref Main//import_generic, inst+31, loaded [symbolic = @impl.1.%.1 (constants.%.2)]
// CHECK:STDOUT:   %import_ref.7: type = import_ref Main//import_generic, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8: type = import_ref Main//import_generic, inst+46, loaded [symbolic = @impl.2.%.2 (constants.%.6)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//import_generic, inst+53, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, imports.%import_ref.2 [template = constants.%I]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_30: type = interface_type @I, @I(constants.%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.1)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.5 as imports.%import_ref.6;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.1)]
// CHECK:STDOUT:   %.1: type = ptr_type @impl.2.%T (%T) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%.1) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.7 as imports.%import_ref.8 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc4_30 {
// CHECK:STDOUT:     %.loc4_34: <witness> = interface_witness () [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc4_34
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.5) {
// CHECK:STDOUT:   %T => constants.%.5
// CHECK:STDOUT:   %T.patt => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%.1) {
// CHECK:STDOUT:   %T => constants.%.5
// CHECK:STDOUT:   %T.patt => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.3.%T.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: @I.%.1 (%.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @I, @I(%.5) [symbolic]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//import_generic, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %I.type = import_ref Main//import_generic, inst+12, loaded [template = constants.%I]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//import_generic, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//import_generic, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.5: type = import_ref Main//import_generic, inst+28, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.6: type = import_ref Main//import_generic, inst+31, loaded [symbolic = @impl.1.%.1 (constants.%.2)]
// CHECK:STDOUT:   %import_ref.7: type = import_ref Main//import_generic, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8: type = import_ref Main//import_generic, inst+46, loaded [symbolic = @impl.2.%.2 (constants.%.6)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//import_generic, inst+53, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc5: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc5, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, imports.%import_ref.2 [template = constants.%I]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5_32: type = ptr_type %T [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:     %.loc5_30: type = interface_type @I, @I(constants.%.5) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc5: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.1)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.5 as imports.%import_ref.6;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.1)]
// CHECK:STDOUT:   %.1: type = ptr_type @impl.2.%T (%T) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%.1) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.7 as imports.%import_ref.8 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc5: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @impl.3.%T.1 (%T) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%.1) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc5_30 {
// CHECK:STDOUT:     %.loc5_35: <witness> = interface_witness () [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc5_35
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.5) {
// CHECK:STDOUT:   %T => constants.%.5
// CHECK:STDOUT:   %T.patt => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%.1) {
// CHECK:STDOUT:   %T => constants.%.5
// CHECK:STDOUT:   %T.patt => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.3.%.1) {
// CHECK:STDOUT:   %T => constants.%.5
// CHECK:STDOUT:   %T.patt => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:

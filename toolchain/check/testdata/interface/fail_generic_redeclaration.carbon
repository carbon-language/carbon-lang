// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_generic_redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_generic_redeclaration.carbon

interface NotGeneric;
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: interface NotGeneric(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: interface NotGeneric;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface NotGeneric(T:! type) {}

interface Generic(T:! type);
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+7]]:1: error: redeclaration differs because of missing parameter list [RedeclParamListDiffers]
// CHECK:STDERR: interface Generic {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:1: note: previously declared with parameter list [RedeclParamListPrevious]
// CHECK:STDERR: interface Generic(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Generic {}

interface DifferentParams(T:! type);
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+7]]:27: error: type `<pattern for ()>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: interface DifferentParams(T:! ()) {}
// CHECK:STDERR:                           ^
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:27: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: interface DifferentParams(T:! type);
// CHECK:STDERR:                           ^
// CHECK:STDERR:
interface DifferentParams(T:! ()) {}

// CHECK:STDOUT: --- fail_generic_redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGeneric.type.6a2: type = facet_type <@NotGeneric.loc15> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %NotGeneric.type.93b: type = generic_interface_type @NotGeneric.loc23 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NotGeneric.generic: %NotGeneric.type.93b = struct_value () [concrete]
// CHECK:STDOUT:   %NotGeneric.type.50a: type = facet_type <@NotGeneric.loc23, @NotGeneric.loc23(%T.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.242: %NotGeneric.type.50a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic.loc25 [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.29a: type = facet_type <@Generic.loc33> [concrete]
// CHECK:STDOUT:   %Self.36a: %Generic.type.29a = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type.d40e5c.1: type = generic_interface_type @DifferentParams.loc35 [concrete]
// CHECK:STDOUT:   %DifferentParams.generic.d33670.1: %DifferentParams.type.d40e5c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.d76: %empty_tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %DifferentParams.type.d40e5c.2: type = generic_interface_type @DifferentParams.loc43 [concrete]
// CHECK:STDOUT:   %DifferentParams.generic.d33670.2: %DifferentParams.type.d40e5c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DifferentParams.type.0ec: type = facet_type <@DifferentParams.loc43, @DifferentParams.loc43(%T.d76)> [symbolic]
// CHECK:STDOUT:   %Self.c69: %DifferentParams.type.0ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl.loc15
// CHECK:STDOUT:     .Generic = %Generic.decl.loc25
// CHECK:STDOUT:     .DifferentParams = %DifferentParams.decl.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGeneric.decl.loc15: type = interface_decl @NotGeneric.loc15 [concrete = constants.%NotGeneric.type.6a2] {} {}
// CHECK:STDOUT:   %NotGeneric.decl.loc23: %NotGeneric.type.93b = interface_decl @NotGeneric.loc23 [concrete = constants.%NotGeneric.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc23_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc23_22.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc25: %Generic.type.c21 = interface_decl @Generic.loc25 [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc25_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc25_19.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc33: type = interface_decl @Generic.loc33 [concrete = constants.%Generic.type.29a] {} {}
// CHECK:STDOUT:   %DifferentParams.decl.loc35: %DifferentParams.type.d40e5c.1 = interface_decl @DifferentParams.loc35 [concrete = constants.%DifferentParams.generic.d33670.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc35_27.2: type = symbolic_binding T, 0 [symbolic = %T.loc35_27.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentParams.decl.loc43: %DifferentParams.type.d40e5c.2 = interface_decl @DifferentParams.loc43 [concrete = constants.%DifferentParams.generic.d33670.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.cb1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc43_32.1: type = splice_block %.loc43_32.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc43_32.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc43_32.3: type = converted %.loc43_32.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc43_27.2: %empty_tuple.type = symbolic_binding T, 0 [symbolic = %T.loc43_27.1 (constants.%T.d76)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGeneric.loc15;
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @NotGeneric.loc23(%T.loc23_22.2: type) {
// CHECK:STDOUT:   %T.loc23_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc23_22.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NotGeneric.type: type = facet_type <@NotGeneric.loc23, @NotGeneric.loc23(%T.loc23_22.1)> [symbolic = %NotGeneric.type (constants.%NotGeneric.type.50a)]
// CHECK:STDOUT:   %Self.loc23_32.2: @NotGeneric.loc23.%NotGeneric.type (%NotGeneric.type.50a) = symbolic_binding Self, 1 [symbolic = %Self.loc23_32.2 (constants.%Self.242)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc23_32.1: @NotGeneric.loc23.%NotGeneric.type (%NotGeneric.type.50a) = symbolic_binding Self, 1 [symbolic = %Self.loc23_32.2 (constants.%Self.242)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc23_32.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic.loc25(%T.loc25_19.2: type) {
// CHECK:STDOUT:   %T.loc25_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc25_19.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Generic.loc33 {
// CHECK:STDOUT:   %Self: %Generic.type.29a = symbolic_binding Self, 0 [symbolic = constants.%Self.36a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams.loc35(%T.loc35_27.2: type) {
// CHECK:STDOUT:   %T.loc35_27.1: type = symbolic_binding T, 0 [symbolic = %T.loc35_27.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams.loc43(%T.loc43_27.2: %empty_tuple.type) {
// CHECK:STDOUT:   %T.loc43_27.1: %empty_tuple.type = symbolic_binding T, 0 [symbolic = %T.loc43_27.1 (constants.%T.d76)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DifferentParams.type: type = facet_type <@DifferentParams.loc43, @DifferentParams.loc43(%T.loc43_27.1)> [symbolic = %DifferentParams.type (constants.%DifferentParams.type.0ec)]
// CHECK:STDOUT:   %Self.loc43_35.2: @DifferentParams.loc43.%DifferentParams.type (%DifferentParams.type.0ec) = symbolic_binding Self, 1 [symbolic = %Self.loc43_35.2 (constants.%Self.c69)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc43_35.1: @DifferentParams.loc43.%DifferentParams.type (%DifferentParams.type.0ec) = symbolic_binding Self, 1 [symbolic = %Self.loc43_35.2 (constants.%Self.c69)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc43_35.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGeneric.loc23(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc23_22.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.loc25(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc25_19.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.loc35(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc35_27.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.loc43(constants.%T.d76) {
// CHECK:STDOUT:   %T.loc43_27.1 => constants.%T.d76
// CHECK:STDOUT: }
// CHECK:STDOUT:

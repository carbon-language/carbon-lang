// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_generic_redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_generic_redeclaration.carbon

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
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+7]]:27: error: type `()` of parameter 1 in redeclaration differs from previous parameter type `type` [RedeclParamDiffersType]
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
// CHECK:STDOUT:   %NotGeneric.type.10e: type = facet_type <@NotGeneric.1> [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %NotGeneric.type.93b: type = generic_interface_type @NotGeneric.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NotGeneric.generic: %NotGeneric.type.93b = struct_value () [concrete]
// CHECK:STDOUT:   %NotGeneric.type.8cb: type = facet_type <@NotGeneric.2, @NotGeneric.2(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.c08: %NotGeneric.type.8cb = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic.1 [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.c99: type = facet_type <@Generic.2> [concrete]
// CHECK:STDOUT:   %Self.5ae: %Generic.type.c99 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type.d40e5c.1: type = generic_interface_type @DifferentParams.1 [concrete]
// CHECK:STDOUT:   %DifferentParams.generic.d33670.1: %DifferentParams.type.d40e5c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.7a6: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e60: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type.d40e5c.2: type = generic_interface_type @DifferentParams.2 [concrete]
// CHECK:STDOUT:   %DifferentParams.generic.d33670.2: %DifferentParams.type.d40e5c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DifferentParams.type.12c: type = facet_type <@DifferentParams.2, @DifferentParams.2(%T.7a6)> [symbolic]
// CHECK:STDOUT:   %Self.8d7: %DifferentParams.type.12c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl.loc11
// CHECK:STDOUT:     .Generic = %Generic.decl.loc21
// CHECK:STDOUT:     .DifferentParams = %DifferentParams.decl.loc31
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGeneric.decl.loc11: type = interface_decl @NotGeneric.1 [concrete = constants.%NotGeneric.type.10e] {} {}
// CHECK:STDOUT:   %NotGeneric.decl.loc19: %NotGeneric.type.93b = interface_decl @NotGeneric.2 [concrete = constants.%NotGeneric.generic] {
// CHECK:STDOUT:     %T.patt.loc19_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_22.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc19_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_22.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc21: %Generic.type.c21 = interface_decl @Generic.1 [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc21_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc21_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_19.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc29: type = interface_decl @Generic.2 [concrete = constants.%Generic.type.c99] {} {}
// CHECK:STDOUT:   %DifferentParams.decl.loc31: %DifferentParams.type.d40e5c.1 = interface_decl @DifferentParams.1 [concrete = constants.%DifferentParams.generic.d33670.1] {
// CHECK:STDOUT:     %T.patt.loc31_27.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_27.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc31_27.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc31_27.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentParams.decl.loc39: %DifferentParams.type.d40e5c.2 = interface_decl @DifferentParams.2 [concrete = constants.%DifferentParams.generic.d33670.2] {
// CHECK:STDOUT:     %T.patt.loc39_27.1: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc39_27.2 (constants.%T.patt.e60)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc39_32.1: type = splice_block %.loc39_32.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc39_32.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc39_32.3: type = converted %.loc39_32.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc39_27.1: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc39_27.2 (constants.%T.7a6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGeneric.1;
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @NotGeneric.2(%T.loc19_22.1: type) {
// CHECK:STDOUT:   %T.loc19_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_22.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc19_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_22.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NotGeneric.type: type = facet_type <@NotGeneric.2, @NotGeneric.2(%T.loc19_22.2)> [symbolic = %NotGeneric.type (constants.%NotGeneric.type.8cb)]
// CHECK:STDOUT:   %Self.2: %NotGeneric.type.8cb = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c08)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @NotGeneric.2.%NotGeneric.type (%NotGeneric.type.8cb) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c08)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic.1(%T.loc21_19.1: type) {
// CHECK:STDOUT:   %T.loc21_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_19.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc21_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Generic.2 {
// CHECK:STDOUT:   %Self: %Generic.type.c99 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.5ae]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams.1(%T.loc31_27.1: type) {
// CHECK:STDOUT:   %T.loc31_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc31_27.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc31_27.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_27.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams.2(%T.loc39_27.1: %empty_tuple.type) {
// CHECK:STDOUT:   %T.loc39_27.2: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc39_27.2 (constants.%T.7a6)]
// CHECK:STDOUT:   %T.patt.loc39_27.2: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc39_27.2 (constants.%T.patt.e60)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DifferentParams.type: type = facet_type <@DifferentParams.2, @DifferentParams.2(%T.loc39_27.2)> [symbolic = %DifferentParams.type (constants.%DifferentParams.type.12c)]
// CHECK:STDOUT:   %Self.2: %DifferentParams.type.12c = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @DifferentParams.2.%DifferentParams.type (%DifferentParams.type.12c) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGeneric.2(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc19_22.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc19_22.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGeneric.2(%T.loc19_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.1(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc21_19.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc21_19.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.1(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc31_27.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc31_27.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.2(constants.%T.7a6) {
// CHECK:STDOUT:   %T.loc39_27.2 => constants.%T.7a6
// CHECK:STDOUT:   %T.patt.loc39_27.2 => constants.%T.7a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.2(%T.loc39_27.2) {}
// CHECK:STDOUT:

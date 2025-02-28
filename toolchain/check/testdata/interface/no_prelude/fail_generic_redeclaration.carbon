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
// CHECK:STDERR:                           ^~~~~~
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:27: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: interface DifferentParams(T:! type);
// CHECK:STDERR:                           ^~~~~~~~
// CHECK:STDERR:
interface DifferentParams(T:! ()) {}

// CHECK:STDOUT: --- fail_generic_redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGeneric.type: type = facet_type <@NotGeneric> [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.type.abf52e.1: type = generic_interface_type @.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.generic.0a9e18.1: %.type.abf52e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.type.eb3: type = facet_type <@.1, @.1(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.037: %.type.eb3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type.4ea: type = facet_type <@.2> [concrete]
// CHECK:STDOUT:   %Self.86e: %.type.4ea = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type: type = generic_interface_type @DifferentParams [concrete]
// CHECK:STDOUT:   %DifferentParams.generic: %DifferentParams.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.7a6: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e60: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.type.abf52e.2: type = generic_interface_type @.3 [concrete]
// CHECK:STDOUT:   %.generic.0a9e18.2: %.type.abf52e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.type.b4a: type = facet_type <@.3, @.3(%T.7a6)> [symbolic]
// CHECK:STDOUT:   %Self.bb6: %.type.b4a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .DifferentParams = %DifferentParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGeneric.decl: type = interface_decl @NotGeneric [concrete = constants.%NotGeneric.type] {} {}
// CHECK:STDOUT:   %.decl.loc19: %.type.abf52e.1 = interface_decl @.1 [concrete = constants.%.generic.0a9e18.1] {
// CHECK:STDOUT:     %T.patt.loc19_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_22.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc19_22.1, runtime_param<none> [symbolic = %T.patt.loc19_22.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc19_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc19_22.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: %Generic.type = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc21_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc21_19.1, runtime_param<none> [symbolic = %T.patt.loc21_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc21_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc21_19.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc29: type = interface_decl @.2 [concrete = constants.%.type.4ea] {} {}
// CHECK:STDOUT:   %DifferentParams.decl: %DifferentParams.type = interface_decl @DifferentParams [concrete = constants.%DifferentParams.generic] {
// CHECK:STDOUT:     %T.patt.loc31_27.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_27.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc31_27.1, runtime_param<none> [symbolic = %T.patt.loc31_27.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc31_27.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc31_27.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc39: %.type.abf52e.2 = interface_decl @.3 [concrete = constants.%.generic.0a9e18.2] {
// CHECK:STDOUT:     %T.patt.loc39_27.1: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc39_27.2 (constants.%T.patt.e60)]
// CHECK:STDOUT:     %T.param_patt: %empty_tuple.type = value_param_pattern %T.patt.loc39_27.1, runtime_param<none> [symbolic = %T.patt.loc39_27.2 (constants.%T.patt.e60)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %empty_tuple.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc39_32.1: type = splice_block %.loc39_32.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc39_32.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc39_32.3: type = converted %.loc39_32.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc39_27.1: %empty_tuple.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc39_27.2 (constants.%T.7a6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGeneric;
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%T.loc19_22.1: type) {
// CHECK:STDOUT:   %T.loc19_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_22.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc19_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_22.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%T.loc19_22.2)> [symbolic = %.type (constants.%.type.eb3)]
// CHECK:STDOUT:   %Self.2: %.type.eb3 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.037)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.eb3) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.037)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%T.loc21_19.1: type) {
// CHECK:STDOUT:   %T.loc21_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_19.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc21_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_19.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.2 {
// CHECK:STDOUT:   %Self: %.type.4ea = bind_symbolic_name Self, 0 [symbolic = constants.%Self.86e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams(%T.loc31_27.1: type) {
// CHECK:STDOUT:   %T.loc31_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc31_27.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc31_27.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_27.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.3(%T.loc39_27.1: %empty_tuple.type) {
// CHECK:STDOUT:   %T.loc39_27.2: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc39_27.2 (constants.%T.7a6)]
// CHECK:STDOUT:   %T.patt.loc39_27.2: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc39_27.2 (constants.%T.patt.e60)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.3, @.3(%T.loc39_27.2)> [symbolic = %.type (constants.%.type.b4a)]
// CHECK:STDOUT:   %Self.2: %.type.b4a = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.bb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.3.%.type (%.type.b4a) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.bb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc19_22.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc19_22.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%T.loc19_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc21_19.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc21_19.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc31_27.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc31_27.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.3(constants.%T.7a6) {
// CHECK:STDOUT:   %T.loc39_27.2 => constants.%T.7a6
// CHECK:STDOUT:   %T.patt.loc39_27.2 => constants.%T.7a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.3(%T.loc39_27.2) {}
// CHECK:STDOUT:

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
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter list
// CHECK:STDERR: interface NotGeneric(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:1: previously declared without parameter list
// CHECK:STDERR: interface NotGeneric;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface NotGeneric(T:! type) {}

interface Generic(T:! type);
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+7]]:1: error: redeclaration differs because of missing parameter list
// CHECK:STDERR: interface Generic {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:1: previously declared with parameter list
// CHECK:STDERR: interface Generic(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Generic {}

interface DifferentParams(T:! type);
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE+6]]:27: error: redeclaration differs at parameter 1
// CHECK:STDERR: interface DifferentParams(T:! ()) {}
// CHECK:STDERR:                           ^
// CHECK:STDERR: fail_generic_redeclaration.carbon:[[@LINE-4]]:27: previous declaration's corresponding parameter here
// CHECK:STDERR: interface DifferentParams(T:! type);
// CHECK:STDERR:                           ^
interface DifferentParams(T:! ()) {}

// CHECK:STDOUT: --- fail_generic_redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @NotGeneric [template]
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%T.1) [symbolic]
// CHECK:STDOUT:   %Self.1: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_interface_type @Generic [template]
// CHECK:STDOUT:   %Generic: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = interface_type @.2 [template]
// CHECK:STDOUT:   %Self.2: %.5 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type: type = generic_interface_type @DifferentParams [template]
// CHECK:STDOUT:   %DifferentParams: %DifferentParams.type = struct_value () [template]
// CHECK:STDOUT:   %T.2: %.2 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.type.2: type = generic_interface_type @.3 [template]
// CHECK:STDOUT:   %.6: %.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = interface_type @.3, @.3(%T.2) [symbolic]
// CHECK:STDOUT:   %Self.3: %.7 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .DifferentParams = %DifferentParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGeneric.decl: type = interface_decl @NotGeneric [template = constants.%.1] {}
// CHECK:STDOUT:   %.decl.loc19: %.type.1 = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %T.loc19_22.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19_22.2: type = bind_symbolic_name T 0, %T.loc19_22.1 [symbolic = @.1.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: %Generic.type = interface_decl @Generic [template = constants.%Generic] {
// CHECK:STDOUT:     %T.loc21_19.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc21_19.2: type = bind_symbolic_name T 0, %T.loc21_19.1 [symbolic = @Generic.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc29: type = interface_decl @.2 [template = constants.%.5] {}
// CHECK:STDOUT:   %DifferentParams.decl: %DifferentParams.type = interface_decl @DifferentParams [template = constants.%DifferentParams] {
// CHECK:STDOUT:     %T.loc31_27.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc31_27.2: type = bind_symbolic_name T 0, %T.loc31_27.1 [symbolic = @DifferentParams.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc38: %.type.2 = interface_decl @.3 [template = constants.%.6] {
// CHECK:STDOUT:     %.loc38_32.1: %.2 = tuple_literal ()
// CHECK:STDOUT:     %.loc38_32.2: type = converted %.loc38_32.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %T.loc38_27.1: %.2 = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc38_27.2: %.2 = bind_symbolic_name T 0, %T.loc38_27.1 [symbolic = @.3.%T (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGeneric;
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%T.loc19_22.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(file.%T.loc21_19.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.2 {
// CHECK:STDOUT:   %Self: %.5 = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams(file.%T.loc31_27.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.3(file.%T.loc38_27.2: %.2) {
// CHECK:STDOUT:   %T: %.2 = bind_symbolic_name T 0 [symbolic = %T (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.3, @.3(%T) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Self.2: %.7 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.3.%.1 (%.7) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%T) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.3(constants.%T.2) {
// CHECK:STDOUT:   %T => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.3(@.3.%T) {
// CHECK:STDOUT:   %T => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

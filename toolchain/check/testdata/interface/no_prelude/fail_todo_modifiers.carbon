// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_modifiers.carbon

interface Modifiers {
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   final fn Final() { }
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final fn Final() { }
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn Default() { }
  // CHECK:STDERR:   ^~~~~~~
  default fn Default() { }
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Modifiers [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Final.type: type = fn_type @Final, @Modifiers[%Self] [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Final: %Final.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %Final.type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Modifiers.%Final.decl [symbolic]
// CHECK:STDOUT:   %Default.type: type = fn_type @Default, @Modifiers[%Self] [symbolic]
// CHECK:STDOUT:   %Default: %Default.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.1, %Default.type [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element1, @Modifiers.%Default.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Modifiers = %Modifiers.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.decl: type = interface_decl @Modifiers [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Modifiers[%Self.1: %.1] {
// CHECK:STDOUT:   %Self.2: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Final.type: type = fn_type @Final, @Modifiers[%Self.2] [symbolic = %Final.type (constants.%Final.type)]
// CHECK:STDOUT:   %Final: @Modifiers.%Final.type (%Final.type) = struct_value () [symbolic = %Final (constants.%Final)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.1, @Modifiers.%Final.type (%Final.type) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @Modifiers.%.1 (%.3) = assoc_entity element0, %Final.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %Default.type: type = fn_type @Default, @Modifiers[%Self.2] [symbolic = %Default.type (constants.%Default.type)]
// CHECK:STDOUT:   %Default: @Modifiers.%Default.type (%Default.type) = struct_value () [symbolic = %Default (constants.%Default)]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, @Modifiers.%Default.type (%Default.type) [symbolic = %.3 (constants.%.5)]
// CHECK:STDOUT:   %.4: @Modifiers.%.3 (%.5) = assoc_entity element1, %Default.decl [symbolic = %.4 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Final.decl: @Modifiers.%Final.type (%Final.type) = fn_decl @Final [symbolic = %Final (constants.%Final)] {}
// CHECK:STDOUT:   %.loc16: @Modifiers.%.1 (%.3) = assoc_entity element0, %Final.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %Default.decl: @Modifiers.%Default.type (%Default.type) = fn_decl @Default [symbolic = %Default (constants.%Default)] {}
// CHECK:STDOUT:   %.loc20: @Modifiers.%.3 (%.5) = assoc_entity element1, %Default.decl [symbolic = %.4 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .Final = %.loc16
// CHECK:STDOUT:   .Default = %.loc20
// CHECK:STDOUT:   witness = (%Final.decl, %Default.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Final(@Modifiers.%Self.1: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Default(@Modifiers.%Self.1: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Modifiers[constants.%Self] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Modifiers[@Modifiers.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier`
  // CHECK:STDERR:   final fn Final() { }
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final fn Final() { }
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: error: semantics TODO: `interface modifier`
  // CHECK:STDERR:   default fn Default() { }
  // CHECK:STDERR:   ^~~~~~~
  default fn Default() { }
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Modifiers.type: type = interface_type @Modifiers [template]
// CHECK:STDOUT:   %Self: %Modifiers.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Final.type: type = fn_type @Final [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Final: %Final.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Modifiers.type, %Final.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @Modifiers.%Final.decl [template]
// CHECK:STDOUT:   %Default.type: type = fn_type @Default [template]
// CHECK:STDOUT:   %Default: %Default.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Modifiers.type, %Default.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, @Modifiers.%Default.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Modifiers = %Modifiers.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.decl: type = interface_decl @Modifiers [template = constants.%Modifiers.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Modifiers {
// CHECK:STDOUT:   %Self: %Modifiers.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Final.decl: %Final.type = fn_decl @Final [template = constants.%Final] {} {}
// CHECK:STDOUT:   %.loc16: %.2 = assoc_entity element0, %Final.decl [template = constants.%.3]
// CHECK:STDOUT:   %Default.decl: %Default.type = fn_decl @Default [template = constants.%Default] {} {}
// CHECK:STDOUT:   %.loc20: %.4 = assoc_entity element1, %Default.decl [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Final = %.loc16
// CHECK:STDOUT:   .Default = %.loc20
// CHECK:STDOUT:   witness = (%Final.decl, %Default.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Final(@Modifiers.%Self: %Modifiers.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Default(@Modifiers.%Self: %Modifiers.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default(constants.%Self) {}
// CHECK:STDOUT:

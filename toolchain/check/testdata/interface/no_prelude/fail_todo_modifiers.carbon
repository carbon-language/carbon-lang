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
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   final fn Final() { }
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final fn Final() { }
  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn Default() { }
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn Default() { }
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Modifiers.type: type = facet_type <@Modifiers> [concrete]
// CHECK:STDOUT:   %Self: %Modifiers.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Final.type: type = fn_type @Final [concrete]
// CHECK:STDOUT:   %Final: %Final.type = struct_value () [concrete]
// CHECK:STDOUT:   %Modifiers.assoc_type: type = assoc_entity_type @Modifiers [concrete]
// CHECK:STDOUT:   %assoc0: %Modifiers.assoc_type = assoc_entity element0, @Modifiers.%Final.decl [concrete]
// CHECK:STDOUT:   %Default.type: type = fn_type @Default [concrete]
// CHECK:STDOUT:   %Default: %Default.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Modifiers.assoc_type = assoc_entity element1, @Modifiers.%Default.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Modifiers = %Modifiers.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.decl: type = interface_decl @Modifiers [concrete = constants.%Modifiers.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Modifiers {
// CHECK:STDOUT:   %Self: %Modifiers.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Final.decl: %Final.type = fn_decl @Final [concrete = constants.%Final] {} {}
// CHECK:STDOUT:   %assoc0: %Modifiers.assoc_type = assoc_entity element0, %Final.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Default.decl: %Default.type = fn_decl @Default [concrete = constants.%Default] {} {}
// CHECK:STDOUT:   %assoc1: %Modifiers.assoc_type = assoc_entity element1, %Default.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Final = %assoc0
// CHECK:STDOUT:   .Default = %assoc1
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

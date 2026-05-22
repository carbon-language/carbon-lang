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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_modifiers.carbon

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
// CHECK:STDOUT:   %Self: %Modifiers.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Modifiers.WithSelf.Final.type: type = fn_type @Modifiers.WithSelf.Final, @Modifiers.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Modifiers.WithSelf.Final: %Modifiers.WithSelf.Final.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Modifiers.assoc_type: type = assoc_entity_type @Modifiers [concrete]
// CHECK:STDOUT:   %assoc0: %Modifiers.assoc_type = assoc_entity element0, @Modifiers.WithSelf.%Modifiers.WithSelf.Final.decl [concrete]
// CHECK:STDOUT:   %Modifiers.WithSelf.Default.type: type = fn_type @Modifiers.WithSelf.Default, @Modifiers.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Modifiers.WithSelf.Default: %Modifiers.WithSelf.Default.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %Modifiers.assoc_type = assoc_entity element1, @Modifiers.WithSelf.%Modifiers.WithSelf.Default.decl [concrete]
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
// CHECK:STDOUT:   %Self: %Modifiers.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Modifiers.WithSelf.decl = interface_with_self_decl @Modifiers [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Modifiers.WithSelf.Final.decl: @Modifiers.WithSelf.%Modifiers.WithSelf.Final.type (%Modifiers.WithSelf.Final.type) = fn_decl @Modifiers.WithSelf.Final [symbolic = @Modifiers.WithSelf.%Modifiers.WithSelf.Final (constants.%Modifiers.WithSelf.Final)] {} {}
// CHECK:STDOUT:   %assoc0: %Modifiers.assoc_type = assoc_entity element0, %Modifiers.WithSelf.Final.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Modifiers.WithSelf.Default.decl: @Modifiers.WithSelf.%Modifiers.WithSelf.Default.type (%Modifiers.WithSelf.Default.type) = fn_decl @Modifiers.WithSelf.Default [symbolic = @Modifiers.WithSelf.%Modifiers.WithSelf.Default (constants.%Modifiers.WithSelf.Default)] {} {}
// CHECK:STDOUT:   %assoc1: %Modifiers.assoc_type = assoc_entity element1, %Modifiers.WithSelf.Default.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Final = @Modifiers.WithSelf.%assoc0
// CHECK:STDOUT:   .Default = @Modifiers.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@Modifiers.WithSelf.%Modifiers.WithSelf.Final.decl, @Modifiers.WithSelf.%Modifiers.WithSelf.Default.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Modifiers.WithSelf.Final(@Modifiers.%Self: %Modifiers.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Modifiers.WithSelf.Default(@Modifiers.%Self: %Modifiers.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Modifiers.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Modifiers.WithSelf.Final.type => constants.%Modifiers.WithSelf.Final.type
// CHECK:STDOUT:   %Modifiers.WithSelf.Final => constants.%Modifiers.WithSelf.Final
// CHECK:STDOUT:   %Modifiers.WithSelf.Default.type => constants.%Modifiers.WithSelf.Default.type
// CHECK:STDOUT:   %Modifiers.WithSelf.Default => constants.%Modifiers.WithSelf.Default
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Modifiers.WithSelf.Final(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Modifiers.WithSelf.Default(constants.%Self) {}
// CHECK:STDOUT:

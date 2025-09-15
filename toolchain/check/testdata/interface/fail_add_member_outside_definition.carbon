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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_add_member_outside_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_add_member_outside_definition.carbon

interface Interface { }

// CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:14: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR:              ^
// CHECK:STDERR:
fn Interface.F() {}

// Nesting interfaces like this is not valid, but make sure we don't crash.
interface Outer {
  interface Inner {
    // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:8: error: name `Outer` not found [NameNotFound]
    // CHECK:STDERR:     fn Outer.F();
    // CHECK:STDERR:        ^~~~~
    // CHECK:STDERR:
    fn Outer.F();
  }
  // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:12: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
  // CHECK:STDERR:   fn Inner.F();
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  fn Inner.F();
}

// CHECK:STDOUT: --- fail_add_member_outside_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.4f1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.F.type: type = fn_type @Interface.F [concrete]
// CHECK:STDOUT:   %Interface.F: %Interface.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.type: type = facet_type <@Outer> [concrete]
// CHECK:STDOUT:   %Self.478: %Outer.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%Self.478)> [symbolic]
// CHECK:STDOUT:   %Self.8f6: %Inner.type = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.F.type.2c750a.1: type = fn_type @Inner.F.loc30, @Inner(%Self.478) [symbolic]
// CHECK:STDOUT:   %Inner.F.91612b.1: %Inner.F.type.2c750a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.F.type.2c750a.2: type = fn_type @Inner.F.loc36, @Inner(%Self.478) [symbolic]
// CHECK:STDOUT:   %Inner.F.91612b.2: %Inner.F.type.2c750a.2 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type = fn_decl @Interface.F [concrete = constants.%Interface.F] {} {}
// CHECK:STDOUT:   %Outer.decl: type = interface_decl @Outer [concrete = constants.%Outer.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4f1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = file.%Interface.F.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Outer {
// CHECK:STDOUT:   %Self: %Outer.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.478]
// CHECK:STDOUT:   %Inner.decl: type = interface_decl @Inner [symbolic = constants.%Inner.type] {} {}
// CHECK:STDOUT:   %Inner.F.decl: %Inner.F.type.2c750a.2 = fn_decl @Inner.F.loc36 [symbolic = constants.%Inner.F.91612b.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%Self: %Outer.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %Outer.type = bind_symbolic_name Self, 0 [symbolic = %Self.2 (constants.%Self.478)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%Self.2)> [symbolic = %Inner.type (constants.%Inner.type)]
// CHECK:STDOUT:   %Self.3: @Inner.%Inner.type (%Inner.type) = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.8f6)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F.loc30, @Inner(%Self.2) [symbolic = %Inner.F.type (constants.%Inner.F.type.2c750a.1)]
// CHECK:STDOUT:   %Inner.F: @Inner.%Inner.F.type (%Inner.F.type.2c750a.1) = struct_value () [symbolic = %Inner.F (constants.%Inner.F.91612b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Inner.%Inner.type (%Inner.type) = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.8f6)]
// CHECK:STDOUT:     %Inner.F.decl: @Inner.%Inner.F.type (%Inner.F.type.2c750a.1) = fn_decl @Inner.F.loc30 [symbolic = @Inner.%Inner.F (constants.%Inner.F.91612b.1)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = @Outer.%Inner.F.decl
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.F.loc30(@Outer.%Self: %Outer.type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.F.loc36(@Outer.%Self: %Outer.type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F(constants.%Self.4f1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Self.478) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2 => constants.%Self.478
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type
// CHECK:STDOUT:   %Self.3 => constants.%Self.8f6
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.2c750a.1
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F.91612b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F.loc30(constants.%Self.478, constants.%Self.8f6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F.loc36(constants.%Self.478, constants.%Self.8f6) {}
// CHECK:STDOUT:

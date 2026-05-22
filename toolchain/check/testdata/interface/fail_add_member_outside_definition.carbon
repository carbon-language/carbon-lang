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
// CHECK:STDOUT:   %Self.7a0: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F.type: type = fn_type @Interface.WithSelf.F, @Interface.WithSelf(%Self.7a0) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F: %Interface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Outer.type: type = facet_type <@Outer> [concrete]
// CHECK:STDOUT:   %Self.807: %Outer.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%Self.807)> [symbolic]
// CHECK:STDOUT:   %Self.f28: %Inner.type = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.b8907a.1: type = fn_type @Inner.WithSelf.F.loc30, @Inner.WithSelf(%Self.807, %Self.f28) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.6fb3c6.1: %Inner.WithSelf.F.type.b8907a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.b8907a.2: type = fn_type @Inner.WithSelf.F.loc36, @Inner.WithSelf(%Self.807, %Self.f28) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.6fb3c6.2: %Inner.WithSelf.F.type.b8907a.2 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %Interface.WithSelf.F.decl: %Interface.WithSelf.F.type = fn_decl @Interface.WithSelf.F [symbolic = constants.%Interface.WithSelf.F] {} {}
// CHECK:STDOUT:   %Outer.decl: type = interface_decl @Outer [concrete = constants.%Outer.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self.7a0]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = file.%Interface.WithSelf.F.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Outer {
// CHECK:STDOUT:   %Self: %Outer.type = symbolic_binding Self, 0 [symbolic = constants.%Self.807]
// CHECK:STDOUT:   %Outer.WithSelf.decl = interface_with_self_decl @Outer [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.WithSelf.%Inner.type (constants.%Inner.type)] {} {}
// CHECK:STDOUT:   %Inner.WithSelf.F.decl: @Outer.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.b8907a.2) = fn_decl @Inner.WithSelf.F.loc36 [symbolic = @Outer.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.6fb3c6.2)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Inner = @Outer.WithSelf.%Inner.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%Self: %Outer.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc25_19.2: %Outer.type = symbolic_binding Self, 0 [symbolic = %Self.loc25_19.2 (constants.%Self.807)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%Self.loc25_19.2)> [symbolic = %Inner.type (constants.%Inner.type)]
// CHECK:STDOUT:   %Self.loc25_19.3: @Inner.%Inner.type (%Inner.type) = symbolic_binding Self, 1 [symbolic = %Self.loc25_19.3 (constants.%Self.f28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc25_19.1: @Inner.%Inner.type (%Inner.type) = symbolic_binding Self, 1 [symbolic = %Self.loc25_19.3 (constants.%Self.f28)]
// CHECK:STDOUT:     %Inner.WithSelf.decl = interface_with_self_decl @Inner [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Inner.WithSelf.F.decl: @Inner.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.b8907a.1) = fn_decl @Inner.WithSelf.F.loc30 [symbolic = @Inner.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.6fb3c6.1)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc25_19.1
// CHECK:STDOUT:     .F = @Outer.WithSelf.%Inner.WithSelf.F.decl
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.WithSelf.F.loc30(@Outer.%Self: %Outer.type, @Inner.%Self.loc25_19.1: @Inner.%Inner.type (%Inner.type)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.WithSelf.F.loc36(@Outer.%Self: %Outer.type, @Inner.%Self.loc25_19.1: @Inner.%Inner.type (%Inner.type)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Self.7a0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.F(constants.%Self.7a0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.WithSelf(constants.%Self.807) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Self.807) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc25_19.2 => constants.%Self.807
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type
// CHECK:STDOUT:   %Self.loc25_19.3 => constants.%Self.f28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%Self.807, constants.%Self.f28) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F.loc30(constants.%Self.807, constants.%Self.f28) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F.loc36(constants.%Self.807, constants.%Self.f28) {}
// CHECK:STDOUT:

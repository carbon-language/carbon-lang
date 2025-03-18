// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_add_member_outside_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_add_member_outside_definition.carbon

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
// CHECK:STDOUT:   %Self.719: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1ad: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.5d3: %F.type.1ad = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.type: type = facet_type <@Outer> [concrete]
// CHECK:STDOUT:   %Self.277: %Outer.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Inner.type.428: type = facet_type <@Inner> [concrete]
// CHECK:STDOUT:   %Inner.type.3d8: type = facet_type <@Inner, @Inner(%Self.277)> [symbolic]
// CHECK:STDOUT:   %Self.b60: %Inner.type.3d8 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.82c77c.1: type = fn_type @F.2, @Inner(%Self.277) [symbolic]
// CHECK:STDOUT:   %F.9789e2.1: %F.type.82c77c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type.82c77c.2: type = fn_type @F.3, @Inner(%Self.277) [symbolic]
// CHECK:STDOUT:   %F.9789e2.2: %F.type.82c77c.2 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.1ad = fn_decl @F.1 [concrete = constants.%F.5d3] {} {}
// CHECK:STDOUT:   %Outer.decl: type = interface_decl @Outer [concrete = constants.%Outer.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.719]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = file.%F.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Outer {
// CHECK:STDOUT:   %Self: %Outer.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.277]
// CHECK:STDOUT:   %Inner.decl: type = interface_decl @Inner [concrete = constants.%Inner.type.428] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.82c77c.2 = fn_decl @F.3 [symbolic = constants.%F.9789e2.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%Self: %Outer.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %Outer.type = bind_symbolic_name Self, 0 [symbolic = %Self.2 (constants.%Self.277)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%Self.2)> [symbolic = %Inner.type (constants.%Inner.type.3d8)]
// CHECK:STDOUT:   %Self.3: %Inner.type.3d8 = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.b60)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @Inner(%Self.2) [symbolic = %F.type (constants.%F.type.82c77c.1)]
// CHECK:STDOUT:   %F: @Inner.%F.type (%F.type.82c77c.1) = struct_value () [symbolic = %F (constants.%F.9789e2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Inner.%Inner.type (%Inner.type.3d8) = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.b60)]
// CHECK:STDOUT:     %F.decl: @Inner.%F.type (%F.type.82c77c.1) = fn_decl @F.2 [symbolic = @Inner.%F (constants.%F.9789e2.1)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = @Outer.%F.decl
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@Outer.%Self: %Outer.type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.3d8)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.3(@Outer.%Self: %Outer.type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.3d8)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.719) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Self.277) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2 => constants.%Self.277
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.3d8
// CHECK:STDOUT:   %Self.3 => constants.%Self.b60
// CHECK:STDOUT:   %F.type => constants.%F.type.82c77c.1
// CHECK:STDOUT:   %F => constants.%F.9789e2.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.277, constants.%Self.b60) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.3(constants.%Self.277, constants.%Self.b60) {}
// CHECK:STDOUT:

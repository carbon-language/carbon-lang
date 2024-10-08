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

// CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:14: error: out-of-line declaration requires a declaration in scoped entity
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR:              ^
// CHECK:STDERR:
fn Interface.F() {}

// Nesting interfaces like this is not valid, but make sure we don't crash.
interface Outer {
  interface Inner {
    // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:8: error: name `Outer` not found
    // CHECK:STDERR:     fn Outer.F();
    // CHECK:STDERR:        ^~~~~
    // CHECK:STDERR:
    fn Outer.F();
  }
  // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+3]]:12: error: out-of-line declaration requires a declaration in scoped entity
  // CHECK:STDERR:   fn Inner.F();
  // CHECK:STDERR:            ^
  fn Inner.F();
}

// CHECK:STDOUT: --- fail_add_member_outside_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self.1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Outer.type: type = interface_type @Outer [template]
// CHECK:STDOUT:   %Self.2: %Outer.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Inner.type.1: type = interface_type @Inner [template]
// CHECK:STDOUT:   %Inner.type.2: type = interface_type @Inner, @Inner(%Self.2) [symbolic]
// CHECK:STDOUT:   %Self.3: %Inner.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @Inner(%Self.2) [symbolic]
// CHECK:STDOUT:   %.2: %.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @Inner(%Self.2) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %Outer.decl: type = interface_decl @Outer [template = constants.%Outer.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = file.%F.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Outer {
// CHECK:STDOUT:   %Self: %Outer.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %Inner.decl: type = interface_decl @Inner [template = constants.%Inner.type.1] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [symbolic = constants.%F.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%Self: %Outer.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %Outer.type = bind_symbolic_name Self, 0 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %Inner.type: type = interface_type @Inner, @Inner(%Self.2) [symbolic = %Inner.type (constants.%Inner.type.2)]
// CHECK:STDOUT:   %Self.3: %Inner.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.3)]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @Inner(%Self.2) [symbolic = %.type (constants.%.type)]
// CHECK:STDOUT:   %.1: @Inner.%.type (%.type) = struct_value () [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Inner.%Inner.type (%Inner.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.3)]
// CHECK:STDOUT:     %.decl: @Inner.%.type (%.type) = fn_decl @.1 [symbolic = @Inner.%.1 (constants.%.2)] {} {}
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
// CHECK:STDOUT: generic fn @.1(@Outer.%Self: %Outer.type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@Outer.%Self: %Outer.type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Self.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Self.3 => constants.%Self.3
// CHECK:STDOUT:   %.type => constants.%.type
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%Self.2, constants.%Self.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@Inner.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.2, constants.%Self.3) {}
// CHECK:STDOUT:

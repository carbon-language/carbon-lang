// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_let_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_let_in_type.carbon

let x: type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn Six() -> x { return 6; }

// TODO: This should probably work.
let y:! type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+7]]:23: error: cannot implicitly convert from `i32` to `y`
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                       ^~~~~~~~~
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:23: note: type `i32` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                       ^~~~~~~~~
// CHECK:STDERR:
fn HalfDozen() -> y { return 6; }

// TODO: This should work.
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:5: error: semantics TODO: `HandleTemplate`
// CHECK:STDERR: let template z:! type = i32;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
let template z:! type = i32;
fn FirstPerfectNumber() -> z { return 6; }

// CHECK:STDOUT: --- fail_let_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Six.type: type = fn_type @Six [template]
// CHECK:STDOUT:   %Six: %Six.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %y: type = bind_symbolic_name y, 0 [symbolic]
// CHECK:STDOUT:   %HalfDozen.type: type = fn_type @HalfDozen [template]
// CHECK:STDOUT:   %HalfDozen: %HalfDozen.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, <unexpected>.inst+54 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%y) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%y) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, <unexpected>.inst+54 [symbolic]
// CHECK:STDOUT:   %.7: %.3 = assoc_entity element0, <unexpected>.inst+71 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.3) = assoc_entity element0, <unexpected>.inst+54 [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = <unexpected>.inst+42
// CHECK:STDOUT:     .Convert = <unexpected>.inst+43
// CHECK:STDOUT:     witness = (<unexpected>.inst+44)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HalfDozen(<unexpected>.inst+21.loc19_5: type) {
// CHECK:STDOUT:   %y: type = bind_symbolic_name y, 0 [symbolic = %y (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type.1: type = interface_type @ImplicitAs, @ImplicitAs(%y) [symbolic = %ImplicitAs.type.1 (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%y) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @HalfDozen.%ImplicitAs.type.1 (%ImplicitAs.type.3), @HalfDozen.%Convert.type (%Convert.type.2) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @HalfDozen.%.1 (%.5) = assoc_entity element0, <unexpected>.inst+54 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @HalfDozen.%y (%y) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc27_30: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:     %ImplicitAs.type.loc27: type = interface_type @ImplicitAs, @ImplicitAs(constants.%y) [symbolic = %ImplicitAs.type.1 (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:     %.loc27_31.1: @HalfDozen.%.1 (%.5) = specific_constant <unexpected>.inst+43, @ImplicitAs(constants.%y) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:     %Convert.ref: @HalfDozen.%.1 (%.5) = name_ref Convert, %.loc27_31.1 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:     %.loc27_31.2: @HalfDozen.%y (%y) = converted %.loc27_30, <error> [template = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HalfDozen(constants.%y) {
// CHECK:STDOUT:   %y => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%y) {
// CHECK:STDOUT:   %Dest => constants.%y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@HalfDozen.%y) {
// CHECK:STDOUT:   %Dest => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:

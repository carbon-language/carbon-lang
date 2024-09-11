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
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:13: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn Six() -> x { return 6; }

// TODO: This should probably work.
let y:! type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+7]]:23: ERROR: Cannot implicitly convert from `i32` to `y`.
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                       ^~~~~~~~~
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:23: Type `i32` does not implement interface `ImplicitAs`.
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                       ^~~~~~~~~
// CHECK:STDERR:
fn HalfDozen() -> y { return 6; }

// TODO: This should work.
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:5: ERROR: Semantics TODO: `HandleTemplate`.
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
// CHECK:STDOUT:   %y: type = bind_symbolic_name y 0 [symbolic]
// CHECK:STDOUT:   %HalfDozen.type: type = fn_type @HalfDozen [template]
// CHECK:STDOUT:   %HalfDozen: %HalfDozen.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.3, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, <unexpected>.inst+54 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(%y) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%y) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.2 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, <unexpected>.inst+54 [symbolic]
// CHECK:STDOUT:   %.9: %.4 = assoc_entity element0, <unexpected>.inst+71 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.3), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.4) = assoc_entity element0, <unexpected>.inst+54 [symbolic = %.3 (constants.%.5)]
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
// CHECK:STDOUT:   %y: type = bind_symbolic_name y 0 [symbolic = %y (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%y) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%y) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @HalfDozen.%.1 (%.6), @HalfDozen.%Convert.type (%Convert.type.2) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @HalfDozen.%.2 (%.7) = assoc_entity element0, <unexpected>.inst+54 [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @HalfDozen.%y (%y) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc27_30: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:     %.loc27_31.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%y) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:     %.loc27_31.2: @HalfDozen.%.2 (%.7) = specific_constant <unexpected>.inst+43, @ImplicitAs(constants.%y) [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:     %Convert.ref: @HalfDozen.%.2 (%.7) = name_ref Convert, %.loc27_31.2 [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:     %.loc27_31.3: @HalfDozen.%y (%y) = converted %.loc27_30, <error> [template = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%.1 (%.3)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
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
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%y) {
// CHECK:STDOUT:   %Dest => constants.%y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@HalfDozen.%y) {
// CHECK:STDOUT:   %Dest => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:

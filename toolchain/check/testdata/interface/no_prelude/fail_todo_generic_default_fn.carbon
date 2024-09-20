// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_generic_default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_generic_default_fn.carbon

interface I(T:! type) {
  // TODO: Use `default` here.
  fn F[self: Self]() -> Self;
}

// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+6]]:1: error: duplicate name being declared in the same scope
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE-6]]:3: note: name is previously declared here
// CHECK:STDERR:   fn F[self: Self]() -> Self;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
fn I(T:! type).F[self: Self]() -> Self { return self; }

// CHECK:STDOUT: --- fail_todo_generic_default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, %F.type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %.5: %.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [symbolic = constants.%.5] {
// CHECK:STDOUT:     %T.loc22_6.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22_6.2: type = bind_symbolic_name T 0, %T.loc22_6.1 [symbolic = @.1.%T (constants.%T)]
// CHECK:STDOUT:     %.loc22_24.1: @.1.%.1 (%.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc22_24: @.1.%.1 (%.2) = name_ref Self, %.loc22_24.1 [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_24.2: type = facet_type_access %Self.ref.loc22_24 [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_24.3: type = converted %Self.ref.loc22_24, %.loc22_24.2 [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %self.loc22_18.1: @.1.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:     @.1.%self: @.1.%Self (%Self) = bind_name self, %self.loc22_18.1
// CHECK:STDOUT:     %.loc22_35.1: @.1.%.1 (%.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc22_35: @.1.%.1 (%.2) = name_ref Self, %.loc22_35.1 [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_35.2: type = facet_type_access %Self.ref.loc22_35 [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_35.3: type = converted %Self.ref.loc22_35, %.loc22_35.2 [symbolic = @.1.%Self (constants.%Self)]
// CHECK:STDOUT:     @.1.%return: ref @.1.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I.%.1 (%.2), @I.%F.type (%F.type) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: @I.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%.1 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {
// CHECK:STDOUT:       %.loc13_14.1: @F.%.1 (%.2) = specific_constant %Self.1, @I(constants.%T) [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_14: @F.%.1 (%.2) = name_ref Self, %.loc13_14.1 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_14.2: type = facet_type_access %Self.ref.loc13_14 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_14.3: type = converted %Self.ref.loc13_14, %.loc13_14.2 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %self.loc13_8.1: @F.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc13_8.2: @F.%Self (%Self) = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:       %.loc13_25.1: @F.%.1 (%.2) = specific_constant %Self.1, @I(constants.%T) [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_25: @F.%.1 (%.2) = name_ref Self, %.loc13_25.1 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_25.2: type = facet_type_access %Self.ref.loc13_25 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_25.3: type = converted %Self.ref.loc13_25, %.loc13_25.2 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:       %return.var: ref @F.%Self (%Self) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc13_29: @I.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc13_29
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(file.%T.loc11_13.2: type, @I.%Self.1: @I.%.1 (%.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@I.%self.loc13_8.2: @F.%Self (%Self)]() -> @F.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(file.%T.loc22_6.2: type, @I.%Self.1: @I.%.1 (%.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @.1.%Self (%Self)]() -> @.1.%Self (%Self) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @.1.%Self (%Self) = name_ref self, %self
// CHECK:STDOUT:     return %self.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT:   %.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:

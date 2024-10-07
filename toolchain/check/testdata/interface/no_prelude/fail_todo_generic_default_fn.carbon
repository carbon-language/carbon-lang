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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = interface_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type.2, %F.type [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %.4: %.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [symbolic = constants.%.4] {
// CHECK:STDOUT:     %self.patt: @.1.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc22_24.1: @.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc22_24: @.1.%I.type (%I.type.2) = name_ref Self, %.loc22_24.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_24.2: type = facet_type_access %Self.ref.loc22_24 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_24.3: type = converted %Self.ref.loc22_24, %.loc22_24.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @.1.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:     %self: @.1.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:     %.loc22_35.1: @.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc22_35: @.1.%I.type (%I.type.2) = name_ref Self, %.loc22_35.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_35.2: type = facet_type_access %Self.ref.loc22_35 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc22_35.3: type = converted %Self.ref.loc22_35, %.loc22_35.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %return: ref @.1.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%T.1) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @I.%.1 (%.2) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = @I.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_14.1: @F.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_14: @F.%I.type (%I.type.2) = name_ref Self, %.loc13_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_14.2: type = facet_type_access %Self.ref.loc13_14 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_14.3: type = converted %Self.ref.loc13_14, %.loc13_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %self.param: @F.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:       %self: @F.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:       %.loc13_25.1: @F.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_25: @F.%I.type (%I.type.2) = name_ref Self, %.loc13_25.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_25.2: type = facet_type_access %Self.ref.loc13_25 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc13_25.3: type = converted %Self.ref.loc13_25, %.loc13_25.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %return: ref @F.%Self (%Self) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc13: @I.%.1 (%.2) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc13
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%T.loc11: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%T) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @F.%Self (%Self)]() -> @F.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%T.loc22: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%T.1) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
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
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@.1.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:

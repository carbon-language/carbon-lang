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

// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+6]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE-6]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   fn F[self: Self]() -> Self;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
fn I(T:! type).F[self: Self]() -> Self { return self; }

// CHECK:STDOUT: --- fail_todo_generic_default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type.2, %F.type [symbolic]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %.1: %.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_13.1, runtime_param<invalid> [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [symbolic = constants.%.1] {
// CHECK:STDOUT:     %self.patt: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc22_24.1: @.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc22_24: @.1.%I.type (%I.type.2) = name_ref Self, %.loc22_24.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc22_24.1: type = facet_access_type %Self.ref.loc22_24 [symbolic = %Self.as_type.loc22_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc22_24.2: type = converted %Self.ref.loc22_24, %Self.as_type.loc22_24.1 [symbolic = %Self.as_type.loc22_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc22_35.1: @.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc22_35: @.1.%I.type (%I.type.2) = name_ref Self, %.loc22_35.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc22_35: type = facet_access_type %Self.ref.loc22_35 [symbolic = %Self.as_type.loc22_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc22_35.2: type = converted %Self.ref.loc22_35, %Self.as_type.loc22_35 [symbolic = %Self.as_type.loc22_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %self: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc11_13.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.loc11_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type) [symbolic = %F.assoc_type (constants.%F.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc13_29.2: @I.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc13_29.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = @I.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%Self.as_type.loc13_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.%Self.as_type.loc13_14.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.%Self.as_type.loc13_14.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.%Self.as_type.loc13_14.1 (%Self.as_type) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_14.1: @F.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_14: @F.%I.type (%I.type.2) = name_ref Self, %.loc13_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc13_14.2: type = facet_access_type %Self.ref.loc13_14 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc13_14.2: type = converted %Self.ref.loc13_14, %Self.as_type.loc13_14.2 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc13_25.1: @F.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_25: @F.%I.type (%I.type.2) = name_ref Self, %.loc13_25.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc13_25: type = facet_access_type %Self.ref.loc13_25 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc13_25.2: type = converted %Self.ref.loc13_25, %Self.as_type.loc13_25 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %self.param: @F.%Self.as_type.loc13_14.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.%Self.as_type.loc13_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.%Self.as_type.loc13_14.1 (%Self.as_type) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.%Self.as_type.loc13_14.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc13_29.1: @I.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc13_29.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc13_29.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%T.loc11_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc13_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc13_14.1 (%Self.as_type)]() -> @F.%Self.as_type.loc13_14.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%T.loc22_6.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %T.loc22_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc22_6.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc22_24.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc22_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @.1.%Self.as_type.loc22_24.2 (%Self.as_type)]() -> @.1.%Self.as_type.loc22_24.2 (%Self.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @.1.%Self.as_type.loc22_24.2 (%Self.as_type) = name_ref self, %self
// CHECK:STDOUT:     return %self.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type
// CHECK:STDOUT:   %assoc0.loc13_29.2 => constants.%assoc0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc13_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@.1.%T.loc22_6.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T.loc22_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc22_24.2 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

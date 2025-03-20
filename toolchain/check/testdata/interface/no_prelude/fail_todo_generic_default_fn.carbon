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

// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+7]]:16: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR:                ^
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE-6]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   fn F[self: Self]() -> Self;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn I(T:! type).F[self: Self]() -> Self { return self; }

// CHECK:STDOUT: --- fail_todo_generic_default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.1: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.1: %F.type.2aef59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.2: type = fn_type @F.2, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.2: %F.type.2aef59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.2aef59.2 = fn_decl @F.2 [symbolic = constants.%F.bb2dd4.2] {
// CHECK:STDOUT:     %self.patt: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc23_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc23_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc23_35.1: @F.2.%I.type (%I.type.325) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.ref.loc23_35: @F.2.%I.type (%I.type.325) = name_ref Self, %.loc23_35.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc23_35: type = facet_access_type %Self.ref.loc23_35 [symbolic = %Self.as_type.loc23_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc23_35.2: type = converted %Self.ref.loc23_35, %Self.as_type.loc23_35 [symbolic = %Self.as_type.loc23_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc23_24.1: type = splice_block %.loc23_24.3 [symbolic = %Self.as_type.loc23_24.2 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %.loc23_24.2: @F.2.%I.type (%I.type.325) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc23_24: @F.2.%I.type (%I.type.325) = name_ref Self, %.loc23_24.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc23_24.1: type = facet_access_type %Self.ref.loc23_24 [symbolic = %Self.as_type.loc23_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc23_24.3: type = converted %Self.ref.loc23_24, %Self.as_type.loc23_24.1 [symbolic = %Self.as_type.loc23_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc11_13.2)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self.2: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T.loc11_13.2) [symbolic = %F.type (constants.%F.type.2aef59.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2aef59.1) = struct_value () [symbolic = %F (constants.%F.bb2dd4.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc13_29.2: @I.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc13_29.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.2aef59.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.bb2dd4.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_25.1: @F.1.%I.type (%I.type.325) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref.loc13_25: @F.1.%I.type (%I.type.325) = name_ref Self, %.loc13_25.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc13_25: type = facet_access_type %Self.ref.loc13_25 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc13_25.2: type = converted %Self.ref.loc13_25, %Self.as_type.loc13_25 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc13_14.1: type = splice_block %.loc13_14.3 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc13_14.2: @F.1.%I.type (%I.type.325) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref.loc13_14: @F.1.%I.type (%I.type.325) = name_ref Self, %.loc13_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc13_14.2: type = facet_access_type %Self.ref.loc13_14 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc13_14.3: type = converted %Self.ref.loc13_14, %Self.as_type.loc13_14.2 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.1.%Self.as_type.loc13_14.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc13_29.1: @I.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc13_29.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc13_29.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%T.loc11_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc13_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc13_14.1 (%Self.as_type)]() -> @F.1.%Self.as_type.loc13_14.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%T.loc23_6.1: type, @I.%Self.1: @I.%I.type (%I.type.325)) {
// CHECK:STDOUT:   %T.loc23_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc23_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc23_6.2)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc23_24.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc23_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type)]() -> @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%Self.as_type.loc23_24.2 (%Self.as_type) = name_ref self, %self
// CHECK:STDOUT:     return %self.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2aef59.1
// CHECK:STDOUT:   %F => constants.%F.bb2dd4.1
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type
// CHECK:STDOUT:   %assoc0.loc13_29.2 => constants.%assoc0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc13_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc11_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T.loc23_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc23_24.2 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.2.%T.loc23_6.2) {}
// CHECK:STDOUT:

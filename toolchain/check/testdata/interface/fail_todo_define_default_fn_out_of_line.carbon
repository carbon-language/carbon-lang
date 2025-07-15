// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_out_of_line.carbon

// --- fail_todo_define_default_fn_out_of_line.carbon

interface Interface {
  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F();

  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn G(a: i32, b: i32) -> i32;
}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:14: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-12]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   default fn F();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.F() {}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:14: error: duplicate name `G` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-15]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";

// --- dependent_return_type.carbon

library "dependent_return_type.carbon";

// Ensure that a dependent return type matches between an interface definition
// and an out-of-line member. This requires `Self` to be properly reintroduced
// into the list of generic parameters.

interface Interface {
  // TODO: This should be
  //   default fn F[self: Self](U:! type, u: U) -> U;
  // rather than a class member, but we don't currently accept that due to the
  // TODOs above.
  class C {
    fn F[self: Self](U:! type, u: U) -> U;
  }
}

// TODO: This should be
//   fn Interface.F[self: Self](U:! type, u: U) -> U { return u; }
fn Interface.C.F[self: Self](U:! type, u: U) -> U { return u; }

// CHECK:STDOUT: --- fail_todo_define_default_fn_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1ad64d.1: type = fn_type @F.loc7 [concrete]
// CHECK:STDOUT:   %F.5d382e.1: %F.type.1ad64d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.%F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %G.type.ef383d.1: type = fn_type @G.loc13 [concrete]
// CHECK:STDOUT:   %G.362a5a.1: %G.type.ef383d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.%G.decl [concrete]
// CHECK:STDOUT:   %F.type.1ad64d.2: type = fn_type @F.loc23 [concrete]
// CHECK:STDOUT:   %F.5d382e.2: %F.type.1ad64d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.ef383d.2: type = fn_type @G.loc32 [concrete]
// CHECK:STDOUT:   %G.362a5a.2: %G.type.ef383d.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.1ad64d.2 = fn_decl @F.loc23 [concrete = constants.%F.5d382e.2] {} {}
// CHECK:STDOUT:   %G.decl: %G.type.ef383d.2 = fn_decl @G.loc32 [concrete = constants.%G.362a5a.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc32_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc32_19: type = splice_block %i32.loc32_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc32_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc32_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc32_27: type = splice_block %i32.loc32_27 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc32_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc32_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1ad64d.1 = fn_decl @F.loc7 [concrete = constants.%F.5d382e.1] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %G.decl: %G.type.ef383d.1 = fn_decl @G.loc13 [concrete = constants.%G.362a5a.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc13_19: type = splice_block %i32.loc13_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc13_27: type = splice_block %i32.loc13_27 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc7(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.loc13(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:   fn(%a.param: %i32, %b.param: %i32) -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc23(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.loc32(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: %i32, %b.param: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc7(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.loc13(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc23(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.loc32(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- dependent_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic]
// CHECK:STDOUT:   %pattern_type.68b: type = pattern_type %C [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.a32: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%Self) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.e44: <witness> = require_complete_type %C [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%pattern_type.loc14_10 (%pattern_type.68b) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.%pattern_type.loc14_10 (%pattern_type.68b) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %u.patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc20_49: type = name_ref U, %U.loc20 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:     %self.param.loc20: @F.%C (%C) = value_param call_param0
// CHECK:STDOUT:     %.loc20_24.1: type = splice_block %Self.ref.loc20 [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:       %.loc20_24.2: type = specific_constant constants.%C, @C(constants.%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %Self.ref.loc20: type = name_ref Self, %.loc20_24.2 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc20: @F.%C (%C) = bind_name self, %self.param.loc20
// CHECK:STDOUT:     %U.loc20: type = bind_symbolic_name U, 1 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:     %u.param.loc20: @F.%U.loc14_22.2 (%U) = value_param call_param1
// CHECK:STDOUT:     %U.ref.loc20_43: type = name_ref U, %U.loc20 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:     %u.loc20: @F.%U.loc14_22.2 (%U) = bind_name u, %u.param.loc20
// CHECK:STDOUT:     %return.param.loc20: ref @F.%U.loc14_22.2 (%U) = out_param call_param2
// CHECK:STDOUT:     %return.loc20: ref @F.%U.loc14_22.2 (%U) = return_slot %return.param.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [symbolic = constants.%C] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%Self) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @C.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type) = fn_decl @F [symbolic = @C.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%pattern_type.loc14_10 (%pattern_type.68b) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.%pattern_type.loc14_10 (%pattern_type.68b) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %u.patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.%pattern_type.loc14_32 (%pattern_type.a32) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref.loc14_41: type = name_ref U, %U.loc14_22.1 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:       %self.param.loc14: @F.%C (%C) = value_param call_param0
// CHECK:STDOUT:       %.loc14_16.1: type = splice_block %Self.ref.loc14 [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:         %.loc14_16.2: type = specific_constant constants.%C, @C(constants.%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:         %Self.ref.loc14: type = name_ref Self, %.loc14_16.2 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc14: @F.%C (%C) = bind_name self, %self.param.loc14
// CHECK:STDOUT:       %U.loc14_22.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:       %u.param.loc14: @F.%U.loc14_22.2 (%U) = value_param call_param1
// CHECK:STDOUT:       %U.ref.loc14_35: type = name_ref U, %U.loc14_22.1 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:       %u.loc14: @F.%U.loc14_22.2 (%U) = bind_name u, %u.param.loc14
// CHECK:STDOUT:       %return.param.loc14: ref @F.%U.loc14_22.2 (%U) = out_param call_param2
// CHECK:STDOUT:       %return.loc14: ref @F.%U.loc14_22.2 (%U) = return_slot %return.param.loc14
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %Interface.type, %U.loc14_22.1: type) {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %pattern_type.loc14_10: type = pattern_type %C [symbolic = %pattern_type.loc14_10 (constants.%pattern_type.68b)]
// CHECK:STDOUT:   %U.loc14_22.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc14_22.2 (constants.%U)]
// CHECK:STDOUT:   %pattern_type.loc14_32: type = pattern_type %U.loc14_22.2 [symbolic = %pattern_type.loc14_32 (constants.%pattern_type.a32)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_46: <witness> = require_complete_type %U.loc14_22.2 [symbolic = %require_complete.loc20_46 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %require_complete.loc20_22: <witness> = require_complete_type %C [symbolic = %require_complete.loc20_22 (constants.%require_complete.e44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc20: @F.%C (%C), %u.param.loc20: @F.%U.loc14_22.2 (%U)) -> @F.%U.loc14_22.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.ref: @F.%U.loc14_22.2 (%U) = name_ref u, %u.loc20
// CHECK:STDOUT:     return %u.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self, constants.%U) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %pattern_type.loc14_10 => constants.%pattern_type.68b
// CHECK:STDOUT:   %U.loc14_22.2 => constants.%U
// CHECK:STDOUT:   %pattern_type.loc14_32 => constants.%pattern_type.a32
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

fn Interface.F() {}

fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";

// --- dependent_return_type.carbon

library "dependent_return_type.carbon";

// Ensure that a dependent return type matches between an interface definition
// and an out-of-line member. This requires `Self` to be properly reintroduced
// into the list of generic parameters.

interface Interface {
  // TODO: This should be
  //   default fn F[self: Self](U:! type, u: U*) -> U*;
  // rather than a class member, but we don't currently accept that due to the
  // TODOs above.
  class C {
    fn F[self: Self](U:! type, u: U*) -> U*;
  }
}

// TODO: This should be
//   fn Interface.F[self: Self](U:! type, u: U*) -> U* { return u; }
fn Interface.C.F[self: Self](U:! type, u: U*) -> U* { return u; }

// CHECK:STDOUT: --- fail_todo_define_default_fn_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.F.type: type = fn_type @Interface.F [concrete]
// CHECK:STDOUT:   %Interface.F: %Interface.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.%Interface.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.88a: form = init_form %i32, call_param2 [concrete]
// CHECK:STDOUT:   %Interface.G.type: type = fn_type @Interface.G [concrete]
// CHECK:STDOUT:   %Interface.G: %Interface.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.%Interface.G.decl [concrete]
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
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type = fn_decl @Interface.F [concrete = constants.%Interface.F] {} {}
// CHECK:STDOUT:   %Interface.G.decl: %Interface.G.type = fn_decl @Interface.G [concrete = constants.%Interface.G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_35: form = init_form %i32.loc18_35, call_param2 [concrete = constants.%.88a]
// CHECK:STDOUT:     %a.param.loc18: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc18_19: type = splice_block %i32.loc18_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc18: %i32 = value_binding a, %a.param.loc18
// CHECK:STDOUT:     %b.param.loc18: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc18_27: type = splice_block %i32.loc18_27 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc18: %i32 = value_binding b, %b.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return.loc18: ref %i32 = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type = fn_decl @Interface.F [concrete = constants.%Interface.F] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Interface.G.decl: %Interface.G.type = fn_decl @Interface.G [concrete = constants.%Interface.G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_35: form = init_form %i32.loc13_35, call_param2 [concrete = constants.%.88a]
// CHECK:STDOUT:     %a.param.loc13: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc13_19: type = splice_block %i32.loc13_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc13: %i32 = value_binding a, %a.param.loc13
// CHECK:STDOUT:     %b.param.loc13: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc13_27: type = splice_block %i32.loc13_27 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc13: %i32 = value_binding b, %b.param.loc13
// CHECK:STDOUT:     %return.param.loc13: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return.loc13: ref %i32 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, %Interface.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Interface.F.decl, %Interface.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.G(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param.loc18: %i32, %b.param.loc18: %i32) -> out %return.param.loc18: %i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- dependent_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.887: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self.887) [symbolic]
// CHECK:STDOUT:   %pattern_type.68e: type = pattern_type %C [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %U.091 [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %.2cd: form = init_form %ptr.18e, call_param2 [symbolic]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%Self.887) [symbolic]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.03c: <witness> = require_complete_type %C [symbolic]
// CHECK:STDOUT:   %require_complete.56d: <witness> = require_complete_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.f1d: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.091) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.484: <witness> = lookup_impl_witness %ptr.18e, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.18e, (%Copy.lookup_impl_witness.484) [symbolic]
// CHECK:STDOUT:   %.ee1: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.d81: %.ee1 = impl_witness_access %Copy.lookup_impl_witness.484, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9f4: <specific function> = specific_impl_function %impl.elem0.d81, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [symbolic = constants.%C.F] {
// CHECK:STDOUT:     %self.patt: @C.F.%pattern_type.loc14_10 (%pattern_type.68e) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @C.F.%pattern_type.loc14_10 (%pattern_type.68e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %u.patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc20_50: type = name_ref U, %U.loc20 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:     %ptr.loc20_51: type = ptr_type %U.ref.loc20_50 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %.loc20_51: form = init_form %ptr.loc20_51, call_param2 [symbolic = %.loc14_43.1 (constants.%.2cd)]
// CHECK:STDOUT:     %self.param.loc20: @C.F.%C (%C) = value_param call_param0
// CHECK:STDOUT:     %.loc20_24.1: type = splice_block %Self.ref.loc20 [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:       %.loc20_24.2: type = specific_constant constants.%C, @C(constants.%Self.887) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %Self.ref.loc20: type = name_ref Self, %.loc20_24.2 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc20: @C.F.%C (%C) = value_binding self, %self.param.loc20
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc20: type = symbolic_binding U, 1 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:     %u.param.loc20: @C.F.%ptr.loc14_36.1 (%ptr.18e) = value_param call_param1
// CHECK:STDOUT:     %.loc20_44: type = splice_block %ptr.loc20_44 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)] {
// CHECK:STDOUT:       %U.ref.loc20_43: type = name_ref U, %U.loc20 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:       %ptr.loc20_44: type = ptr_type %U.ref.loc20_43 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.loc20: @C.F.%ptr.loc14_36.1 (%ptr.18e) = value_binding u, %u.param.loc20
// CHECK:STDOUT:     %return.param.loc20: ref @C.F.%ptr.loc14_36.1 (%ptr.18e) = out_param call_param2
// CHECK:STDOUT:     %return.loc20: ref @C.F.%ptr.loc14_36.1 (%ptr.18e) = return_slot %return.param.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self.887]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [symbolic = constants.%C] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.887)]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%Self) [symbolic = %C.F.type (constants.%C.F.type)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type) = struct_value () [symbolic = %C.F (constants.%C.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F)] {
// CHECK:STDOUT:       %self.patt: @C.F.%pattern_type.loc14_10 (%pattern_type.68e) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.F.%pattern_type.loc14_10 (%pattern_type.68e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %u.patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.F.%pattern_type.loc14_32 (%pattern_type.423) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref.loc14_42: type = name_ref U, %U.loc14_22.2 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:       %ptr.loc14_43: type = ptr_type %U.ref.loc14_42 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)]
// CHECK:STDOUT:       %.loc14_43.2: form = init_form %ptr.loc14_43, call_param2 [symbolic = %.loc14_43.1 (constants.%.2cd)]
// CHECK:STDOUT:       %self.param.loc14: @C.F.%C (%C) = value_param call_param0
// CHECK:STDOUT:       %.loc14_16.1: type = splice_block %Self.ref.loc14 [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:         %.loc14_16.2: type = specific_constant constants.%C, @C(constants.%Self.887) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:         %Self.ref.loc14: type = name_ref Self, %.loc14_16.2 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc14: @C.F.%C (%C) = value_binding self, %self.param.loc14
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc14_22.2: type = symbolic_binding U, 1 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:       %u.param.loc14: @C.F.%ptr.loc14_36.1 (%ptr.18e) = value_param call_param1
// CHECK:STDOUT:       %.loc14_36: type = splice_block %ptr.loc14_36.2 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)] {
// CHECK:STDOUT:         %U.ref.loc14_35: type = name_ref U, %U.loc14_22.2 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:         %ptr.loc14_36.2: type = ptr_type %U.ref.loc14_35 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u.loc14: @C.F.%ptr.loc14_36.1 (%ptr.18e) = value_binding u, %u.param.loc14
// CHECK:STDOUT:       %return.param.loc14: ref @C.F.%ptr.loc14_36.1 (%ptr.18e) = out_param call_param2
// CHECK:STDOUT:       %return.loc14: ref @C.F.%ptr.loc14_36.1 (%ptr.18e) = return_slot %return.param.loc14
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     .F = %C.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@Interface.%Self: %Interface.type, %U.loc14_22.2: type) {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.887)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %pattern_type.loc14_10: type = pattern_type %C [symbolic = %pattern_type.loc14_10 (constants.%pattern_type.68e)]
// CHECK:STDOUT:   %U.loc14_22.1: type = symbolic_binding U, 1 [symbolic = %U.loc14_22.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc14_36.1: type = ptr_type %U.loc14_22.1 [symbolic = %ptr.loc14_36.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %pattern_type.loc14_32: type = pattern_type %ptr.loc14_36.1 [symbolic = %pattern_type.loc14_32 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %.loc14_43.1: form = init_form %ptr.loc14_36.1, call_param2 [symbolic = %.loc14_43.1 (constants.%.2cd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_22: <witness> = require_complete_type %C [symbolic = %require_complete.loc20_22 (constants.%require_complete.03c)]
// CHECK:STDOUT:   %require_complete.loc20_41: <witness> = require_complete_type %ptr.loc14_36.1 [symbolic = %require_complete.loc20_41 (constants.%require_complete.56d)]
// CHECK:STDOUT:   %.loc20_62.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc14_22.1) [symbolic = %.loc20_62.1 (constants.%.f1d)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc14_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.484)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc14_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc20_62.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc20_62.2 (constants.%.ee1)]
// CHECK:STDOUT:   %impl.elem0.loc20_62.2: @C.F.%.loc20_62.2 (%.ee1) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_62.2 (constants.%impl.elem0.d81)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_62.2: <specific function> = specific_impl_function %impl.elem0.loc20_62.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc20_62.2 (constants.%specific_impl_fn.9f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc20: @C.F.%C (%C), %u.param.loc20: @C.F.%ptr.loc14_36.1 (%ptr.18e)) -> out %return.param.loc20: @C.F.%ptr.loc14_36.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.ref: @C.F.%ptr.loc14_36.1 (%ptr.18e) = name_ref u, %u.loc20
// CHECK:STDOUT:     %impl.elem0.loc20_62.1: @C.F.%.loc20_62.2 (%.ee1) = impl_witness_access constants.%Copy.lookup_impl_witness.484, element0 [symbolic = %impl.elem0.loc20_62.2 (constants.%impl.elem0.d81)]
// CHECK:STDOUT:     %bound_method.loc20_62.1: <bound method> = bound_method %u.ref, %impl.elem0.loc20_62.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_62.1: <specific function> = specific_impl_function %impl.elem0.loc20_62.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc20_62.2 (constants.%specific_impl_fn.9f4)]
// CHECK:STDOUT:     %bound_method.loc20_62.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc20_62.1
// CHECK:STDOUT:     %Copy.Op.call: init @C.F.%ptr.loc14_36.1 (%ptr.18e) = call %bound_method.loc20_62.2(%u.ref)
// CHECK:STDOUT:     return %Copy.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self.887) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.887
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type
// CHECK:STDOUT:   %C.F => constants.%C.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%Self.887, constants.%U.091) {
// CHECK:STDOUT:   %Self => constants.%Self.887
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %pattern_type.loc14_10 => constants.%pattern_type.68e
// CHECK:STDOUT:   %U.loc14_22.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc14_36.1 => constants.%ptr.18e
// CHECK:STDOUT:   %pattern_type.loc14_32 => constants.%pattern_type.423
// CHECK:STDOUT:   %.loc14_43.1 => constants.%.2cd
// CHECK:STDOUT: }
// CHECK:STDOUT:

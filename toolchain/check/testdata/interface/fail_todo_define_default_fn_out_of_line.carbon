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
  //   default fn F(self, U:! type, u: U*) -> U*;
  // rather than a class member, but we don't currently accept that due to the
  // TODOs above.
  class C {
    fn F(self, U:! type, u: U*) -> U*;
  }
}

// TODO: This should be
//   fn Interface.F(self, U:! type, u: U*) -> U* { return u; }
fn Interface.C.F(unused self, U:! type, u: U*) -> U* { return u; }


// CHECK:STDOUT: --- fail_todo_define_default_fn_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F.type: type = fn_type @Interface.WithSelf.F, @Interface.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F: %Interface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.WithSelf.%Interface.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %Interface.WithSelf.G.type: type = fn_type @Interface.WithSelf.G, @Interface.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.G: %Interface.WithSelf.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.WithSelf.%Interface.WithSelf.G.decl [concrete]
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
// CHECK:STDOUT:   %Interface.WithSelf.F.decl: %Interface.WithSelf.F.type = fn_decl @Interface.WithSelf.F [symbolic = constants.%Interface.WithSelf.F] {} {}
// CHECK:STDOUT:   %Interface.WithSelf.G.decl: %Interface.WithSelf.G.type = fn_decl @Interface.WithSelf.G [symbolic = constants.%Interface.WithSelf.G] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc18_35 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc18_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32.loc18_35 [concrete = constants.%.795]
// CHECK:STDOUT:     %a.param.loc18: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc18_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %a.loc18: %i32 = value_binding a, %a.param.loc18
// CHECK:STDOUT:     %b.param.loc18: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc18_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b.loc18: %i32 = value_binding b, %b.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return.loc18: ref %i32 = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Interface.WithSelf.F.decl: @Interface.WithSelf.%Interface.WithSelf.F.type (%Interface.WithSelf.F.type) = fn_decl @Interface.WithSelf.F [symbolic = @Interface.WithSelf.%Interface.WithSelf.F (constants.%Interface.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Interface.WithSelf.G.decl: @Interface.WithSelf.%Interface.WithSelf.G.type (%Interface.WithSelf.G.type) = fn_decl @Interface.WithSelf.G [symbolic = @Interface.WithSelf.%Interface.WithSelf.G (constants.%Interface.WithSelf.G)] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc18_35 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc13_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13: Core.Form = init_form %i32.loc13_35 [concrete = constants.%.795]
// CHECK:STDOUT:     %a.param.loc13: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc13_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %a.loc13: %i32 = value_binding a, %a.param.loc13
// CHECK:STDOUT:     %b.param.loc13: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc13_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b.loc13: %i32 = value_binding b, %b.param.loc13
// CHECK:STDOUT:     %return.param.loc13: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return.loc13: ref %i32 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, %Interface.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Interface.WithSelf.%assoc0
// CHECK:STDOUT:   .G = @Interface.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@Interface.WithSelf.%Interface.WithSelf.F.decl, @Interface.WithSelf.%Interface.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.G(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param.loc18: %i32, %b.param.loc18: %i32) -> out %return.param.loc18: %i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- dependent_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.3ce: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self.3ce) [symbolic]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %C [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %U.091 [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %.6d8: Core.Form = init_form %ptr.18e [symbolic]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%Self.3ce) [symbolic]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.f37: <witness> = require_complete_type %C [symbolic]
// CHECK:STDOUT:   %require_complete.56d: <witness> = require_complete_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.088: <witness> = lookup_impl_witness %ptr.18e, @Copy [symbolic]
// CHECK:STDOUT:   %.ddf: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.091) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.18e, (%Copy.lookup_impl_witness.088) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.da6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.a56: type = fn_type_with_self_type %Copy.WithSelf.Op.type.da6, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.fbe: %.a56 = impl_witness_access %Copy.lookup_impl_witness.088, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.16d: <specific function> = specific_impl_function %impl.elem0.fbe, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %self.param_patt: @C.F.%pattern_type.loc14_10 (%pattern_type.f25) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @C.F.%pattern_type.loc14_10 (%pattern_type.f25) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %u.param_patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = return_slot_pattern %return.param_patt, %ptr.loc20_52 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc20_51: type = name_ref U, %U.loc20 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:     %ptr.loc20_52: type = ptr_type %U.ref.loc20_51 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %.loc20_52: Core.Form = init_form %ptr.loc20_52 [symbolic = %.loc14_37.1 (constants.%.6d8)]
// CHECK:STDOUT:     %self.param.loc20: @C.F.%C (%C) = value_param call_param0
// CHECK:STDOUT:     %.loc20_25.1: type = splice_block %Self.ref.loc20 [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:       %.loc20_25.2: type = specific_constant constants.%C, @C(constants.%Self.3ce) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %Self.ref.loc20: type = name_ref Self, %.loc20_25.2 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc20: @C.F.%C (%C) = value_binding self, %self.param.loc20
// CHECK:STDOUT:     %.loc20_35.1: type = splice_block %.loc20_35.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc20: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc20_35.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc20: type = symbolic_binding U, 1 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:     %u.param.loc20: @C.F.%ptr.loc14_30.1 (%ptr.18e) = value_param call_param1
// CHECK:STDOUT:     %.loc20_45: type = splice_block %ptr.loc20_45 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)] {
// CHECK:STDOUT:       %U.ref.loc20_44: type = name_ref U, %U.loc20 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:       %ptr.loc20_45: type = ptr_type %U.ref.loc20_44 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.loc20: @C.F.%ptr.loc14_30.1 (%ptr.18e) = value_binding u, %u.param.loc20
// CHECK:STDOUT:     %return.param.loc20: ref @C.F.%ptr.loc14_30.1 (%ptr.18e) = out_param call_param2
// CHECK:STDOUT:     %return.loc20: ref @C.F.%ptr.loc14_30.1 (%ptr.18e) = return_slot %return.param.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3ce]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %C.decl: type = class_decl @C [symbolic = @Interface.WithSelf.%C (constants.%C)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = @Interface.WithSelf.%C.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.3ce)]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%Self) [symbolic = %C.F.type (constants.%C.F.type)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type) = struct_value () [symbolic = %C.F (constants.%C.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F)] {
// CHECK:STDOUT:       %self.param_patt: @C.F.%pattern_type.loc14_10 (%pattern_type.f25) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @C.F.%pattern_type.loc14_10 (%pattern_type.f25) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:       %u.param_patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = value_param_pattern [concrete]
// CHECK:STDOUT:       %u.patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @C.F.%pattern_type.loc14_27 (%pattern_type.423) = return_slot_pattern %return.param_patt, %ptr.loc20_52 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref.loc14_36: type = name_ref U, %U.loc14_17.2 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:       %ptr.loc14_37: type = ptr_type %U.ref.loc14_36 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:       %.loc14_37.2: Core.Form = init_form %ptr.loc14_37 [symbolic = %.loc14_37.1 (constants.%.6d8)]
// CHECK:STDOUT:       %self.param.loc14: @C.F.%C (%C) = value_param call_param0
// CHECK:STDOUT:       %.loc14_10.1: type = splice_block %Self.ref.loc14 [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:         %.loc14_10.2: type = specific_constant constants.%C, @C(constants.%Self.3ce) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:         %Self.ref.loc14: type = name_ref Self, %.loc14_10.2 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc14: @C.F.%C (%C) = value_binding self, %self.param.loc14
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.2 [concrete = type] {
// CHECK:STDOUT:         %.Self.loc14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc14_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc14_17.2: type = symbolic_binding U, 1 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:       %u.param.loc14: @C.F.%ptr.loc14_30.1 (%ptr.18e) = value_param call_param1
// CHECK:STDOUT:       %.loc14_30: type = splice_block %ptr.loc14_30.2 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)] {
// CHECK:STDOUT:         %U.ref.loc14_29: type = name_ref U, %U.loc14_17.2 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:         %ptr.loc14_30.2: type = ptr_type %U.ref.loc14_29 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u.loc14: @C.F.%ptr.loc14_30.1 (%ptr.18e) = value_binding u, %u.param.loc14
// CHECK:STDOUT:       %return.param.loc14: ref @C.F.%ptr.loc14_30.1 (%ptr.18e) = out_param call_param2
// CHECK:STDOUT:       %return.loc14: ref @C.F.%ptr.loc14_30.1 (%ptr.18e) = return_slot %return.param.loc14
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
// CHECK:STDOUT: generic fn @C.F(@Interface.%Self: %Interface.type, %U.loc14_17.2: type) {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.3ce)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %pattern_type.loc14_10: type = pattern_type %C [symbolic = %pattern_type.loc14_10 (constants.%pattern_type.f25)]
// CHECK:STDOUT:   %U.loc14_17.1: type = symbolic_binding U, 1 [symbolic = %U.loc14_17.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc14_30.1: type = ptr_type %U.loc14_17.1 [symbolic = %ptr.loc14_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %pattern_type.loc14_27: type = pattern_type %ptr.loc14_30.1 [symbolic = %pattern_type.loc14_27 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %.loc14_37.1: Core.Form = init_form %ptr.loc14_30.1 [symbolic = %.loc14_37.1 (constants.%.6d8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_25: <witness> = require_complete_type %C [symbolic = %require_complete.loc20_25 (constants.%require_complete.f37)]
// CHECK:STDOUT:   %require_complete.loc20_42: <witness> = require_complete_type %ptr.loc14_30.1 [symbolic = %require_complete.loc20_42 (constants.%require_complete.56d)]
// CHECK:STDOUT:   %.loc20_63.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc14_17.1) [symbolic = %.loc20_63.2 (constants.%.ddf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc14_30.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.088)]
// CHECK:STDOUT:   %Copy.facet.loc20_63.2: %Copy.type = facet_value %ptr.loc14_30.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc20_63.2 (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc20_63.2) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.da6)]
// CHECK:STDOUT:   %.loc20_63.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc20_63.2 [symbolic = %.loc20_63.3 (constants.%.a56)]
// CHECK:STDOUT:   %impl.elem0.loc20_63.2: @C.F.%.loc20_63.3 (%.a56) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_63.2 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_63.2: <specific function> = specific_impl_function %impl.elem0.loc20_63.2, @Copy.WithSelf.Op(%Copy.facet.loc20_63.2) [symbolic = %specific_impl_fn.loc20_63.2 (constants.%specific_impl_fn.16d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc20: @C.F.%C (%C), %u.param.loc20: @C.F.%ptr.loc14_30.1 (%ptr.18e)) -> out %return.param.loc20: @C.F.%ptr.loc14_30.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.ref: @C.F.%ptr.loc14_30.1 (%ptr.18e) = name_ref u, %u.loc20
// CHECK:STDOUT:     %impl.elem0.loc20_63.1: @C.F.%.loc20_63.3 (%.a56) = impl_witness_access constants.%Copy.lookup_impl_witness.088, element0 [symbolic = %impl.elem0.loc20_63.2 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:     %bound_method.loc20_63.1: <bound method> = bound_method %u.ref, %impl.elem0.loc20_63.1
// CHECK:STDOUT:     %Copy.facet.loc20_63.1: %Copy.type = facet_value constants.%ptr.18e, (constants.%Copy.lookup_impl_witness.088) [symbolic = %Copy.facet.loc20_63.2 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc20_63.1: %Copy.type = converted constants.%ptr.18e, %Copy.facet.loc20_63.1 [symbolic = %Copy.facet.loc20_63.2 (constants.%Copy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc20_63.1: <specific function> = specific_impl_function %impl.elem0.loc20_63.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc20_63.2 (constants.%specific_impl_fn.16d)]
// CHECK:STDOUT:     %bound_method.loc20_63.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc20_63.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @C.F.%ptr.loc14_30.1 (%ptr.18e) = call %bound_method.loc20_63.2(%u.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Self.3ce) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self.3ce) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.3ce
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type
// CHECK:STDOUT:   %C.F => constants.%C.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%Self.3ce, constants.%U.091) {
// CHECK:STDOUT:   %Self => constants.%Self.3ce
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %pattern_type.loc14_10 => constants.%pattern_type.f25
// CHECK:STDOUT:   %U.loc14_17.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc14_30.1 => constants.%ptr.18e
// CHECK:STDOUT:   %pattern_type.loc14_27 => constants.%pattern_type.423
// CHECK:STDOUT:   %.loc14_37.1 => constants.%.6d8
// CHECK:STDOUT: }
// CHECK:STDOUT:

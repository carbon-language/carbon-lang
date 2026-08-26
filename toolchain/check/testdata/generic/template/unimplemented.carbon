// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon

// --- fail_todo_unimplemented_operator.carbon

library "[[@TEST_NAME]]";

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent expression.
fn F[template T: type](x: T) -> i32 {
  return x.n * 3;
}

// --- fail_todo_unimplemented_value.carbon

library "[[@TEST_NAME]]";

class C {
  var n: i32;
}

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent value where the type is concrete but determined through
// the template dependent value.
fn F(template c: C) -> i32 {
  return c.n * 3;
}

// --- fail_todo_unimplemented_convert.carbon

library "[[@TEST_NAME]]";

fn F[template T: Core.Destroy](x: T) {
  // TODO: These diagnostics aren't very good, and we should only produce one error here.
  var unused v: T = 0;
  var unused w: i32 = x;
}


// CHECK:STDOUT: --- fail_todo_unimplemented_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %.b3e: %T = splice_inst @F.%.loc7_11.3 [template]
// CHECK:STDOUT:   %.432: @F.%.loc7_11.6 (@F.%.loc7_11.6) = splice_inst @F.%.loc7_11.5 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %MulWith.type.9f9: type = generic_interface_type @MulWith [concrete]
// CHECK:STDOUT:   %MulWith.generic: %MulWith.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4dd: type = assoc_entity_type @MulWith, @MulWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc1.37c: %MulWith.assoc_type.4dd = assoc_entity element1, imports.%Core.import_ref.9555a9.1 [concrete]
// CHECK:STDOUT:   %.b64: @F.%.loc7_14.4 (@F.%.loc7_14.4) = splice_inst @F.%.loc7_14.3 [template]
// CHECK:STDOUT:   %.557: @F.%.loc7_14.7 (@F.%.loc7_14.7) = splice_inst @F.%.loc7_14.6 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e1a: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.f9d913.1 [concrete]
// CHECK:STDOUT:   %.3d4: @F.%.loc7_17.6 (@F.%.loc7_17.6) = splice_inst @F.%.loc7_17.5 [template]
// CHECK:STDOUT:   %.a16: @F.%.loc7_17.9 (@F.%.loc7_17.9) = splice_inst @F.%.loc7_17.8 [template]
// CHECK:STDOUT:   %.edd: %i32 = splice_inst @F.%.loc7_17.11 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .MulWith = %Core.MulWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.MulWith: %MulWith.type.9f9 = import_ref Core//prelude/operators/arithmetic, MulWith, loaded [concrete = constants.%MulWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9555a9.1 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f9d913.1 = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc6_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc6_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc6_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc6_25.1 [template = %x.patt.loc6_25.2 (constants.%x.patt)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_33: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_16.2: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc6_16.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_16.2 [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc6_16.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc6_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc6_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc6_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc6_25.2 [template = %x.patt.loc6_25.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc7_11.3: <instruction> = refine_type_action %x.ref, %T.loc6_16.1 [template]
// CHECK:STDOUT:   %.loc7_11.4: @F.%T.loc6_16.1 (%T) = splice_inst %.loc7_11.3 [template = %.loc7_11.4 (constants.%.b3e)]
// CHECK:STDOUT:   %.loc7_11.5: <instruction> = access_member_action %.loc7_11.1, n [template]
// CHECK:STDOUT:   %.loc7_11.6: type = type_of_inst %.loc7_11.5 [template]
// CHECK:STDOUT:   %.loc7_11.7: @F.%.loc7_11.6 (@F.%.loc7_11.6) = splice_inst %.loc7_11.5 [template = %.loc7_11.7 (constants.%.432)]
// CHECK:STDOUT:   %.loc7_14.3: <instruction> = compound_member_access_action %.loc7_11.2, constants.%assoc1.37c [template]
// CHECK:STDOUT:   %.loc7_14.4: type = type_of_inst %.loc7_14.3 [template]
// CHECK:STDOUT:   %.loc7_14.5: @F.%.loc7_14.4 (@F.%.loc7_14.4) = splice_inst %.loc7_14.3 [template = %.loc7_14.5 (constants.%.b64)]
// CHECK:STDOUT:   %.loc7_14.6: <instruction> = call_action %.loc7_14.1, (%int_3), true [template]
// CHECK:STDOUT:   %.loc7_14.7: type = type_of_inst %.loc7_14.6 [template]
// CHECK:STDOUT:   %.loc7_14.8: @F.%.loc7_14.7 (@F.%.loc7_14.7) = splice_inst %.loc7_14.6 [template = %.loc7_14.8 (constants.%.557)]
// CHECK:STDOUT:   %.loc7_17.5: <instruction> = compound_member_access_action %.loc7_14.2, constants.%assoc0.e1a [template]
// CHECK:STDOUT:   %.loc7_17.6: type = type_of_inst %.loc7_17.5 [template]
// CHECK:STDOUT:   %.loc7_17.7: @F.%.loc7_17.6 (@F.%.loc7_17.6) = splice_inst %.loc7_17.5 [template = %.loc7_17.7 (constants.%.3d4)]
// CHECK:STDOUT:   %.loc7_17.8: <instruction> = call_action %.loc7_17.1, (), true [template]
// CHECK:STDOUT:   %.loc7_17.9: type = type_of_inst %.loc7_17.8 [template]
// CHECK:STDOUT:   %.loc7_17.10: @F.%.loc7_17.9 (@F.%.loc7_17.9) = splice_inst %.loc7_17.8 [template = %.loc7_17.10 (constants.%.a16)]
// CHECK:STDOUT:   %.loc7_17.11: <instruction> = convert_to_category_action %.loc7_17.3, element10 [template]
// CHECK:STDOUT:   %.loc7_17.12: %i32 = splice_inst %.loc7_17.11 [template = %.loc7_17.12 (constants.%.edd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc6_16.1 (%T)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc6_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc7_11.1: @F.%T.loc6_16.1 (%T) = splice_inst %.loc7_11.3 [template = %.loc7_11.4 (constants.%.b3e)]
// CHECK:STDOUT:     %.loc7_11.2: @F.%.loc7_11.6 (@F.%.loc7_11.6) = splice_inst %.loc7_11.5 [template = %.loc7_11.7 (constants.%.432)]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc7_14.1: @F.%.loc7_14.4 (@F.%.loc7_14.4) = splice_inst %.loc7_14.3 [template = %.loc7_14.5 (constants.%.b64)]
// CHECK:STDOUT:     %.loc7_14.2: @F.%.loc7_14.7 (@F.%.loc7_14.7) = splice_inst %.loc7_14.6 [template = %.loc7_14.8 (constants.%.557)]
// CHECK:STDOUT:     %.loc7_17.1: @F.%.loc7_17.6 (@F.%.loc7_17.6) = splice_inst %.loc7_17.5 [template = %.loc7_17.7 (constants.%.3d4)]
// CHECK:STDOUT:     %.loc7_17.2: @F.%.loc7_17.9 (@F.%.loc7_17.9) = splice_inst %.loc7_17.8 [template = %.loc7_17.10 (constants.%.a16)]
// CHECK:STDOUT:     %.loc7_17.3: %i32 = converted %.loc7_14.2, %.loc7_17.2 [template = %.loc7_17.10 (constants.%.a16)]
// CHECK:STDOUT:     %.loc7_17.4: %i32 = splice_inst %.loc7_17.11 [template = %.loc7_17.12 (constants.%.edd)]
// CHECK:STDOUT:     return %.loc7_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc6_25.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc6_25.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template]
// CHECK:STDOUT:   %c: %C = symbolic_binding c, 0, template [template]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.26f: @F.%.loc12_11.3 (@F.%.loc12_11.3) = splice_inst @F.%.loc12_11.2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %MulWith.type.9f9: type = generic_interface_type @MulWith [concrete]
// CHECK:STDOUT:   %MulWith.generic: %MulWith.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4dd: type = assoc_entity_type @MulWith, @MulWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc1.37c: %MulWith.assoc_type.4dd = assoc_entity element1, imports.%Core.import_ref.9555a9.1 [concrete]
// CHECK:STDOUT:   %.a77: @F.%.loc12_14.4 (@F.%.loc12_14.4) = splice_inst @F.%.loc12_14.3 [template]
// CHECK:STDOUT:   %.acc: @F.%.loc12_14.7 (@F.%.loc12_14.7) = splice_inst @F.%.loc12_14.6 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e1a: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.f9d913.1 [concrete]
// CHECK:STDOUT:   %.b05: @F.%.loc12_17.6 (@F.%.loc12_17.6) = splice_inst @F.%.loc12_17.5 [template]
// CHECK:STDOUT:   %.f8f: @F.%.loc12_17.9 (@F.%.loc12_17.9) = splice_inst @F.%.loc12_17.8 [template]
// CHECK:STDOUT:   %.132: %i32 = splice_inst @F.%.loc12_17.11 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .MulWith = %Core.MulWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.MulWith: %MulWith.type.9f9 = import_ref Core//prelude/operators/arithmetic, MulWith, loaded [concrete = constants.%MulWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9555a9.1 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f9d913.1 = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt.loc11_16.1: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_16.2 (constants.%c.patt)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_24: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc11_18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc11_16.2: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.cdf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%c.loc11_16.2: %C) {
// CHECK:STDOUT:   %c.patt.loc11_16.2: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_16.2 (constants.%c.patt)]
// CHECK:STDOUT:   %c.loc11_16.1: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_11.2: <instruction> = access_member_action %c.ref, n [template]
// CHECK:STDOUT:   %.loc12_11.3: type = type_of_inst %.loc12_11.2 [template]
// CHECK:STDOUT:   %.loc12_11.4: @F.%.loc12_11.3 (@F.%.loc12_11.3) = splice_inst %.loc12_11.2 [template = %.loc12_11.4 (constants.%.26f)]
// CHECK:STDOUT:   %.loc12_14.3: <instruction> = compound_member_access_action %.loc12_11.1, constants.%assoc1.37c [template]
// CHECK:STDOUT:   %.loc12_14.4: type = type_of_inst %.loc12_14.3 [template]
// CHECK:STDOUT:   %.loc12_14.5: @F.%.loc12_14.4 (@F.%.loc12_14.4) = splice_inst %.loc12_14.3 [template = %.loc12_14.5 (constants.%.a77)]
// CHECK:STDOUT:   %.loc12_14.6: <instruction> = call_action %.loc12_14.1, (%int_3), true [template]
// CHECK:STDOUT:   %.loc12_14.7: type = type_of_inst %.loc12_14.6 [template]
// CHECK:STDOUT:   %.loc12_14.8: @F.%.loc12_14.7 (@F.%.loc12_14.7) = splice_inst %.loc12_14.6 [template = %.loc12_14.8 (constants.%.acc)]
// CHECK:STDOUT:   %.loc12_17.5: <instruction> = compound_member_access_action %.loc12_14.2, constants.%assoc0.e1a [template]
// CHECK:STDOUT:   %.loc12_17.6: type = type_of_inst %.loc12_17.5 [template]
// CHECK:STDOUT:   %.loc12_17.7: @F.%.loc12_17.6 (@F.%.loc12_17.6) = splice_inst %.loc12_17.5 [template = %.loc12_17.7 (constants.%.b05)]
// CHECK:STDOUT:   %.loc12_17.8: <instruction> = call_action %.loc12_17.1, (), true [template]
// CHECK:STDOUT:   %.loc12_17.9: type = type_of_inst %.loc12_17.8 [template]
// CHECK:STDOUT:   %.loc12_17.10: @F.%.loc12_17.9 (@F.%.loc12_17.9) = splice_inst %.loc12_17.8 [template = %.loc12_17.10 (constants.%.f8f)]
// CHECK:STDOUT:   %.loc12_17.11: <instruction> = convert_to_category_action %.loc12_17.3, element10 [template]
// CHECK:STDOUT:   %.loc12_17.12: %i32 = splice_inst %.loc12_17.11 [template = %.loc12_17.12 (constants.%.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c.loc11_16.2 [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %.loc12_11.1: @F.%.loc12_11.3 (@F.%.loc12_11.3) = splice_inst %.loc12_11.2 [template = %.loc12_11.4 (constants.%.26f)]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc12_14.1: @F.%.loc12_14.4 (@F.%.loc12_14.4) = splice_inst %.loc12_14.3 [template = %.loc12_14.5 (constants.%.a77)]
// CHECK:STDOUT:     %.loc12_14.2: @F.%.loc12_14.7 (@F.%.loc12_14.7) = splice_inst %.loc12_14.6 [template = %.loc12_14.8 (constants.%.acc)]
// CHECK:STDOUT:     %.loc12_17.1: @F.%.loc12_17.6 (@F.%.loc12_17.6) = splice_inst %.loc12_17.5 [template = %.loc12_17.7 (constants.%.b05)]
// CHECK:STDOUT:     %.loc12_17.2: @F.%.loc12_17.9 (@F.%.loc12_17.9) = splice_inst %.loc12_17.8 [template = %.loc12_17.10 (constants.%.f8f)]
// CHECK:STDOUT:     %.loc12_17.3: %i32 = converted %.loc12_14.2, %.loc12_17.2 [template = %.loc12_17.10 (constants.%.f8f)]
// CHECK:STDOUT:     %.loc12_17.4: %i32 = splice_inst %.loc12_17.11 [template = %.loc12_17.12 (constants.%.132)]
// CHECK:STDOUT:     return %.loc12_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %c.patt.loc11_16.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc11_16.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: %Destroy.type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [template]
// CHECK:STDOUT:   %pattern_type.f4e485.2: type = pattern_type %T.as_type [template]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.f4e485.2 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt: %pattern_type.f4e485.2 = wrapper_binding_pattern x, %x.param_patt [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.14c: <witness> = require_complete_type %T.as_type [template]
// CHECK:STDOUT:   %v.patt: %pattern_type.f4e485.2 = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.f4e485.2 = var_pattern %v.patt [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.be1: @F.%.loc6_3.12 (@F.%.loc6_3.12) = splice_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %.463: @F.%.loc6_3.15 (@F.%.loc6_3.15) = splice_inst @F.%.loc6_3.14 [template]
// CHECK:STDOUT:   %.306: @F.%.loc6_3.18 (@F.%.loc6_3.18) = splice_inst @F.%.loc6_3.17 [template]
// CHECK:STDOUT:   %.6c5: @F.%.loc6_3.21 (@F.%.loc6_3.21) = splice_inst @F.%.loc6_3.20 [template]
// CHECK:STDOUT:   %.080: %T.as_type = splice_inst @F.%.loc6_3.23 [template]
// CHECK:STDOUT:   %.b62: %T.as_type = splice_inst @F.%.loc6_3.25 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.6b6 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.6b6 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e1a: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.f9d913.1 [concrete]
// CHECK:STDOUT:   %.75e: %T.as_type = splice_inst @F.%.loc7_3.6 [template]
// CHECK:STDOUT:   %.2f2: @F.%.loc7_3.9 (@F.%.loc7_3.9) = splice_inst @F.%.loc7_3.8 [template]
// CHECK:STDOUT:   %.56f: @F.%.loc7_3.12 (@F.%.loc7_3.12) = splice_inst @F.%.loc7_3.11 [template]
// CHECK:STDOUT:   %.224: %i32 = splice_inst @F.%.loc7_3.14 [template]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.407: %T.as_type = splice_inst @F.%.loc6_3.27 [template]
// CHECK:STDOUT:   %.026: @F.%.loc6_3.30 (@F.%.loc6_3.30) = splice_inst @F.%.loc6_3.29 [template]
// CHECK:STDOUT:   %.0a8: @F.%.loc6_3.33 (@F.%.loc6_3.33) = splice_inst @F.%.loc6_3.32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f9d913.1 = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc4_33.1: @F.%pattern_type (%pattern_type.f4e485.2) = value_param_pattern [template = %x.param_patt.loc4_33.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc4_33.1: @F.%pattern_type (%pattern_type.f4e485.2) = wrapper_binding_pattern x, %x.param_patt.loc4_33.1 [template = %x.patt.loc4_33.2 (constants.%x.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_22: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc4_35.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_35.1: type = splice_block %.loc4_35.2 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc4: %Destroy.type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc4_35.2: type = facet_access_type %T.ref.loc4 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_35.2: type = converted %T.ref.loc4, %T.as_type.loc4_35.2 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc4_35.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_16.2: %Destroy.type) {
// CHECK:STDOUT:   %T.patt.loc4_16.2: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_16.1: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc4_35.1: type = facet_access_type %T.loc4_16.1 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc4_35.1 [template = %pattern_type (constants.%pattern_type.f4e485.2)]
// CHECK:STDOUT:   %x.param_patt.loc4_33.2: @F.%pattern_type (%pattern_type.f4e485.2) = value_param_pattern [template = %x.param_patt.loc4_33.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc4_33.2: @F.%pattern_type (%pattern_type.f4e485.2) = wrapper_binding_pattern x, %x.param_patt.loc4_33.2 [template = %x.patt.loc4_33.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc4_35.1 [template = %require_complete (constants.%require_complete.14c)]
// CHECK:STDOUT:   %v.patt.loc6_15.2: @F.%pattern_type (%pattern_type.f4e485.2) = ref_binding_pattern v [template = %v.patt.loc6_15.2 (constants.%v.patt)]
// CHECK:STDOUT:   %v.var_patt.loc6_3.2: @F.%pattern_type (%pattern_type.f4e485.2) = var_pattern %v.patt.loc6_15.2 [template = %v.var_patt.loc6_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:   %.loc6_3.11: <instruction> = call_action constants.%ImplicitAs.generic, (constants.%T.as_type), false [template]
// CHECK:STDOUT:   %.loc6_3.12: type = type_of_inst %.loc6_3.11 [template]
// CHECK:STDOUT:   %.loc6_3.13: @F.%.loc6_3.12 (@F.%.loc6_3.12) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.be1)]
// CHECK:STDOUT:   %.loc6_3.14: <instruction> = access_member_action %.loc6_3.1, Convert [template]
// CHECK:STDOUT:   %.loc6_3.15: type = type_of_inst %.loc6_3.14 [template]
// CHECK:STDOUT:   %.loc6_3.16: @F.%.loc6_3.15 (@F.%.loc6_3.15) = splice_inst %.loc6_3.14 [template = %.loc6_3.16 (constants.%.463)]
// CHECK:STDOUT:   %.loc6_3.17: <instruction> = compound_member_access_action %int_0, %.loc6_3.2 [template]
// CHECK:STDOUT:   %.loc6_3.18: type = type_of_inst %.loc6_3.17 [template]
// CHECK:STDOUT:   %.loc6_3.19: @F.%.loc6_3.18 (@F.%.loc6_3.18) = splice_inst %.loc6_3.17 [template = %.loc6_3.19 (constants.%.306)]
// CHECK:STDOUT:   %.loc6_3.20: <instruction> = call_action %.loc6_3.3, (), true [template]
// CHECK:STDOUT:   %.loc6_3.21: type = type_of_inst %.loc6_3.20 [template]
// CHECK:STDOUT:   %.loc6_3.22: @F.%.loc6_3.21 (@F.%.loc6_3.21) = splice_inst %.loc6_3.20 [template = %.loc6_3.22 (constants.%.6c5)]
// CHECK:STDOUT:   %.loc6_3.23: <instruction> = refine_type_action %.loc6_3.5, %T.as_type.loc4_35.1 [template]
// CHECK:STDOUT:   %.loc6_3.24: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc6_3.23 [template = %.loc6_3.24 (constants.%.080)]
// CHECK:STDOUT:   %.loc6_3.25: <instruction> = convert_to_category_action %.loc6_3.6, element10 [template]
// CHECK:STDOUT:   %.loc6_3.26: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc6_3.25 [template = %.loc6_3.26 (constants.%.b62)]
// CHECK:STDOUT:   %.loc7_3.6: <instruction> = refine_type_action %x.ref, %T.as_type.loc4_35.1 [template]
// CHECK:STDOUT:   %.loc7_3.7: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc7_3.6 [template = %.loc7_3.7 (constants.%.75e)]
// CHECK:STDOUT:   %.loc7_3.8: <instruction> = compound_member_access_action %.loc7_3.1, constants.%assoc0.e1a [template]
// CHECK:STDOUT:   %.loc7_3.9: type = type_of_inst %.loc7_3.8 [template]
// CHECK:STDOUT:   %.loc7_3.10: @F.%.loc7_3.9 (@F.%.loc7_3.9) = splice_inst %.loc7_3.8 [template = %.loc7_3.10 (constants.%.2f2)]
// CHECK:STDOUT:   %.loc7_3.11: <instruction> = call_action %.loc7_3.2, (), true [template]
// CHECK:STDOUT:   %.loc7_3.12: type = type_of_inst %.loc7_3.11 [template]
// CHECK:STDOUT:   %.loc7_3.13: @F.%.loc7_3.12 (@F.%.loc7_3.12) = splice_inst %.loc7_3.11 [template = %.loc7_3.13 (constants.%.56f)]
// CHECK:STDOUT:   %.loc7_3.14: <instruction> = convert_to_category_action %.loc7_3.4, element10 [template]
// CHECK:STDOUT:   %.loc7_3.15: %i32 = splice_inst %.loc7_3.14 [template = %.loc7_3.15 (constants.%.224)]
// CHECK:STDOUT:   %.loc6_3.27: <instruction> = refine_type_action %v.var, %T.as_type.loc4_35.1 [template]
// CHECK:STDOUT:   %.loc6_3.28: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc6_3.27 [template = %.loc6_3.28 (constants.%.407)]
// CHECK:STDOUT:   %.loc6_3.29: <instruction> = compound_member_access_action %.loc6_3.8, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc6_3.30: type = type_of_inst %.loc6_3.29 [template]
// CHECK:STDOUT:   %.loc6_3.31: @F.%.loc6_3.30 (@F.%.loc6_3.30) = splice_inst %.loc6_3.29 [template = %.loc6_3.31 (constants.%.026)]
// CHECK:STDOUT:   %.loc6_3.32: <instruction> = call_action %.loc6_3.9, (), true [template]
// CHECK:STDOUT:   %.loc6_3.33: type = type_of_inst %.loc6_3.32 [template]
// CHECK:STDOUT:   %.loc6_3.34: @F.%.loc6_3.33 (@F.%.loc6_3.33) = splice_inst %.loc6_3.32 [template = %.loc6_3.34 (constants.%.0a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc4_35.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%T.as_type.loc4_35.1 (%T.as_type) = var_storage %v.var_patt.loc6_3.1
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc6_3.1: @F.%.loc6_3.12 (@F.%.loc6_3.12) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.be1)]
// CHECK:STDOUT:     %.loc6_3.2: @F.%.loc6_3.15 (@F.%.loc6_3.15) = splice_inst %.loc6_3.14 [template = %.loc6_3.16 (constants.%.463)]
// CHECK:STDOUT:     %.loc6_3.3: @F.%.loc6_3.18 (@F.%.loc6_3.18) = splice_inst %.loc6_3.17 [template = %.loc6_3.19 (constants.%.306)]
// CHECK:STDOUT:     %.loc6_3.4: @F.%.loc6_3.21 (@F.%.loc6_3.21) = splice_inst %.loc6_3.20 [template = %.loc6_3.22 (constants.%.6c5)]
// CHECK:STDOUT:     %.loc6_3.5: @F.%T.as_type.loc4_35.1 (%T.as_type) = converted %int_0, %.loc6_3.4 [template = %.loc6_3.22 (constants.%.6c5)]
// CHECK:STDOUT:     %.loc6_3.6: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc6_3.23 [template = %.loc6_3.24 (constants.%.080)]
// CHECK:STDOUT:     %.loc6_3.7: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc6_3.25 [template = %.loc6_3.26 (constants.%.b62)]
// CHECK:STDOUT:     assign %v.var, %.loc6_3.7
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc6: %Destroy.type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6: type = facet_access_type %T.ref.loc6 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17.2: type = converted %T.ref.loc6, %T.as_type.loc6 [template = %T.as_type.loc4_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%T.as_type.loc4_35.1 (%T.as_type) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc6_15.1: @F.%pattern_type (%pattern_type.f4e485.2) = ref_binding_pattern v [template = %v.patt.loc6_15.2 (constants.%v.patt)]
// CHECK:STDOUT:       %v.var_patt.loc6_3.1: @F.%pattern_type (%pattern_type.f4e485.2) = var_pattern %v.patt.loc6_15.1 [template = %v.var_patt.loc6_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref %i32 = var_storage %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.as_type.loc4_35.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc7_3.1: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc7_3.6 [template = %.loc7_3.7 (constants.%.75e)]
// CHECK:STDOUT:     %.loc7_3.2: @F.%.loc7_3.9 (@F.%.loc7_3.9) = splice_inst %.loc7_3.8 [template = %.loc7_3.10 (constants.%.2f2)]
// CHECK:STDOUT:     %.loc7_3.3: @F.%.loc7_3.12 (@F.%.loc7_3.12) = splice_inst %.loc7_3.11 [template = %.loc7_3.13 (constants.%.56f)]
// CHECK:STDOUT:     %.loc7_3.4: %i32 = converted %x.ref, %.loc7_3.3 [template = %.loc7_3.13 (constants.%.56f)]
// CHECK:STDOUT:     %.loc7_3.5: %i32 = splice_inst %.loc7_3.14 [template = %.loc7_3.15 (constants.%.224)]
// CHECK:STDOUT:     assign %w.var, %.loc7_3.5
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %w: ref %i32 = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.6b6 = ref_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.6b6 = var_pattern %w.patt [concrete = constants.%w.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %w.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%w.var)
// CHECK:STDOUT:     %.loc6_3.8: @F.%T.as_type.loc4_35.1 (%T.as_type) = splice_inst %.loc6_3.27 [template = %.loc6_3.28 (constants.%.407)]
// CHECK:STDOUT:     %.loc6_3.9: @F.%.loc6_3.30 (@F.%.loc6_3.30) = splice_inst %.loc6_3.29 [template = %.loc6_3.31 (constants.%.026)]
// CHECK:STDOUT:     %.loc6_3.10: @F.%.loc6_3.33 (@F.%.loc6_3.33) = splice_inst %.loc6_3.32 [template = %.loc6_3.34 (constants.%.0a8)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc4_35.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %x.param_patt.loc4_33.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc4_33.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:

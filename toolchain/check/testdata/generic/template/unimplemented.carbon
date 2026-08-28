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

fn G() -> i32 {
  // CHECK:STDERR: fail_todo_unimplemented_operator.carbon:[[@LINE+7]]:10: error: unable to monomorphize specific `F(())` [ResolvingSpecificHere]
  // CHECK:STDERR:   return F(());
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_todo_unimplemented_operator.carbon:[[@LINE-7]]:10: note: type `()` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   return x.n * 3;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return F(());
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

fn G() -> i32 {
  // CHECK:STDERR: fail_todo_unimplemented_value.carbon:[[@LINE+7]]:10: error: unable to monomorphize specific `F({.n = 123})` [ResolvingSpecificHere]
  // CHECK:STDERR:   return F({ .n = 123 } as C);
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_todo_unimplemented_value.carbon:[[@LINE-7]]:10: note: value of type `<bound method>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   return c.n * 3;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return F({ .n = 123 } as C);
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
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %.4321: @F.%.loc7_11.6 (@F.%.loc7_11.6) = splice_inst @F.%.loc7_11.5 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %MulWith.type.9f9: type = generic_interface_type @MulWith [concrete]
// CHECK:STDOUT:   %MulWith.generic: %MulWith.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4dd: type = assoc_entity_type @MulWith, @MulWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc1.37c: %MulWith.assoc_type.4dd = assoc_entity element1, imports.%Core.import_ref.9555a9.1 [concrete]
// CHECK:STDOUT:   %.b64: @F.%.loc7_14.4 (@F.%.loc7_14.4) = splice_inst @F.%.loc7_14.3 [template]
// CHECK:STDOUT:   %.eb3: @F.%.loc7_14.7 (@F.%.loc7_14.7) = splice_inst @F.%.loc7_14.6 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e1a: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.f9d913.1 [concrete]
// CHECK:STDOUT:   %.e5e: @F.%.loc7_17.6 (@F.%.loc7_17.6) = splice_inst @F.%.loc7_17.5 [template]
// CHECK:STDOUT:   %.fb2: @F.%.loc7_17.9 (@F.%.loc7_17.9) = splice_inst @F.%.loc7_17.8 [template]
// CHECK:STDOUT:   %.740: %i32 = splice_inst @F.%.loc7_17.11 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt.334: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.218: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt.334 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.269: %empty_tuple.type = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432c: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc6_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc6_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc6_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc6_25.1 [template = %x.patt.loc6_25.2 (constants.%x.patt.260)]
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
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc6_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc6_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc6_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc6_25.2 [template = %x.patt.loc6_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc7_11.3: <instruction> = refine_type_action %x.ref, %T.loc6_16.1 [template]
// CHECK:STDOUT:   %.loc7_11.4: @F.%T.loc6_16.1 (%T) = splice_inst %.loc7_11.3 [template = %.loc7_11.4 (constants.%.b3e)]
// CHECK:STDOUT:   %.loc7_11.5: <instruction> = access_member_action %.loc7_11.1, n [template]
// CHECK:STDOUT:   %.loc7_11.6: type = type_of_inst %.loc7_11.5 [template]
// CHECK:STDOUT:   %.loc7_11.7: @F.%.loc7_11.6 (@F.%.loc7_11.6) = splice_inst %.loc7_11.5 [template = %.loc7_11.7 (constants.%.4321)]
// CHECK:STDOUT:   %.loc7_14.3: <instruction> = compound_member_access_action %.loc7_11.2, constants.%assoc1.37c [template]
// CHECK:STDOUT:   %.loc7_14.4: type = type_of_inst %.loc7_14.3 [template]
// CHECK:STDOUT:   %.loc7_14.5: @F.%.loc7_14.4 (@F.%.loc7_14.4) = splice_inst %.loc7_14.3 [template = %.loc7_14.5 (constants.%.b64)]
// CHECK:STDOUT:   %.loc7_14.6: <instruction> = call_action (%.loc7_14.1, %int_3), true [template]
// CHECK:STDOUT:   %.loc7_14.7: type = type_of_inst %.loc7_14.6 [template]
// CHECK:STDOUT:   %.loc7_14.8: @F.%.loc7_14.7 (@F.%.loc7_14.7) = splice_inst %.loc7_14.6 [template = %.loc7_14.8 (constants.%.eb3)]
// CHECK:STDOUT:   %.loc7_17.5: <instruction> = compound_member_access_action %.loc7_14.2, constants.%assoc0.e1a [template]
// CHECK:STDOUT:   %.loc7_17.6: type = type_of_inst %.loc7_17.5 [template]
// CHECK:STDOUT:   %.loc7_17.7: @F.%.loc7_17.6 (@F.%.loc7_17.6) = splice_inst %.loc7_17.5 [template = %.loc7_17.7 (constants.%.e5e)]
// CHECK:STDOUT:   %.loc7_17.8: <instruction> = call_action (%.loc7_17.1), true [template]
// CHECK:STDOUT:   %.loc7_17.9: type = type_of_inst %.loc7_17.8 [template]
// CHECK:STDOUT:   %.loc7_17.10: @F.%.loc7_17.9 (@F.%.loc7_17.9) = splice_inst %.loc7_17.8 [template = %.loc7_17.10 (constants.%.fb2)]
// CHECK:STDOUT:   %.loc7_17.11: <instruction> = convert_to_category_action %.loc7_17.3, element10 [template]
// CHECK:STDOUT:   %.loc7_17.12: %i32 = splice_inst %.loc7_17.11 [template = %.loc7_17.12 (constants.%.740)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc6_16.1 (%T)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc6_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc7_11.1: @F.%T.loc6_16.1 (%T) = splice_inst %.loc7_11.3 [template = %.loc7_11.4 (constants.%.b3e)]
// CHECK:STDOUT:     %.loc7_11.2: @F.%.loc7_11.6 (@F.%.loc7_11.6) = splice_inst %.loc7_11.5 [template = %.loc7_11.7 (constants.%.4321)]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc7_14.1: @F.%.loc7_14.4 (@F.%.loc7_14.4) = splice_inst %.loc7_14.3 [template = %.loc7_14.5 (constants.%.b64)]
// CHECK:STDOUT:     %.loc7_14.2: @F.%.loc7_14.7 (@F.%.loc7_14.7) = splice_inst %.loc7_14.6 [template = %.loc7_14.8 (constants.%.eb3)]
// CHECK:STDOUT:     %.loc7_17.1: @F.%.loc7_17.6 (@F.%.loc7_17.6) = splice_inst %.loc7_17.5 [template = %.loc7_17.7 (constants.%.e5e)]
// CHECK:STDOUT:     %.loc7_17.2: @F.%.loc7_17.9 (@F.%.loc7_17.9) = splice_inst %.loc7_17.8 [template = %.loc7_17.10 (constants.%.fb2)]
// CHECK:STDOUT:     %.loc7_17.3: %i32 = converted %.loc7_14.2, %.loc7_17.2 [template = %.loc7_17.10 (constants.%.fb2)]
// CHECK:STDOUT:     %.loc7_17.4: %i32 = splice_inst %.loc7_17.11 [template = %.loc7_17.12 (constants.%.740)]
// CHECK:STDOUT:     return %.loc7_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_13.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%empty_tuple.type) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_13.2: %empty_tuple.type = converted %.loc18_13.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc18_13.2)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc6_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc6_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT:   %x.param_patt.loc6_25.2 => constants.%x.param_patt.334
// CHECK:STDOUT:   %x.patt.loc6_25.2 => constants.%x.patt.218
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.782
// CHECK:STDOUT:   %.loc7_11.3 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc7_11.4 => invalid
// CHECK:STDOUT:   %.loc7_11.5 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc7_11.6 => <error>
// CHECK:STDOUT:   %.loc7_11.7 => <error>
// CHECK:STDOUT:   %.loc7_14.3 => <error>
// CHECK:STDOUT:   %.loc7_14.4 => <error>
// CHECK:STDOUT:   %.loc7_14.5 => <error>
// CHECK:STDOUT:   %.loc7_14.6 => <error>
// CHECK:STDOUT:   %.loc7_14.7 => <error>
// CHECK:STDOUT:   %.loc7_14.8 => <error>
// CHECK:STDOUT:   %.loc7_17.5 => <error>
// CHECK:STDOUT:   %.loc7_17.6 => <error>
// CHECK:STDOUT:   %.loc7_17.7 => <error>
// CHECK:STDOUT:   %.loc7_17.8 => <error>
// CHECK:STDOUT:   %.loc7_17.9 => <error>
// CHECK:STDOUT:   %.loc7_17.10 => <error>
// CHECK:STDOUT:   %.loc7_17.11 => <error>
// CHECK:STDOUT:   %.loc7_17.12 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.9ed: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n.9ed [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template]
// CHECK:STDOUT:   %c: %C = symbolic_binding c, 0, template [template]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.26f: @F.%.loc12_11.3 (@F.%.loc12_11.3) = splice_inst @F.%.loc12_11.2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %MulWith.type.9f9: type = generic_interface_type @MulWith [concrete]
// CHECK:STDOUT:   %MulWith.generic: %MulWith.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %MulWith.type.f84: type = facet_type <@MulWith, @MulWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.941: %MulWith.type.f84 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.176: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.941) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.a75: %MulWith.WithSelf.Op.type.176 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.fbc: type = facet_type <@MulWith, @MulWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4dd: type = assoc_entity_type @MulWith, @MulWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc1.936: %MulWith.assoc_type.4dd = assoc_entity element1, imports.%Core.import_ref.5a0 [concrete]
// CHECK:STDOUT:   %.990: @F.%.loc12_14.4 (@F.%.loc12_14.4) = splice_inst @F.%.loc12_14.3 [template]
// CHECK:STDOUT:   %.211: @F.%.loc12_14.7 (@F.%.loc12_14.7) = splice_inst @F.%.loc12_14.6 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa489.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.29426c.1: %ImplicitAs.type.3aa489.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97ed63.1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.29426c.1) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.9453b4.1: %ImplicitAs.WithSelf.Convert.type.97ed63.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.368: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %.e9d: @F.%.loc12_17.6 (@F.%.loc12_17.6) = splice_inst @F.%.loc12_17.5 [template]
// CHECK:STDOUT:   %.de0: @F.%.loc12_17.9 (@F.%.loc12_17.9) = splice_inst @F.%.loc12_17.8 [template]
// CHECK:STDOUT:   %.087: %i32 = splice_inst @F.%.loc12_17.11 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.44a = struct_value (%int_123.fff) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.450: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_123.2bf) [concrete]
// CHECK:STDOUT:   %.77a: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc23_25.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.77a, %Destroy.Op.1a2547.4 [concrete]
// CHECK:STDOUT:   %inst.splice_block.821: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.def: %i32 = splice_block %.84a {
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%.loc5 [concrete = @C.%.loc5]
// CHECK:STDOUT:       %.eb7: ref %i32 = class_element_access %C.val, element0 [concrete = %int_123.2bf]
// CHECK:STDOUT:       %.84a: %i32 = acquire_value %.eb7
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.type.b2cf9a.1: type = fn_type @Int.as.MulWith.impl.Op.2, @Int.as.MulWith.impl.55f(%N, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.6353dd.1: %Int.as.MulWith.impl.Op.type.b2cf9a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.impl_witness.e35: <witness> = impl_witness imports.%MulWith.impl_witness_table.3fd, @Int.as.MulWith.impl.55f(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.type.8a9684.2: type = fn_type @Int.as.MulWith.impl.Op.2, @Int.as.MulWith.impl.55f(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.472625.2: %Int.as.MulWith.impl.Op.type.8a9684.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet.4c6: %MulWith.type.fbc = facet_value %i32, (%MulWith.impl_witness.e35) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.f7e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(Core.IntLiteral, %MulWith.facet.4c6) [concrete]
// CHECK:STDOUT:   %.3bb: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.f7e, %MulWith.facet.4c6 [concrete]
// CHECK:STDOUT:   %inst.splice_block.5ad: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c42: <bound method> = splice_block %bound_method.716 {
// CHECK:STDOUT:       %impl.elem1.f2d: %.3bb = impl_witness_access %MulWith.impl_witness.e35, element1 [concrete = %Int.as.MulWith.impl.Op.472625.2]
// CHECK:STDOUT:       %bound_method.716: <bound method> = bound_method %.def, %impl.elem1.f2d
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.edd: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .MulWith = %Core.MulWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.MulWith: %MulWith.type.9f9 = import_ref Core//prelude/operators/arithmetic, MulWith, loaded [concrete = constants.%MulWith.generic]
// CHECK:STDOUT:   %Core.import_ref.5a0: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.176) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.a75)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97ed63.1) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.9453b4.1)]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.b4fd3a.2 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.704: @Int.as.MulWith.impl.55f.%Int.as.MulWith.impl.Op.type.2 (%Int.as.MulWith.impl.Op.type.b2cf9a.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulWith.impl.55f.%Int.as.MulWith.impl.Op.2 (constants.%Int.as.MulWith.impl.Op.6353dd.1)]
// CHECK:STDOUT:   %MulWith.impl_witness_table.3fd = impl_witness_table (%Core.import_ref.b4fd3a.2, %Core.import_ref.704), @Int.as.MulWith.impl.55f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt.loc11_16.1: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_16.2 (constants.%c.patt)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_24: Core.Form = init_form %i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %.loc11_18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc11_16.2: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.9ed [concrete = constants.%complete_type.cdf]
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
// CHECK:STDOUT:   %.loc12_14.3: <instruction> = compound_member_access_action %.loc12_11.1, constants.%assoc1.936 [template]
// CHECK:STDOUT:   %.loc12_14.4: type = type_of_inst %.loc12_14.3 [template]
// CHECK:STDOUT:   %.loc12_14.5: @F.%.loc12_14.4 (@F.%.loc12_14.4) = splice_inst %.loc12_14.3 [template = %.loc12_14.5 (constants.%.990)]
// CHECK:STDOUT:   %.loc12_14.6: <instruction> = call_action (%.loc12_14.1, %int_3), true [template]
// CHECK:STDOUT:   %.loc12_14.7: type = type_of_inst %.loc12_14.6 [template]
// CHECK:STDOUT:   %.loc12_14.8: @F.%.loc12_14.7 (@F.%.loc12_14.7) = splice_inst %.loc12_14.6 [template = %.loc12_14.8 (constants.%.211)]
// CHECK:STDOUT:   %.loc12_17.5: <instruction> = compound_member_access_action %.loc12_14.2, constants.%assoc0.368 [template]
// CHECK:STDOUT:   %.loc12_17.6: type = type_of_inst %.loc12_17.5 [template]
// CHECK:STDOUT:   %.loc12_17.7: @F.%.loc12_17.6 (@F.%.loc12_17.6) = splice_inst %.loc12_17.5 [template = %.loc12_17.7 (constants.%.e9d)]
// CHECK:STDOUT:   %.loc12_17.8: <instruction> = call_action (%.loc12_17.1), true [template]
// CHECK:STDOUT:   %.loc12_17.9: type = type_of_inst %.loc12_17.8 [template]
// CHECK:STDOUT:   %.loc12_17.10: @F.%.loc12_17.9 (@F.%.loc12_17.9) = splice_inst %.loc12_17.8 [template = %.loc12_17.10 (constants.%.de0)]
// CHECK:STDOUT:   %.loc12_17.11: <instruction> = convert_to_category_action %.loc12_17.3, element10 [template]
// CHECK:STDOUT:   %.loc12_17.12: %i32 = splice_inst %.loc12_17.11 [template = %.loc12_17.12 (constants.%.087)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c.loc11_16.2 [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %.loc12_11.1: @F.%.loc12_11.3 (@F.%.loc12_11.3) = splice_inst %.loc12_11.2 [template = %.loc12_11.4 (constants.%.26f)]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc12_14.1: @F.%.loc12_14.4 (@F.%.loc12_14.4) = splice_inst %.loc12_14.3 [template = %.loc12_14.5 (constants.%.990)]
// CHECK:STDOUT:     %.loc12_14.2: @F.%.loc12_14.7 (@F.%.loc12_14.7) = splice_inst %.loc12_14.6 [template = %.loc12_14.8 (constants.%.211)]
// CHECK:STDOUT:     %.loc12_17.1: @F.%.loc12_17.6 (@F.%.loc12_17.6) = splice_inst %.loc12_17.5 [template = %.loc12_17.7 (constants.%.e9d)]
// CHECK:STDOUT:     %.loc12_17.2: @F.%.loc12_17.9 (@F.%.loc12_17.9) = splice_inst %.loc12_17.8 [template = %.loc12_17.10 (constants.%.de0)]
// CHECK:STDOUT:     %.loc12_17.3: %i32 = converted %.loc12_14.2, %.loc12_17.2 [template = %.loc12_17.10 (constants.%.de0)]
// CHECK:STDOUT:     %.loc12_17.4: %i32 = splice_inst %.loc12_17.11 [template = %.loc12_17.12 (constants.%.087)]
// CHECK:STDOUT:     return %.loc12_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %.loc23_23.1: %struct_type.n.44a = struct_literal (%int_123) [concrete = constants.%struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc23_23.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_23.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method.450]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc23_23.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc23_23.2: init %i32 = converted %int_123, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc23_23.3: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc23_23.4: ref %i32 = class_element_access %.loc23_23.3, element0
// CHECK:STDOUT:   %.loc23_23.5: init %i32 to %.loc23_23.4 = in_place_init %.loc23_23.2 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc23_23.6: init %C to %.loc23_23.3 = class_init (%.loc23_23.5) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_25.1: init %C = converted %.loc23_23.1, %.loc23_23.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_25.2: ref %C = temporary %.loc23_23.3, %.loc23_25.1 [concrete = constants.%.77a]
// CHECK:STDOUT:   %.loc23_25.3: %C = acquire_value %.loc23_25.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C.val) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.77a)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_25.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_25.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_25.3(%self.param: ref %struct_type.n.9ed) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_25.4(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %c.patt.loc11_16.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc11_16.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C.val) {
// CHECK:STDOUT:   %c.patt.loc11_16.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc11_16.1 => constants.%C.val
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_11.2 => constants.%inst.splice_block.821
// CHECK:STDOUT:   %.loc12_11.3 => constants.%i32
// CHECK:STDOUT:   %.loc12_11.4 => invalid
// CHECK:STDOUT:   %.loc12_14.3 => constants.%inst.splice_block.5ad
// CHECK:STDOUT:   %.loc12_14.4 => <bound method>
// CHECK:STDOUT:   %.loc12_14.5 => invalid
// CHECK:STDOUT:   %.loc12_14.6 => constants.%inst.splice_block.edd
// CHECK:STDOUT:   %.loc12_14.7 => <error>
// CHECK:STDOUT:   %.loc12_14.8 => <error>
// CHECK:STDOUT:   %.loc12_17.5 => <error>
// CHECK:STDOUT:   %.loc12_17.6 => <error>
// CHECK:STDOUT:   %.loc12_17.7 => <error>
// CHECK:STDOUT:   %.loc12_17.8 => <error>
// CHECK:STDOUT:   %.loc12_17.9 => <error>
// CHECK:STDOUT:   %.loc12_17.10 => <error>
// CHECK:STDOUT:   %.loc12_17.11 => <error>
// CHECK:STDOUT:   %.loc12_17.12 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

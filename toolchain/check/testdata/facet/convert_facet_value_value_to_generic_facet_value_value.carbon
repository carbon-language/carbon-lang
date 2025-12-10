// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_facet_value_value_to_generic_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_facet_value_value_to_generic_facet_value_value.carbon

interface Edible {}

class Grass {}
impl Grass as Edible {}

interface Animal {}
interface Eats(Food:! type) {}

// When answering a query "does Goat impl Animal", we must avoid trying to deduce
// parameters for this impl. Not only is doing so unnecessary, it would start a new
// "does Goat impl Animal" query, leading to a "cycle in impl lookup" error.
impl forall [T:! Animal, U:! Edible] T as Eats(U) {}

class Goat {}
impl Goat as Animal {}

fn Feed[Food:! Edible, T:! Eats(Food)](e: T, food: Food) {}
fn HandleAnimal[A:! Animal, Food:! Edible](a: A, food: Food) { Feed(a, food); }

fn F() {
  HandleAnimal({} as Goat, {} as Grass);
}

// CHECK:STDOUT: --- convert_facet_value_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Edible.type: type = facet_type <@Edible> [concrete]
// CHECK:STDOUT:   %Self.cc5: %Edible.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Grass: type = class_type @Grass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness @Grass.as.Edible.impl.%Edible.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.614: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Food.67d: type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Eats.type.321: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.321 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.b83: type = facet_type <@Eats, @Eats(%Food.67d)> [symbolic]
// CHECK:STDOUT:   %Self.032: %Eats.type.b83 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.bfe: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.71b: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %U: %Edible.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.981: type = pattern_type %Edible.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.d94: type = symbolic_binding_type T, 0, %T.bfe [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %Eats.type.1b7e7c.1: type = facet_type <@Eats, @Eats(%U.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.cce0fb.1: <witness> = impl_witness @T.binding.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%T.bfe, %U) [symbolic]
// CHECK:STDOUT:   %Self.42a: %Eats.type.1b7e7c.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.52c920.1: <witness> = require_complete_type %Eats.type.1b7e7c.1 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Food.4a8: %Edible.type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.binding.as_type.d04: type = symbolic_binding_type Food, 0, %Food.4a8 [symbolic]
// CHECK:STDOUT:   %Eats.type.133: type = facet_type <@Eats, @Eats(%Food.binding.as_type.d04)> [symbolic]
// CHECK:STDOUT:   %T.e62: %Eats.type.133 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.016: type = pattern_type %Eats.type.133 [symbolic]
// CHECK:STDOUT:   %Self.84f: %Eats.type.133 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.a55: type = symbolic_binding_type T, 1, %T.e62 [symbolic]
// CHECK:STDOUT:   %pattern_type.6fa: type = pattern_type %T.binding.as_type.a55 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b5: type = pattern_type %Food.binding.as_type.d04 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.8cc: <witness> = require_complete_type %T.binding.as_type.a55 [symbolic]
// CHECK:STDOUT:   %require_complete.13c: <witness> = require_complete_type %Food.binding.as_type.d04 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %Food.a7e: %Edible.type = symbolic_binding Food, 1 [symbolic]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A [symbolic]
// CHECK:STDOUT:   %pattern_type.5aa: type = pattern_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %Food.binding.as_type.cd7: type = symbolic_binding_type Food, 1, %Food.a7e [symbolic]
// CHECK:STDOUT:   %pattern_type.797: type = pattern_type %Food.binding.as_type.cd7 [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1b9: <witness> = require_complete_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.9fb: <witness> = require_complete_type %Food.binding.as_type.cd7 [symbolic]
// CHECK:STDOUT:   %Eats.type.1b7e7c.2: type = facet_type <@Eats, @Eats(%Food.binding.as_type.cd7)> [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.cce0fb.2: <witness> = impl_witness @T.binding.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%A, %Food.a7e) [symbolic]
// CHECK:STDOUT:   %require_complete.52c920.2: <witness> = require_complete_type %Eats.type.1b7e7c.2 [symbolic]
// CHECK:STDOUT:   %.c9c: require_specific_def_type = require_specific_def @T.binding.as_type.as.Eats.impl(%A, %Food.a7e) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A, @Eats, @Eats(%Food.binding.as_type.cd7) [symbolic]
// CHECK:STDOUT:   %Eats.facet.c04: %Eats.type.1b7e7c.2 = facet_value %A.binding.as_type, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.451: type = pattern_type %Eats.type.1b7e7c.2 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.9db: <specific function> = specific_function %Feed, @Feed(%Food.a7e, %Eats.facet.c04) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value %Grass, (%Edible.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %Grass [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.d37: %type_where = facet_value %Grass, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.776: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d37) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.b72: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.776 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.f94: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.b72, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.d37) [concrete]
// CHECK:STDOUT:   %facet_value.cfa: %type_where = facet_value %Goat, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.e32: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.cfa) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.8f6: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.9be: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.8f6, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.cfa) [concrete]
// CHECK:STDOUT:   %Eats.type.749: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Eats.impl_witness.61e: <witness> = impl_witness @T.binding.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Self.813: %Eats.type.749 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.d50: <witness> = complete_type_witness %Eats.type.749 [concrete]
// CHECK:STDOUT:   %.12a: require_specific_def_type = require_specific_def @T.binding.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.ad3: %Eats.type.749 = facet_value %Goat, (%Eats.impl_witness.61e) [concrete]
// CHECK:STDOUT:   %pattern_type.3f6: type = pattern_type %Eats.type.749 [concrete]
// CHECK:STDOUT:   %Feed.specific_fn.177: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.ad3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Edible = %Edible.decl
// CHECK:STDOUT:     .Grass = %Grass.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Edible.decl: type = interface_decl @Edible [concrete = constants.%Edible.type] {} {}
// CHECK:STDOUT:   %Grass.decl: type = class_decl @Grass [concrete = constants.%Grass] {} {}
// CHECK:STDOUT:   impl_decl @Grass.as.Edible.impl [concrete] {} {
// CHECK:STDOUT:     %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Eats.decl: %Eats.type.321 = interface_decl @Eats [concrete = constants.%Eats.generic] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.98f = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Food.loc21_16.2: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.71b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.981 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T.bfe)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d94)]
// CHECK:STDOUT:     %.loc26_38: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d94)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.321 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc26_26.1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc26_49: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %Eats.type.loc26_49.1: type = facet_type <@Eats, @Eats(constants.%U.binding.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.1b7e7c.1)]
// CHECK:STDOUT:     %.loc26_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_14.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T.bfe)]
// CHECK:STDOUT:     %.loc26_30: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc26_26.1: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.981 = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @Feed.%pattern_type.loc31_24 (%pattern_type.016) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type.loc31_40 (%pattern_type.6fa) = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type.loc31_40 (%pattern_type.6fa) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:     %food.patt: @Feed.%pattern_type.loc31_46 (%pattern_type.3b5) = value_binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @Feed.%pattern_type.loc31_46 (%pattern_type.3b5) = value_param_pattern %food.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_16: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc31_9.2: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.4a8)]
// CHECK:STDOUT:     %.loc31_37.1: type = splice_block %Eats.type.loc31_37.2 [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.133)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: %Eats.type.321 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc31_33: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.4a8)]
// CHECK:STDOUT:       %Food.as_type.loc31_37: type = facet_access_type %Food.ref.loc31_33 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.d04)]
// CHECK:STDOUT:       %.loc31_37.2: type = converted %Food.ref.loc31_33, %Food.as_type.loc31_37 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.d04)]
// CHECK:STDOUT:       %Eats.type.loc31_37.2: type = facet_type <@Eats, @Eats(constants.%Food.binding.as_type.d04)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.133)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.133) = symbolic_binding T, 1 [symbolic = %T.loc31_24.1 (constants.%T.e62)]
// CHECK:STDOUT:     %e.param: @Feed.%T.binding.as_type (%T.binding.as_type.a55) = value_param call_param0
// CHECK:STDOUT:     %.loc31_43.1: type = splice_block %.loc31_43.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.a55)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc31_37.1 (%Eats.type.133) = name_ref T, %T.loc31_24.2 [symbolic = %T.loc31_24.1 (constants.%T.e62)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.a55)]
// CHECK:STDOUT:       %.loc31_43.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.a55)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.binding.as_type (%T.binding.as_type.a55) = value_binding e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.binding.as_type (%Food.binding.as_type.d04) = value_param call_param1
// CHECK:STDOUT:     %.loc31_52.1: type = splice_block %.loc31_52.2 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.d04)] {
// CHECK:STDOUT:       %Food.ref.loc31_52: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.4a8)]
// CHECK:STDOUT:       %Food.as_type.loc31_52: type = facet_access_type %Food.ref.loc31_52 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.d04)]
// CHECK:STDOUT:       %.loc31_52.2: type = converted %Food.ref.loc31_52, %Food.as_type.loc31_52 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.d04)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.binding.as_type (%Food.binding.as_type.d04) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %A.patt: %pattern_type.71b = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %Food.patt: %pattern_type.981 = symbolic_binding_pattern Food, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.5aa) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.5aa) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.797) = value_binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.797) = value_param_pattern %food.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc32_17.2: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:     %.loc32_36: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc32_29.2: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.a7e)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc32_47.1: type = splice_block %.loc32_47.2 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)] {
// CHECK:STDOUT:       %A.ref: %Animal.type = name_ref A, %A.loc32_17.2 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type: type = facet_access_type %A.ref [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:       %.loc32_47.2: type = converted %A.ref, %A.as_type [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = value_binding a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.cd7) = value_param call_param1
// CHECK:STDOUT:     %.loc32_56.1: type = splice_block %.loc32_56.2 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.cd7)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc32_29.2 [symbolic = %Food.loc32_29.1 (constants.%Food.a7e)]
// CHECK:STDOUT:       %Food.as_type: type = facet_access_type %Food.ref [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.cd7)]
// CHECK:STDOUT:       %.loc32_56.2: type = converted %Food.ref, %Food.as_type [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.cd7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.cd7) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Edible {
// CHECK:STDOUT:   %Self: %Edible.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cc5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.614]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Eats(%Food.loc21_16.2: type) {
// CHECK:STDOUT:   %Food.loc21_16.1: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc21_16.1)> [symbolic = %Eats.type (constants.%Eats.type.b83)]
// CHECK:STDOUT:   %Self.loc21_29.2: @Eats.%Eats.type (%Eats.type.b83) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.032)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc21_29.1: @Eats.%Eats.type (%Eats.type.b83) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.032)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc21_29.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Grass.as.Edible.impl: %Grass.ref as %Edible.ref {
// CHECK:STDOUT:   %Edible.impl_witness_table = impl_witness_table (), @Grass.as.Edible.impl [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness %Edible.impl_witness_table [concrete = constants.%Edible.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Edible.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Eats.impl(%T.loc26_14.1: %Animal.type, %U.loc26_26.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc26_14.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T.bfe)]
// CHECK:STDOUT:   %U.loc26_26.2: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc26_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d94)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc26_26.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %Eats.type.loc26_49.2: type = facet_type <@Eats, @Eats(%U.binding.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.1b7e7c.1)]
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2: <witness> = impl_witness %Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%T.loc26_14.2, %U.loc26_26.2) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.cce0fb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Eats.type.loc26_49.2 [symbolic = %require_complete (constants.%require_complete.52c920.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc26_38 as %Eats.type.loc26_49.1 {
// CHECK:STDOUT:     %Eats.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:     %Eats.impl_witness.loc26_51.1: <witness> = impl_witness %Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(constants.%T.bfe, constants.%U) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.cce0fb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Eats.impl_witness.loc26_51.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Animal.impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Animal.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Grass {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Grass
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%Food.loc31_9.2: %Edible.type, %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.133)) {
// CHECK:STDOUT:   %Food.loc31_9.1: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.4a8)]
// CHECK:STDOUT:   %Food.binding.as_type: type = symbolic_binding_type Food, 0, %Food.loc31_9.1 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.d04)]
// CHECK:STDOUT:   %Eats.type.loc31_37.1: type = facet_type <@Eats, @Eats(%Food.binding.as_type)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.133)]
// CHECK:STDOUT:   %T.loc31_24.1: @Feed.%Eats.type.loc31_37.1 (%Eats.type.133) = symbolic_binding T, 1 [symbolic = %T.loc31_24.1 (constants.%T.e62)]
// CHECK:STDOUT:   %pattern_type.loc31_24: type = pattern_type %Eats.type.loc31_37.1 [symbolic = %pattern_type.loc31_24 (constants.%pattern_type.016)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 1, %T.loc31_24.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.a55)]
// CHECK:STDOUT:   %pattern_type.loc31_40: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc31_40 (constants.%pattern_type.6fa)]
// CHECK:STDOUT:   %pattern_type.loc31_46: type = pattern_type %Food.binding.as_type [symbolic = %pattern_type.loc31_46 (constants.%pattern_type.3b5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc31_41 (constants.%require_complete.8cc)]
// CHECK:STDOUT:   %require_complete.loc31_50: <witness> = require_complete_type %Food.binding.as_type [symbolic = %require_complete.loc31_50 (constants.%require_complete.13c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.binding.as_type (%T.binding.as_type.a55), %food.param: @Feed.%Food.binding.as_type (%Food.binding.as_type.d04)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%A.loc32_17.2: %Animal.type, %Food.loc32_29.2: %Edible.type) {
// CHECK:STDOUT:   %A.loc32_17.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:   %Food.loc32_29.1: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.a7e)]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A.loc32_17.1 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc32_44: type = pattern_type %A.binding.as_type [symbolic = %pattern_type.loc32_44 (constants.%pattern_type.5aa)]
// CHECK:STDOUT:   %Food.binding.as_type: type = symbolic_binding_type Food, 1, %Food.loc32_29.1 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.cd7)]
// CHECK:STDOUT:   %pattern_type.loc32_50: type = pattern_type %Food.binding.as_type [symbolic = %pattern_type.loc32_50 (constants.%pattern_type.797)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45: <witness> = require_complete_type %A.binding.as_type [symbolic = %require_complete.loc32_45 (constants.%require_complete.1b9)]
// CHECK:STDOUT:   %require_complete.loc32_54: <witness> = require_complete_type %Food.binding.as_type [symbolic = %require_complete.loc32_54 (constants.%require_complete.9fb)]
// CHECK:STDOUT:   %.loc32_76.4: require_specific_def_type = require_specific_def @T.binding.as_type.as.Eats.impl(%A.loc32_17.1, %Food.loc32_29.1) [symbolic = %.loc32_76.4 (constants.%.c9c)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc32_17.1, @Eats, @Eats(%Food.binding.as_type) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.binding.as_type)> [symbolic = %Eats.type (constants.%Eats.type.1b7e7c.2)]
// CHECK:STDOUT:   %Eats.facet.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.1b7e7c.2) = facet_value %A.binding.as_type, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.c04)]
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc32_29.1, %Eats.facet.loc32_76.2) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.9db)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%A.binding.as_type (%A.binding.as_type), %food.param: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.cd7)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.cd7) = name_ref food, %food
// CHECK:STDOUT:     %.loc32_76.1: %Edible.type = converted constants.%Food.binding.as_type.cd7, constants.%Food.a7e [symbolic = %Food.loc32_29.1 (constants.%Food.a7e)]
// CHECK:STDOUT:     %.loc32_76.2: %Edible.type = converted constants.%Food.binding.as_type.cd7, constants.%Food.a7e [symbolic = %Food.loc32_29.1 (constants.%Food.a7e)]
// CHECK:STDOUT:     %Eats.facet.loc32_76.1: @HandleAnimal.%Eats.type (%Eats.type.1b7e7c.2) = facet_value constants.%A.binding.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.c04)]
// CHECK:STDOUT:     %.loc32_76.3: @HandleAnimal.%Eats.type (%Eats.type.1b7e7c.2) = converted constants.%A.binding.as_type, %Eats.facet.loc32_76.1 [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.c04)]
// CHECK:STDOUT:     %Feed.specific_fn.loc32_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.a7e, constants.%Eats.facet.c04) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.9db)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc32_64.1(%a.ref, %food.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc35_17.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc35_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc35_17.3: init %Goat = class_init (), %.loc35_17.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc35_17.4: ref %Goat = temporary %.loc35_17.2, %.loc35_17.3
// CHECK:STDOUT:   %.loc35_19.1: ref %Goat = converted %.loc35_17.1, %.loc35_17.4
// CHECK:STDOUT:   %.loc35_29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:   %.loc35_29.2: ref %Grass = temporary_storage
// CHECK:STDOUT:   %.loc35_29.3: init %Grass = class_init (), %.loc35_29.2 [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %.loc35_29.4: ref %Grass = temporary %.loc35_29.2, %.loc35_29.3
// CHECK:STDOUT:   %.loc35_31.1: ref %Grass = converted %.loc35_29.1, %.loc35_29.4
// CHECK:STDOUT:   %Animal.facet.loc35_39.1: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc35_39.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc35_39.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc35_39.2: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc35_39.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc35_39.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Edible.facet.loc35_39.1: %Edible.type = facet_value constants.%Grass, (constants.%Edible.impl_witness) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc35_39.3: %Edible.type = converted constants.%Grass, %Edible.facet.loc35_39.1 [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %Edible.facet.loc35_39.2: %Edible.type = facet_value constants.%Grass, (constants.%Edible.impl_witness) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc35_39.4: %Edible.type = converted constants.%Grass, %Edible.facet.loc35_39.2 [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc35_19.2: %Goat = acquire_value %.loc35_19.1
// CHECK:STDOUT:   %.loc35_31.2: %Grass = acquire_value %.loc35_31.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc35_19.2, %.loc35_31.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc35_29: <bound method> = bound_method %.loc35_29.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b72
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b72, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.d37) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.f94]
// CHECK:STDOUT:   %bound_method.loc35_29: <bound method> = bound_method %.loc35_29.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc35_29: init %empty_tuple.type = call %bound_method.loc35_29(%.loc35_29.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc35_17: <bound method> = bound_method %.loc35_17.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.8f6
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.8f6, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.cfa) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.9be]
// CHECK:STDOUT:   %bound_method.loc35_17: <bound method> = bound_method %.loc35_17.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc35_17: init %empty_tuple.type = call %bound_method.loc35_17(%.loc35_17.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.67d) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.binding.as_type) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%U.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1b7e7c.1
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.42a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%T.bfe, constants.%U) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T.bfe
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%U
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d94
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.1b7e7c.1
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.cce0fb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.binding.as_type.d04) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.binding.as_type.d04
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.133
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.84f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.4a8, constants.%T.e62) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.4a8
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.d04
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.133
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%T.e62
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.016
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.a55
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.6fa
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.3b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%A, constants.%Food.a7e) {
// CHECK:STDOUT:   %A.loc32_17.1 => constants.%A
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Food.a7e
// CHECK:STDOUT:   %A.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.5aa
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.cd7
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.797
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.binding.as_type.cd7) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.binding.as_type.cd7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%A, constants.%Food.a7e) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%A
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Food.a7e
// CHECK:STDOUT:   %T.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %U.binding.as_type => constants.%Food.binding.as_type.cd7
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.1b7e7c.2
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.cce0fb.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.52c920.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.a7e, constants.%Eats.facet.c04) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.a7e
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.cd7
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.1b7e7c.2
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.c04
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.451
// CHECK:STDOUT:   %T.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.5aa
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.797
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41 => constants.%require_complete.1b9
// CHECK:STDOUT:   %require_complete.loc31_50 => constants.%require_complete.9fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %A.loc32_17.1 => constants.%Animal.facet
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Edible.facet
// CHECK:STDOUT:   %A.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.234
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.df6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc32_54 => constants.%complete_type.357
// CHECK:STDOUT:   %.loc32_76.4 => constants.%.12a
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%Eats.impl_witness.61e
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.749
// CHECK:STDOUT:   %Eats.facet.loc32_76.2 => constants.%Eats.facet.ad3
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2 => constants.%Feed.specific_fn.177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%Animal.facet
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Edible.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %U.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.749
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.61e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.749
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.813
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.ad3) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.749
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.ad3
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.3f6
// CHECK:STDOUT:   %T.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.234
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.df6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_50 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

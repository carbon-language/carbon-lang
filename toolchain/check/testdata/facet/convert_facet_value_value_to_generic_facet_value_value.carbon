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

fn Feed[Food:! Edible, T:! Eats(Food)](unused e: T, unused food: Food) {}
fn HandleAnimal[A:! Animal, Food:! Edible](a: A, food: Food) { Feed(a, food); }

fn F() {
  HandleAnimal({} as Goat, {} as Grass);
}

// CHECK:STDOUT: --- convert_facet_value_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Edible.type: type = facet_type <@Edible> [concrete]
// CHECK:STDOUT:   %Self.461: %Edible.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Grass: type = class_type @Grass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness @Grass.as.Edible.impl.%Edible.impl_witness_table [concrete]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value %Grass, (%Edible.impl_witness) [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.c49: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Food.67d: type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Eats.type.321: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.321 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.394: type = facet_type <@Eats, @Eats(%Food.67d)> [symbolic]
// CHECK:STDOUT:   %Self.857: %Eats.type.394 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.998: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e10: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %U: %Edible.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b51: type = pattern_type %Edible.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.e4f: type = symbolic_binding_type T, 0, %T.998 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %Eats.type.bb4cf6.1: type = facet_type <@Eats, @Eats(%U.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.0150b7.1: <witness> = impl_witness @T.binding.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%T.998, %U) [symbolic]
// CHECK:STDOUT:   %Self.027: %Eats.type.bb4cf6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.a2722e.1: <witness> = require_complete_type %Eats.type.bb4cf6.1 [symbolic]
// CHECK:STDOUT:   %Eats.facet.31f: %Eats.type.bb4cf6.1 = facet_value %T.binding.as_type.e4f, (%Eats.impl_witness.0150b7.1) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Food.d1c: %Edible.type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.binding.as_type.8cc: type = symbolic_binding_type Food, 0, %Food.d1c [symbolic]
// CHECK:STDOUT:   %Eats.type.570: type = facet_type <@Eats, @Eats(%Food.binding.as_type.8cc)> [symbolic]
// CHECK:STDOUT:   %T.2b9: %Eats.type.570 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.350: type = pattern_type %Eats.type.570 [symbolic]
// CHECK:STDOUT:   %Self.a66: %Eats.type.570 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.8ae: type = symbolic_binding_type T, 1, %T.2b9 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b9: type = pattern_type %T.binding.as_type.8ae [symbolic]
// CHECK:STDOUT:   %pattern_type.12a: type = pattern_type %Food.binding.as_type.8cc [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0c1: <witness> = require_complete_type %T.binding.as_type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.bd7: <witness> = require_complete_type %Food.binding.as_type.8cc [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %Food.2fc: %Edible.type = symbolic_binding Food, 1 [symbolic]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A [symbolic]
// CHECK:STDOUT:   %pattern_type.892: type = pattern_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %Food.binding.as_type.e7e: type = symbolic_binding_type Food, 1, %Food.2fc [symbolic]
// CHECK:STDOUT:   %pattern_type.9ed: type = pattern_type %Food.binding.as_type.e7e [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.72f: <witness> = require_complete_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.0a9: <witness> = require_complete_type %Food.binding.as_type.e7e [symbolic]
// CHECK:STDOUT:   %Eats.type.bb4cf6.2: type = facet_type <@Eats, @Eats(%Food.binding.as_type.e7e)> [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.0150b7.2: <witness> = impl_witness @T.binding.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%A, %Food.2fc) [symbolic]
// CHECK:STDOUT:   %require_complete.a2722e.2: <witness> = require_complete_type %Eats.type.bb4cf6.2 [symbolic]
// CHECK:STDOUT:   %.f9e: require_specific_def_type = require_specific_def @T.binding.as_type.as.Eats.impl(%A, %Food.2fc) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A, @Eats, @Eats(%Food.binding.as_type.e7e) [symbolic]
// CHECK:STDOUT:   %Eats.facet.702: %Eats.type.bb4cf6.2 = facet_value %A.binding.as_type, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.67b: type = pattern_type %Eats.type.bb4cf6.2 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.1f9: <specific function> = specific_function %Feed, @Feed(%Food.2fc, %Eats.facet.702) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %Grass [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc35_29 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc35_17 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.8c2: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Eats.impl_witness.f54: <witness> = impl_witness @T.binding.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Self.ebd: %Eats.type.8c2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.cf8: <witness> = complete_type_witness %Eats.type.8c2 [concrete]
// CHECK:STDOUT:   %.767: require_specific_def_type = require_specific_def @T.binding.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.a4e: %Eats.type.8c2 = facet_value %Goat, (%Eats.impl_witness.f54) [concrete]
// CHECK:STDOUT:   %pattern_type.968: type = pattern_type %Eats.type.8c2 [concrete]
// CHECK:STDOUT:   %Feed.specific_fn.7dd: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.a4e) [concrete]
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
// CHECK:STDOUT:     %.loc21_23.1: type = splice_block %.loc21_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc21_16.2: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e10 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.b51 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T.998)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e4f)]
// CHECK:STDOUT:     %.loc26_38: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e4f)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.321 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc26_26.1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc26_49: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %Eats.type.loc26_49.1: type = facet_type <@Eats, @Eats(constants.%U.binding.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.bb4cf6.1)]
// CHECK:STDOUT:     %.loc26_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_14.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T.998)]
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
// CHECK:STDOUT:     %Food.patt: %pattern_type.b51 = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @Feed.%pattern_type.loc31_24 (%pattern_type.350) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type.loc31_47 (%pattern_type.2b9) = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type.loc31_47 (%pattern_type.2b9) = value_param_pattern %e.patt [concrete]
// CHECK:STDOUT:     %food.patt: @Feed.%pattern_type.loc31_60 (%pattern_type.12a) = value_binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @Feed.%pattern_type.loc31_60 (%pattern_type.12a) = value_param_pattern %food.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_16: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc31_9.2: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.d1c)]
// CHECK:STDOUT:     %.loc31_37.1: type = splice_block %Eats.type.loc31_37.2 [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.570)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: %Eats.type.321 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc31_33: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.d1c)]
// CHECK:STDOUT:       %Food.as_type.loc31_37: type = facet_access_type %Food.ref.loc31_33 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.8cc)]
// CHECK:STDOUT:       %.loc31_37.2: type = converted %Food.ref.loc31_33, %Food.as_type.loc31_37 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.8cc)]
// CHECK:STDOUT:       %Eats.type.loc31_37.2: type = facet_type <@Eats, @Eats(constants.%Food.binding.as_type.8cc)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.570)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.570) = symbolic_binding T, 1 [symbolic = %T.loc31_24.1 (constants.%T.2b9)]
// CHECK:STDOUT:     %e.param: @Feed.%T.binding.as_type (%T.binding.as_type.8ae) = value_param call_param0
// CHECK:STDOUT:     %.loc31_50.1: type = splice_block %.loc31_50.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.8ae)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc31_37.1 (%Eats.type.570) = name_ref T, %T.loc31_24.2 [symbolic = %T.loc31_24.1 (constants.%T.2b9)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.8ae)]
// CHECK:STDOUT:       %.loc31_50.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.8ae)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.binding.as_type (%T.binding.as_type.8ae) = value_binding e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.binding.as_type (%Food.binding.as_type.8cc) = value_param call_param1
// CHECK:STDOUT:     %.loc31_66.1: type = splice_block %.loc31_66.2 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.8cc)] {
// CHECK:STDOUT:       %Food.ref.loc31_66: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.d1c)]
// CHECK:STDOUT:       %Food.as_type.loc31_66: type = facet_access_type %Food.ref.loc31_66 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.8cc)]
// CHECK:STDOUT:       %.loc31_66.2: type = converted %Food.ref.loc31_66, %Food.as_type.loc31_66 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.8cc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.binding.as_type (%Food.binding.as_type.8cc) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %A.patt: %pattern_type.e10 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %Food.patt: %pattern_type.b51 = symbolic_binding_pattern Food, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.892) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.892) = value_param_pattern %a.patt [concrete]
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.9ed) = value_binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.9ed) = value_param_pattern %food.patt [concrete]
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
// CHECK:STDOUT:     %Food.loc32_29.2: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.2fc)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc32_47.1: type = splice_block %.loc32_47.2 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)] {
// CHECK:STDOUT:       %A.ref: %Animal.type = name_ref A, %A.loc32_17.2 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type: type = facet_access_type %A.ref [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:       %.loc32_47.2: type = converted %A.ref, %A.as_type [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = value_binding a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.e7e) = value_param call_param1
// CHECK:STDOUT:     %.loc32_56.1: type = splice_block %.loc32_56.2 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.e7e)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc32_29.2 [symbolic = %Food.loc32_29.1 (constants.%Food.2fc)]
// CHECK:STDOUT:       %Food.as_type: type = facet_access_type %Food.ref [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.e7e)]
// CHECK:STDOUT:       %.loc32_56.2: type = converted %Food.ref, %Food.as_type [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.e7e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.e7e) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Edible {
// CHECK:STDOUT:   %Self: %Edible.type = symbolic_binding Self, 0 [symbolic = constants.%Self.461]
// CHECK:STDOUT:   %Edible.WithSelf.decl = interface_with_self_decl @Edible [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c49]
// CHECK:STDOUT:   %Animal.WithSelf.decl = interface_with_self_decl @Animal [concrete]
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
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc21_16.1)> [symbolic = %Eats.type (constants.%Eats.type.394)]
// CHECK:STDOUT:   %Self.loc21_29.2: @Eats.%Eats.type (%Eats.type.394) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.857)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc21_29.1: @Eats.%Eats.type (%Eats.type.394) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.857)]
// CHECK:STDOUT:     %Eats.WithSelf.decl = interface_with_self_decl @Eats [concrete]
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
// CHECK:STDOUT:   %T.loc26_14.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T.998)]
// CHECK:STDOUT:   %U.loc26_26.2: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc26_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e4f)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc26_26.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %Eats.type.loc26_49.2: type = facet_type <@Eats, @Eats(%U.binding.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.bb4cf6.1)]
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2: <witness> = impl_witness %Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%T.loc26_14.2, %U.loc26_26.2) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.0150b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Eats.type.loc26_49.2 [symbolic = %require_complete (constants.%require_complete.a2722e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc26_38 as %Eats.type.loc26_49.1 {
// CHECK:STDOUT:     %Eats.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:     %Eats.impl_witness.loc26_51.1: <witness> = impl_witness %Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(constants.%T.998, constants.%U) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.0150b7.1)]
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
// CHECK:STDOUT: generic fn @Feed(%Food.loc31_9.2: %Edible.type, %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.570)) {
// CHECK:STDOUT:   %Food.loc31_9.1: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.d1c)]
// CHECK:STDOUT:   %Food.binding.as_type: type = symbolic_binding_type Food, 0, %Food.loc31_9.1 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.8cc)]
// CHECK:STDOUT:   %Eats.type.loc31_37.1: type = facet_type <@Eats, @Eats(%Food.binding.as_type)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.570)]
// CHECK:STDOUT:   %T.loc31_24.1: @Feed.%Eats.type.loc31_37.1 (%Eats.type.570) = symbolic_binding T, 1 [symbolic = %T.loc31_24.1 (constants.%T.2b9)]
// CHECK:STDOUT:   %pattern_type.loc31_24: type = pattern_type %Eats.type.loc31_37.1 [symbolic = %pattern_type.loc31_24 (constants.%pattern_type.350)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 1, %T.loc31_24.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.8ae)]
// CHECK:STDOUT:   %pattern_type.loc31_47: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc31_47 (constants.%pattern_type.2b9)]
// CHECK:STDOUT:   %pattern_type.loc31_60: type = pattern_type %Food.binding.as_type [symbolic = %pattern_type.loc31_60 (constants.%pattern_type.12a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc31_48 (constants.%require_complete.0c1)]
// CHECK:STDOUT:   %require_complete.loc31_64: <witness> = require_complete_type %Food.binding.as_type [symbolic = %require_complete.loc31_64 (constants.%require_complete.bd7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.binding.as_type (%T.binding.as_type.8ae), %food.param: @Feed.%Food.binding.as_type (%Food.binding.as_type.8cc)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%A.loc32_17.2: %Animal.type, %Food.loc32_29.2: %Edible.type) {
// CHECK:STDOUT:   %A.loc32_17.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:   %Food.loc32_29.1: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.2fc)]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A.loc32_17.1 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc32_44: type = pattern_type %A.binding.as_type [symbolic = %pattern_type.loc32_44 (constants.%pattern_type.892)]
// CHECK:STDOUT:   %Food.binding.as_type: type = symbolic_binding_type Food, 1, %Food.loc32_29.1 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.e7e)]
// CHECK:STDOUT:   %pattern_type.loc32_50: type = pattern_type %Food.binding.as_type [symbolic = %pattern_type.loc32_50 (constants.%pattern_type.9ed)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45: <witness> = require_complete_type %A.binding.as_type [symbolic = %require_complete.loc32_45 (constants.%require_complete.72f)]
// CHECK:STDOUT:   %require_complete.loc32_54: <witness> = require_complete_type %Food.binding.as_type [symbolic = %require_complete.loc32_54 (constants.%require_complete.0a9)]
// CHECK:STDOUT:   %.loc32_76.3: require_specific_def_type = require_specific_def @T.binding.as_type.as.Eats.impl(%A.loc32_17.1, %Food.loc32_29.1) [symbolic = %.loc32_76.3 (constants.%.f9e)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc32_17.1, @Eats, @Eats(%Food.binding.as_type) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.binding.as_type)> [symbolic = %Eats.type (constants.%Eats.type.bb4cf6.2)]
// CHECK:STDOUT:   %Eats.facet.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.bb4cf6.2) = facet_value %A.binding.as_type, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.702)]
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc32_29.1, %Eats.facet.loc32_76.2) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.1f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%A.binding.as_type (%A.binding.as_type), %food.param: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.e7e)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.e7e) = name_ref food, %food
// CHECK:STDOUT:     %.loc32_76.1: %Edible.type = converted constants.%Food.binding.as_type.e7e, constants.%Food.2fc [symbolic = %Food.loc32_29.1 (constants.%Food.2fc)]
// CHECK:STDOUT:     %Eats.facet.loc32_76.1: @HandleAnimal.%Eats.type (%Eats.type.bb4cf6.2) = facet_value constants.%A.binding.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.702)]
// CHECK:STDOUT:     %.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.bb4cf6.2) = converted constants.%A.binding.as_type, %Eats.facet.loc32_76.1 [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.702)]
// CHECK:STDOUT:     %Feed.specific_fn.loc32_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.2fc, constants.%Eats.facet.702) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.1f9)]
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
// CHECK:STDOUT:   %.loc35_17.3: init %Goat to %.loc35_17.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc35_17.4: ref %Goat = temporary %.loc35_17.2, %.loc35_17.3
// CHECK:STDOUT:   %.loc35_19.1: ref %Goat = converted %.loc35_17.1, %.loc35_17.4
// CHECK:STDOUT:   %.loc35_29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:   %.loc35_29.2: ref %Grass = temporary_storage
// CHECK:STDOUT:   %.loc35_29.3: init %Grass to %.loc35_29.2 = class_init () [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %.loc35_29.4: ref %Grass = temporary %.loc35_29.2, %.loc35_29.3
// CHECK:STDOUT:   %.loc35_31.1: ref %Grass = converted %.loc35_29.1, %.loc35_29.4
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc35_39.1: %Animal.type = converted constants.%Goat, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value constants.%Grass, (constants.%Edible.impl_witness) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc35_39.2: %Edible.type = converted constants.%Grass, %Edible.facet [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc35_19.2: %Goat = acquire_value %.loc35_19.1
// CHECK:STDOUT:   %.loc35_31.2: %Grass = acquire_value %.loc35_31.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc35_19.2, %.loc35_31.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc35_29: <bound method> = bound_method %.loc35_29.4, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc35_29: init %empty_tuple.type = call %Destroy.Op.bound.loc35_29(%.loc35_29.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc35_17: <bound method> = bound_method %.loc35_17.4, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc35_17: init %empty_tuple.type = call %Destroy.Op.bound.loc35_17(%.loc35_17.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_29(%self.param: ref %Grass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_17(%self.param: ref %Goat) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Edible.WithSelf(constants.%Self.461) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Edible.WithSelf(constants.%Edible.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Self.c49) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.67d) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Food.67d, constants.%Self.857) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.binding.as_type) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%U.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.bb4cf6.1
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.027
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%T.998, constants.%U) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T.998
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%U
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.e4f
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.bb4cf6.1
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.0150b7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%U.binding.as_type, constants.%Self.857) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%U.binding.as_type, constants.%Eats.facet.31f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Animal.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.binding.as_type.8cc) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.binding.as_type.8cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.570
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.a66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Food.binding.as_type.8cc, constants.%Self.857) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.d1c, constants.%T.2b9) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.d1c
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.8cc
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.570
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%T.2b9
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.350
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.8ae
// CHECK:STDOUT:   %pattern_type.loc31_47 => constants.%pattern_type.2b9
// CHECK:STDOUT:   %pattern_type.loc31_60 => constants.%pattern_type.12a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%A, constants.%Food.2fc) {
// CHECK:STDOUT:   %A.loc32_17.1 => constants.%A
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Food.2fc
// CHECK:STDOUT:   %A.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.892
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.e7e
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.9ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.binding.as_type.e7e) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.binding.as_type.e7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%A, constants.%Food.2fc) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%A
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Food.2fc
// CHECK:STDOUT:   %T.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %U.binding.as_type => constants.%Food.binding.as_type.e7e
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.bb4cf6.2
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.0150b7.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a2722e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.2fc, constants.%Eats.facet.702) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.2fc
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.e7e
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.bb4cf6.2
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.702
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.67b
// CHECK:STDOUT:   %T.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc31_47 => constants.%pattern_type.892
// CHECK:STDOUT:   %pattern_type.loc31_60 => constants.%pattern_type.9ed
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48 => constants.%require_complete.72f
// CHECK:STDOUT:   %require_complete.loc31_64 => constants.%require_complete.0a9
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
// CHECK:STDOUT:   %.loc32_76.3 => constants.%.767
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%Eats.impl_witness.f54
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.8c2
// CHECK:STDOUT:   %Eats.facet.loc32_76.2 => constants.%Eats.facet.a4e
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2 => constants.%Feed.specific_fn.7dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%Animal.facet
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Edible.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %U.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.8c2
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.f54
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cf8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.8c2
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.ebd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Grass, constants.%Self.857) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.a4e) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.8c2
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.a4e
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.968
// CHECK:STDOUT:   %T.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc31_47 => constants.%pattern_type.234
// CHECK:STDOUT:   %pattern_type.loc31_60 => constants.%pattern_type.df6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_64 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Food.67d: type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Eats.type.321: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.321 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.394: type = facet_type <@Eats, @Eats(%Food.67d)> [symbolic]
// CHECK:STDOUT:   %Self.857: %Eats.type.394 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.e10: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.998: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b51: type = pattern_type %Edible.type [concrete]
// CHECK:STDOUT:   %U: %Edible.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.d11: type = facet_access_type %T.998 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %Eats.type.5eeba5.1: type = facet_type <@Eats, @Eats(%U.as_type)> [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.e8df35.1: <witness> = impl_witness @T.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.998, %U) [symbolic]
// CHECK:STDOUT:   %Self.0ed: %Eats.type.5eeba5.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.2dfc69.1: <witness> = require_complete_type %Eats.type.5eeba5.1 [symbolic]
// CHECK:STDOUT:   %Eats.facet.9e6: %Eats.type.5eeba5.1 = facet_value %T.as_type.d11, (%Eats.impl_witness.e8df35.1) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Food.d1c: %Edible.type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.as_type.098: type = facet_access_type %Food.d1c [symbolic]
// CHECK:STDOUT:   %Eats.type.3ac: type = facet_type <@Eats, @Eats(%Food.as_type.098)> [symbolic]
// CHECK:STDOUT:   %pattern_type.a18: type = pattern_type %Eats.type.3ac [symbolic]
// CHECK:STDOUT:   %T.bd3: %Eats.type.3ac = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %Self.746: %Eats.type.3ac = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.8f9: type = facet_access_type %T.bd3 [symbolic]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %T.as_type.8f9 [symbolic]
// CHECK:STDOUT:   %pattern_type.a11: type = pattern_type %Food.as_type.098 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1ba: <witness> = require_complete_type %T.as_type.8f9 [symbolic]
// CHECK:STDOUT:   %require_complete.2e4: <witness> = require_complete_type %Food.as_type.098 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %Food.2fc: %Edible.type = symbolic_binding Food, 1 [symbolic]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %pattern_type.f0b: type = pattern_type %A.as_type [symbolic]
// CHECK:STDOUT:   %Food.as_type.6c2: type = facet_access_type %Food.2fc [symbolic]
// CHECK:STDOUT:   %pattern_type.725: type = pattern_type %Food.as_type.6c2 [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.04b: <witness> = require_complete_type %A.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.a7e: <witness> = require_complete_type %Food.as_type.6c2 [symbolic]
// CHECK:STDOUT:   %Eats.type.5eeba5.2: type = facet_type <@Eats, @Eats(%Food.as_type.6c2)> [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A, @Eats, @Eats(%Food.as_type.6c2) [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.e8df35.2: <witness> = impl_witness @T.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%A, %Food.2fc) [symbolic]
// CHECK:STDOUT:   %require_complete.2dfc69.2: <witness> = require_complete_type %Eats.type.5eeba5.2 [symbolic]
// CHECK:STDOUT:   %.bc9: require_specific_def_type = require_specific_def @T.as_type.as.Eats.impl(%A, %Food.2fc) [symbolic]
// CHECK:STDOUT:   %Eats.facet.0ae: %Eats.type.5eeba5.2 = facet_value %A.as_type, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.4ed: type = pattern_type %Eats.type.5eeba5.2 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.4f5: <specific function> = specific_function %Feed, @Feed(%Food.2fc, %Eats.facet.0ae) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %Grass [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %.aec: ref %Goat = temporary invalid, %Goat.val [concrete]
// CHECK:STDOUT:   %.cf4: ref %Grass = temporary invalid, %Grass.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc35_31.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.11c: <bound method> = bound_method %.cf4, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc35_19 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.f91: <bound method> = bound_method %.aec, %Destroy.Op.651ba6.3 [concrete]
// CHECK:STDOUT:   %Eats.type.8c2: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Eats.impl_witness.d52: <witness> = impl_witness @T.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Self.ebd: %Eats.type.8c2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.cf8: <witness> = complete_type_witness %Eats.type.8c2 [concrete]
// CHECK:STDOUT:   %.db2: require_specific_def_type = require_specific_def @T.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.f22: %Eats.type.8c2 = facet_value %Goat, (%Eats.impl_witness.d52) [concrete]
// CHECK:STDOUT:   %pattern_type.968: type = pattern_type %Eats.type.8c2 [concrete]
// CHECK:STDOUT:   %Feed.specific_fn.583: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.f22) [concrete]
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
// CHECK:STDOUT:     %Food.loc21_20.2: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_20.1 (constants.%Food.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e10 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.b51 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc26_15.1 [symbolic = %T.loc26_15.2 (constants.%T.998)]
// CHECK:STDOUT:     %T.as_type.loc26_38.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.d11)]
// CHECK:STDOUT:     %.loc26_38: type = converted %T.ref, %T.as_type.loc26_38.1 [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.d11)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.321 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc26_27.1 [symbolic = %U.loc26_27.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc26_49.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc26_49: type = converted %U.ref, %U.as_type.loc26_49.1 [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %Eats.type.loc26_49.1: type = facet_type <@Eats, @Eats(constants.%U.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.5eeba5.1)]
// CHECK:STDOUT:     %.loc26_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_15.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T.998)]
// CHECK:STDOUT:     %.loc26_30: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc26_27.1: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_27.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.b51 = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @Feed.%pattern_type.loc31_25 (%pattern_type.a18) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type.loc31_48 (%pattern_type.76e) = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type.loc31_48 (%pattern_type.76e) = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:     %food.param_patt: @Feed.%pattern_type.loc31_64 (%pattern_type.a11) = value_param_pattern [concrete]
// CHECK:STDOUT:     %food.patt: @Feed.%pattern_type.loc31_64 (%pattern_type.a11) = at_binding_pattern food, %food.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_16: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc31_13.2: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_13.1 (constants.%Food.d1c)]
// CHECK:STDOUT:     %.loc31_37.1: type = splice_block %Eats.type.loc31_37.2 [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.3ac)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: %Eats.type.321 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc31_33: %Edible.type = name_ref Food, %Food.loc31_13.2 [symbolic = %Food.loc31_13.1 (constants.%Food.d1c)]
// CHECK:STDOUT:       %Food.as_type.loc31_37.2: type = facet_access_type %Food.ref.loc31_33 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.098)]
// CHECK:STDOUT:       %.loc31_37.2: type = converted %Food.ref.loc31_33, %Food.as_type.loc31_37.2 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.098)]
// CHECK:STDOUT:       %Eats.type.loc31_37.2: type = facet_type <@Eats, @Eats(constants.%Food.as_type.098)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.3ac)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_25.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3ac) = symbolic_binding T, 1 [symbolic = %T.loc31_25.1 (constants.%T.bd3)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc31_50.1 (%T.as_type.8f9) = value_param call_param0
// CHECK:STDOUT:     %.loc31_50.1: type = splice_block %.loc31_50.2 [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.8f9)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3ac) = name_ref T, %T.loc31_25.2 [symbolic = %T.loc31_25.1 (constants.%T.bd3)]
// CHECK:STDOUT:       %T.as_type.loc31_50.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.8f9)]
// CHECK:STDOUT:       %.loc31_50.2: type = converted %T.ref, %T.as_type.loc31_50.2 [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.8f9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc31_50.1 (%T.as_type.8f9) = value_binding e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.098) = value_param call_param1
// CHECK:STDOUT:     %.loc31_66.1: type = splice_block %.loc31_66.2 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.098)] {
// CHECK:STDOUT:       %Food.ref.loc31_66: %Edible.type = name_ref Food, %Food.loc31_13.2 [symbolic = %Food.loc31_13.1 (constants.%Food.d1c)]
// CHECK:STDOUT:       %Food.as_type.loc31_66: type = facet_access_type %Food.ref.loc31_66 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.098)]
// CHECK:STDOUT:       %.loc31_66.2: type = converted %Food.ref.loc31_66, %Food.as_type.loc31_66 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.098)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.098) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %A.patt: %pattern_type.e10 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %Food.patt: %pattern_type.b51 = symbolic_binding_pattern Food, 1 [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type.loc32_45 (%pattern_type.f0b) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type.loc32_45 (%pattern_type.f0b) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%pattern_type.loc32_54 (%pattern_type.725) = value_param_pattern [concrete]
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%pattern_type.loc32_54 (%pattern_type.725) = at_binding_pattern food, %food.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc32_18.2: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_18.1 (constants.%A)]
// CHECK:STDOUT:     %.loc32_36: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc32_33.2: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_33.1 (constants.%Food.2fc)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc32_47.1: type = splice_block %.loc32_47.2 [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)] {
// CHECK:STDOUT:       %A.ref: %Animal.type = name_ref A, %A.loc32_18.2 [symbolic = %A.loc32_18.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type.loc32_47.2: type = facet_access_type %A.ref [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)]
// CHECK:STDOUT:       %.loc32_47.2: type = converted %A.ref, %A.as_type.loc32_47.2 [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type) = value_binding a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.6c2) = value_param call_param1
// CHECK:STDOUT:     %.loc32_56.1: type = splice_block %.loc32_56.2 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.6c2)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc32_33.2 [symbolic = %Food.loc32_33.1 (constants.%Food.2fc)]
// CHECK:STDOUT:       %Food.as_type.loc32_56.2: type = facet_access_type %Food.ref [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.6c2)]
// CHECK:STDOUT:       %.loc32_56.2: type = converted %Food.ref, %Food.as_type.loc32_56.2 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.6c2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.6c2) = value_binding food, %food.param
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
// CHECK:STDOUT: generic interface @Eats(%Food.loc21_20.2: type) {
// CHECK:STDOUT:   %Food.loc21_20.1: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_20.1 (constants.%Food.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc21_20.1)> [symbolic = %Eats.type (constants.%Eats.type.394)]
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
// CHECK:STDOUT: generic impl @T.as_type.as.Eats.impl(%T.loc26_15.1: %Animal.type, %U.loc26_27.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc26_15.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T.998)]
// CHECK:STDOUT:   %U.loc26_27.2: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_27.2 (constants.%U)]
// CHECK:STDOUT:   %T.as_type.loc26_38.2: type = facet_access_type %T.loc26_15.2 [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.d11)]
// CHECK:STDOUT:   %U.as_type.loc26_49.2: type = facet_access_type %U.loc26_27.2 [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %Eats.type.loc26_49.2: type = facet_type <@Eats, @Eats(%U.as_type.loc26_49.2)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.5eeba5.1)]
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2: <witness> = impl_witness %Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.loc26_15.2, %U.loc26_27.2) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.e8df35.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Eats.type.loc26_49.2 [symbolic = %require_complete (constants.%require_complete.2dfc69.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc26_38 as %Eats.type.loc26_49.1 {
// CHECK:STDOUT:     %Eats.impl_witness_table = impl_witness_table (), @T.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:     %Eats.impl_witness.loc26_51.1: <witness> = impl_witness %Eats.impl_witness_table, @T.as_type.as.Eats.impl(constants.%T.998, constants.%U) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.e8df35.1)]
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
// CHECK:STDOUT: generic fn @Feed(%Food.loc31_13.2: %Edible.type, %T.loc31_25.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3ac)) {
// CHECK:STDOUT:   %Food.loc31_13.1: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_13.1 (constants.%Food.d1c)]
// CHECK:STDOUT:   %Food.as_type.loc31_37.1: type = facet_access_type %Food.loc31_13.1 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.098)]
// CHECK:STDOUT:   %Eats.type.loc31_37.1: type = facet_type <@Eats, @Eats(%Food.as_type.loc31_37.1)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.3ac)]
// CHECK:STDOUT:   %T.loc31_25.1: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3ac) = symbolic_binding T, 1 [symbolic = %T.loc31_25.1 (constants.%T.bd3)]
// CHECK:STDOUT:   %pattern_type.loc31_25: type = pattern_type %Eats.type.loc31_37.1 [symbolic = %pattern_type.loc31_25 (constants.%pattern_type.a18)]
// CHECK:STDOUT:   %T.as_type.loc31_50.1: type = facet_access_type %T.loc31_25.1 [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.8f9)]
// CHECK:STDOUT:   %pattern_type.loc31_48: type = pattern_type %T.as_type.loc31_50.1 [symbolic = %pattern_type.loc31_48 (constants.%pattern_type.76e)]
// CHECK:STDOUT:   %pattern_type.loc31_64: type = pattern_type %Food.as_type.loc31_37.1 [symbolic = %pattern_type.loc31_64 (constants.%pattern_type.a11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48: <witness> = require_complete_type %T.as_type.loc31_50.1 [symbolic = %require_complete.loc31_48 (constants.%require_complete.1ba)]
// CHECK:STDOUT:   %require_complete.loc31_64: <witness> = require_complete_type %Food.as_type.loc31_37.1 [symbolic = %require_complete.loc31_64 (constants.%require_complete.2e4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc31_50.1 (%T.as_type.8f9), %food.param: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.098)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%A.loc32_18.2: %Animal.type, %Food.loc32_33.2: %Edible.type) {
// CHECK:STDOUT:   %A.loc32_18.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_18.1 (constants.%A)]
// CHECK:STDOUT:   %Food.loc32_33.1: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_33.1 (constants.%Food.2fc)]
// CHECK:STDOUT:   %A.as_type.loc32_47.1: type = facet_access_type %A.loc32_18.1 [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)]
// CHECK:STDOUT:   %pattern_type.loc32_45: type = pattern_type %A.as_type.loc32_47.1 [symbolic = %pattern_type.loc32_45 (constants.%pattern_type.f0b)]
// CHECK:STDOUT:   %Food.as_type.loc32_56.1: type = facet_access_type %Food.loc32_33.1 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.6c2)]
// CHECK:STDOUT:   %pattern_type.loc32_54: type = pattern_type %Food.as_type.loc32_56.1 [symbolic = %pattern_type.loc32_54 (constants.%pattern_type.725)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45: <witness> = require_complete_type %A.as_type.loc32_47.1 [symbolic = %require_complete.loc32_45 (constants.%require_complete.04b)]
// CHECK:STDOUT:   %require_complete.loc32_54: <witness> = require_complete_type %Food.as_type.loc32_56.1 [symbolic = %require_complete.loc32_54 (constants.%require_complete.a7e)]
// CHECK:STDOUT:   %.loc32_76.3: require_specific_def_type = require_specific_def @T.as_type.as.Eats.impl(%A.loc32_18.1, %Food.loc32_33.1) [symbolic = %.loc32_76.3 (constants.%.bc9)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.as_type.loc32_56.1)> [symbolic = %Eats.type (constants.%Eats.type.5eeba5.2)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc32_18.1, @Eats, @Eats(%Food.as_type.loc32_56.1) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.5eeba5.2) = facet_value %A.as_type.loc32_47.1, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.0ae)]
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc32_33.1, %Eats.facet.loc32_76.2) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.4f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type), %food.param: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.6c2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.6c2) = name_ref food, %food
// CHECK:STDOUT:     %.loc32_76.1: %Edible.type = converted constants.%Food.as_type.6c2, constants.%Food.2fc [symbolic = %Food.loc32_33.1 (constants.%Food.2fc)]
// CHECK:STDOUT:     %Eats.facet.loc32_76.1: @HandleAnimal.%Eats.type (%Eats.type.5eeba5.2) = facet_value constants.%A.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.0ae)]
// CHECK:STDOUT:     %.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.5eeba5.2) = converted constants.%A.as_type, %Eats.facet.loc32_76.1 [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.0ae)]
// CHECK:STDOUT:     %Feed.specific_fn.loc32_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.2fc, constants.%Eats.facet.0ae) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.4f5)]
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
// CHECK:STDOUT:   %.loc35_19.1: init %Goat = converted %.loc35_17.1, %.loc35_17.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc35_29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:   %.loc35_29.2: ref %Grass = temporary_storage
// CHECK:STDOUT:   %.loc35_29.3: init %Grass to %.loc35_29.2 = class_init () [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %.loc35_31.1: init %Grass = converted %.loc35_29.1, %.loc35_29.3 [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc35_39.1: %Animal.type = converted constants.%Goat, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value constants.%Grass, (constants.%Edible.impl_witness) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc35_39.2: %Edible.type = converted constants.%Grass, %Edible.facet [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc35_19.2: ref %Goat = temporary %.loc35_17.2, %.loc35_19.1 [concrete = constants.%.aec]
// CHECK:STDOUT:   %.loc35_19.3: %Goat = acquire_value %.loc35_19.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc35_31.2: ref %Grass = temporary %.loc35_29.2, %.loc35_31.1 [concrete = constants.%.cf4]
// CHECK:STDOUT:   %.loc35_31.3: %Grass = acquire_value %.loc35_31.2 [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc35_19.3, %.loc35_31.3)
// CHECK:STDOUT:   %Destroy.Op.call.loc35_31: init %empty_tuple.type = call constants.%Destroy.Op.bound.11c(constants.%.cf4)
// CHECK:STDOUT:   %Destroy.Op.call.loc35_19: init %empty_tuple.type = call constants.%Destroy.Op.bound.f91(constants.%.aec)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_31.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_31.2(%self.param: ref %Grass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_19(%self.param: ref %Goat) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Food.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Food.67d, constants.%Self.857) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.as_type) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.5eeba5.1
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.0ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%T.998, constants.%U) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%T.998
// CHECK:STDOUT:   %U.loc26_27.2 => constants.%U
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%T.as_type.d11
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%U.as_type
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.5eeba5.1
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.e8df35.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%U.as_type, constants.%Self.857) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%U.as_type, constants.%Eats.facet.9e6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Animal.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.098) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Food.as_type.098
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.3ac
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.746
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Food.as_type.098, constants.%Self.857) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.d1c, constants.%T.bd3) {
// CHECK:STDOUT:   %Food.loc31_13.1 => constants.%Food.d1c
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Food.as_type.098
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.3ac
// CHECK:STDOUT:   %T.loc31_25.1 => constants.%T.bd3
// CHECK:STDOUT:   %pattern_type.loc31_25 => constants.%pattern_type.a18
// CHECK:STDOUT:   %T.as_type.loc31_50.1 => constants.%T.as_type.8f9
// CHECK:STDOUT:   %pattern_type.loc31_48 => constants.%pattern_type.76e
// CHECK:STDOUT:   %pattern_type.loc31_64 => constants.%pattern_type.a11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%A, constants.%Food.2fc) {
// CHECK:STDOUT:   %A.loc32_18.1 => constants.%A
// CHECK:STDOUT:   %Food.loc32_33.1 => constants.%Food.2fc
// CHECK:STDOUT:   %A.as_type.loc32_47.1 => constants.%A.as_type
// CHECK:STDOUT:   %pattern_type.loc32_45 => constants.%pattern_type.f0b
// CHECK:STDOUT:   %Food.as_type.loc32_56.1 => constants.%Food.as_type.6c2
// CHECK:STDOUT:   %pattern_type.loc32_54 => constants.%pattern_type.725
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.6c2) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Food.as_type.6c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%A, constants.%Food.2fc) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%A
// CHECK:STDOUT:   %U.loc26_27.2 => constants.%Food.2fc
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%A.as_type
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%Food.as_type.6c2
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.5eeba5.2
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.e8df35.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2dfc69.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.2fc, constants.%Eats.facet.0ae) {
// CHECK:STDOUT:   %Food.loc31_13.1 => constants.%Food.2fc
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Food.as_type.6c2
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.5eeba5.2
// CHECK:STDOUT:   %T.loc31_25.1 => constants.%Eats.facet.0ae
// CHECK:STDOUT:   %pattern_type.loc31_25 => constants.%pattern_type.4ed
// CHECK:STDOUT:   %T.as_type.loc31_50.1 => constants.%A.as_type
// CHECK:STDOUT:   %pattern_type.loc31_48 => constants.%pattern_type.f0b
// CHECK:STDOUT:   %pattern_type.loc31_64 => constants.%pattern_type.725
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48 => constants.%require_complete.04b
// CHECK:STDOUT:   %require_complete.loc31_64 => constants.%require_complete.a7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %A.loc32_18.1 => constants.%Animal.facet
// CHECK:STDOUT:   %Food.loc32_33.1 => constants.%Edible.facet
// CHECK:STDOUT:   %A.as_type.loc32_47.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc32_45 => constants.%pattern_type.234
// CHECK:STDOUT:   %Food.as_type.loc32_56.1 => constants.%Grass
// CHECK:STDOUT:   %pattern_type.loc32_54 => constants.%pattern_type.df6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc32_54 => constants.%complete_type.357
// CHECK:STDOUT:   %.loc32_76.3 => constants.%.db2
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.8c2
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%Eats.impl_witness.d52
// CHECK:STDOUT:   %Eats.facet.loc32_76.2 => constants.%Eats.facet.f22
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2 => constants.%Feed.specific_fn.583
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %U.loc26_27.2 => constants.%Edible.facet
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%Goat
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.8c2
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.d52
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cf8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Grass
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
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.f22) {
// CHECK:STDOUT:   %Food.loc31_13.1 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.8c2
// CHECK:STDOUT:   %T.loc31_25.1 => constants.%Eats.facet.f22
// CHECK:STDOUT:   %pattern_type.loc31_25 => constants.%pattern_type.968
// CHECK:STDOUT:   %T.as_type.loc31_50.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc31_48 => constants.%pattern_type.234
// CHECK:STDOUT:   %pattern_type.loc31_64 => constants.%pattern_type.df6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_64 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

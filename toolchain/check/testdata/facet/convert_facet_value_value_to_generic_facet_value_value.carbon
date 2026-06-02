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
// CHECK:STDOUT:   %Self.33f: %Edible.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Grass: type = class_type @Grass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness @Grass.as.Edible.impl.%Edible.impl_witness_table [concrete]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value %Grass, (%Edible.impl_witness) [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.d73: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Food.67d: type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Eats.type.131: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.131 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.707: type = facet_type <@Eats, @Eats(%Food.67d)> [symbolic]
// CHECK:STDOUT:   %Self.1ca: %Eats.type.707 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.69f: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.123: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.53d: type = pattern_type %Edible.type [concrete]
// CHECK:STDOUT:   %U: %Edible.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.6eb: type = facet_access_type %T.123 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %Eats.type.837543.1: type = facet_type <@Eats, @Eats(%U.as_type)> [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.9746ac.1: <witness> = impl_witness @T.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.123, %U) [symbolic]
// CHECK:STDOUT:   %Self.13a: %Eats.type.837543.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.8fa479.1: <witness> = require_complete_type %Eats.type.837543.1 [symbolic]
// CHECK:STDOUT:   %Eats.facet.430: %Eats.type.837543.1 = facet_value %T.as_type.6eb, (%Eats.impl_witness.9746ac.1) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Food.ffc: %Edible.type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.as_type.7d8: type = facet_access_type %Food.ffc [symbolic]
// CHECK:STDOUT:   %Eats.type.ab0: type = facet_type <@Eats, @Eats(%Food.as_type.7d8)> [symbolic]
// CHECK:STDOUT:   %pattern_type.f0e: type = pattern_type %Eats.type.ab0 [symbolic]
// CHECK:STDOUT:   %T.57c: %Eats.type.ab0 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %Self.a74: %Eats.type.ab0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.54c: type = facet_access_type %T.57c [symbolic]
// CHECK:STDOUT:   %pattern_type.20f: type = pattern_type %T.as_type.54c [symbolic]
// CHECK:STDOUT:   %pattern_type.c7d: type = pattern_type %Food.as_type.7d8 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.bb7: <witness> = require_complete_type %T.as_type.54c [symbolic]
// CHECK:STDOUT:   %require_complete.082: <witness> = require_complete_type %Food.as_type.7d8 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %Food.7a5: %Edible.type = symbolic_binding Food, 1 [symbolic]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %pattern_type.106: type = pattern_type %A.as_type [symbolic]
// CHECK:STDOUT:   %Food.as_type.a50: type = facet_access_type %Food.7a5 [symbolic]
// CHECK:STDOUT:   %pattern_type.fc4: type = pattern_type %Food.as_type.a50 [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.284: <witness> = require_complete_type %A.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.2a2: <witness> = require_complete_type %Food.as_type.a50 [symbolic]
// CHECK:STDOUT:   %Eats.type.837543.2: type = facet_type <@Eats, @Eats(%Food.as_type.a50)> [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A, @Eats, @Eats(%Food.as_type.a50) [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.9746ac.2: <witness> = impl_witness @T.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%A, %Food.7a5) [symbolic]
// CHECK:STDOUT:   %require_complete.8fa479.2: <witness> = require_complete_type %Eats.type.837543.2 [symbolic]
// CHECK:STDOUT:   %.bdc: require_specific_def_type = require_specific_def @T.as_type.as.Eats.impl(%A, %Food.7a5) [symbolic]
// CHECK:STDOUT:   %Eats.facet.f1f: %Eats.type.837543.2 = facet_value %A.as_type, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.ba3: type = pattern_type %Eats.type.837543.2 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.1f3: <specific function> = specific_function %Feed, @Feed(%Food.7a5, %Eats.facet.f1f) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.332: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.413: type = pattern_type %Grass [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %.385: ref %Goat = temporary invalid, %Goat.val [concrete]
// CHECK:STDOUT:   %.79b: ref %Grass = temporary invalid, %Grass.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc35_31.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.4fe: <bound method> = bound_method %.79b, %Destroy.Op.1dc86d.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc35_19 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.ce0: <bound method> = bound_method %.385, %Destroy.Op.1dc86d.3 [concrete]
// CHECK:STDOUT:   %Eats.type.04c: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Eats.impl_witness.931: <witness> = impl_witness @T.as_type.as.Eats.impl.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Self.c37: %Eats.type.04c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.5c0: <witness> = complete_type_witness %Eats.type.04c [concrete]
// CHECK:STDOUT:   %.ed3: require_specific_def_type = require_specific_def @T.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.a1f: %Eats.type.04c = facet_value %Goat, (%Eats.impl_witness.931) [concrete]
// CHECK:STDOUT:   %pattern_type.593: type = pattern_type %Eats.type.04c [concrete]
// CHECK:STDOUT:   %Feed.specific_fn.b2f: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.a1f) [concrete]
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
// CHECK:STDOUT:   %Eats.decl: %Eats.type.131 = interface_decl @Eats [concrete = constants.%Eats.generic] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.98f = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_23.1: type = splice_block %.loc21_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc21_20.2: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_20.1 (constants.%Food.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.69f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.53d = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc26_15.1 [symbolic = %T.loc26_15.2 (constants.%T.123)]
// CHECK:STDOUT:     %T.as_type.loc26_38.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.6eb)]
// CHECK:STDOUT:     %.loc26_38: type = converted %T.ref, %T.as_type.loc26_38.1 [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.6eb)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.131 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc26_27.1 [symbolic = %U.loc26_27.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc26_49.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc26_49: type = converted %U.ref, %U.as_type.loc26_49.1 [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %Eats.type.loc26_49.1: type = facet_type <@Eats, @Eats(constants.%U.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.837543.1)]
// CHECK:STDOUT:     %.loc26_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.loc26_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_15.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T.123)]
// CHECK:STDOUT:     %.loc26_30: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.loc26_27: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
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
// CHECK:STDOUT:     %Food.patt: %pattern_type.53d = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @Feed.%pattern_type.loc31_25 (%pattern_type.f0e) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type.loc31_48 (%pattern_type.20f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type.loc31_48 (%pattern_type.20f) = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:     %food.param_patt: @Feed.%pattern_type.loc31_64 (%pattern_type.c7d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %food.patt: @Feed.%pattern_type.loc31_64 (%pattern_type.c7d) = at_binding_pattern food, %food.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_16: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.loc31_13: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc31_13.2: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_13.1 (constants.%Food.ffc)]
// CHECK:STDOUT:     %.loc31_37.1: type = splice_block %Eats.type.loc31_37.2 [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.ab0)] {
// CHECK:STDOUT:       %.Self.loc31_25: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: %Eats.type.131 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc31_33: %Edible.type = name_ref Food, %Food.loc31_13.2 [symbolic = %Food.loc31_13.1 (constants.%Food.ffc)]
// CHECK:STDOUT:       %Food.as_type.loc31_37.2: type = facet_access_type %Food.ref.loc31_33 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.7d8)]
// CHECK:STDOUT:       %.loc31_37.2: type = converted %Food.ref.loc31_33, %Food.as_type.loc31_37.2 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.7d8)]
// CHECK:STDOUT:       %Eats.type.loc31_37.2: type = facet_type <@Eats, @Eats(constants.%Food.as_type.7d8)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.ab0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_25.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.ab0) = symbolic_binding T, 1 [symbolic = %T.loc31_25.1 (constants.%T.57c)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc31_50.1 (%T.as_type.54c) = value_param call_param0
// CHECK:STDOUT:     %.loc31_50.1: type = splice_block %.loc31_50.2 [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.54c)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc31_37.1 (%Eats.type.ab0) = name_ref T, %T.loc31_25.2 [symbolic = %T.loc31_25.1 (constants.%T.57c)]
// CHECK:STDOUT:       %T.as_type.loc31_50.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.54c)]
// CHECK:STDOUT:       %.loc31_50.2: type = converted %T.ref, %T.as_type.loc31_50.2 [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.54c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc31_50.1 (%T.as_type.54c) = value_binding e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.7d8) = value_param call_param1
// CHECK:STDOUT:     %.loc31_66.1: type = splice_block %.loc31_66.2 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.7d8)] {
// CHECK:STDOUT:       %Food.ref.loc31_66: %Edible.type = name_ref Food, %Food.loc31_13.2 [symbolic = %Food.loc31_13.1 (constants.%Food.ffc)]
// CHECK:STDOUT:       %Food.as_type.loc31_66: type = facet_access_type %Food.ref.loc31_66 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.7d8)]
// CHECK:STDOUT:       %.loc31_66.2: type = converted %Food.ref.loc31_66, %Food.as_type.loc31_66 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.7d8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.7d8) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %A.patt: %pattern_type.69f = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %Food.patt: %pattern_type.53d = symbolic_binding_pattern Food, 1 [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type.loc32_45 (%pattern_type.106) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type.loc32_45 (%pattern_type.106) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%pattern_type.loc32_54 (%pattern_type.fc4) = value_param_pattern [concrete]
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%pattern_type.loc32_54 (%pattern_type.fc4) = at_binding_pattern food, %food.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.loc32_18: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc32_18.2: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_18.1 (constants.%A)]
// CHECK:STDOUT:     %.loc32_36: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.loc32_33: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc32_33.2: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_33.1 (constants.%Food.7a5)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc32_47.1: type = splice_block %.loc32_47.2 [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)] {
// CHECK:STDOUT:       %A.ref: %Animal.type = name_ref A, %A.loc32_18.2 [symbolic = %A.loc32_18.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type.loc32_47.2: type = facet_access_type %A.ref [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)]
// CHECK:STDOUT:       %.loc32_47.2: type = converted %A.ref, %A.as_type.loc32_47.2 [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type) = value_binding a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.a50) = value_param call_param1
// CHECK:STDOUT:     %.loc32_56.1: type = splice_block %.loc32_56.2 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.a50)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc32_33.2 [symbolic = %Food.loc32_33.1 (constants.%Food.7a5)]
// CHECK:STDOUT:       %Food.as_type.loc32_56.2: type = facet_access_type %Food.ref [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.a50)]
// CHECK:STDOUT:       %.loc32_56.2: type = converted %Food.ref, %Food.as_type.loc32_56.2 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.a50)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.a50) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Edible {
// CHECK:STDOUT:   %Self: %Edible.type = symbolic_binding Self, 0 [symbolic = constants.%Self.33f]
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
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d73]
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
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc21_20.1)> [symbolic = %Eats.type (constants.%Eats.type.707)]
// CHECK:STDOUT:   %Self.loc21_29.2: @Eats.%Eats.type (%Eats.type.707) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.1ca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc21_29.1: @Eats.%Eats.type (%Eats.type.707) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.1ca)]
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
// CHECK:STDOUT:   extend %Edible.ref
// CHECK:STDOUT:   witness = %Edible.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.Eats.impl(%T.loc26_15.1: %Animal.type, %U.loc26_27.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc26_15.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T.123)]
// CHECK:STDOUT:   %U.loc26_27.2: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_27.2 (constants.%U)]
// CHECK:STDOUT:   %T.as_type.loc26_38.2: type = facet_access_type %T.loc26_15.2 [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.6eb)]
// CHECK:STDOUT:   %U.as_type.loc26_49.2: type = facet_access_type %U.loc26_27.2 [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %Eats.type.loc26_49.2: type = facet_type <@Eats, @Eats(%U.as_type.loc26_49.2)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.837543.1)]
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2: <witness> = impl_witness %Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.loc26_15.2, %U.loc26_27.2) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.9746ac.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Eats.type.loc26_49.2 [symbolic = %require_complete (constants.%require_complete.8fa479.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc26_38 as %Eats.type.loc26_49.1 {
// CHECK:STDOUT:     %Eats.impl_witness_table = impl_witness_table (), @T.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:     %Eats.impl_witness.loc26_51.1: <witness> = impl_witness %Eats.impl_witness_table, @T.as_type.as.Eats.impl(constants.%T.123, constants.%U) [symbolic = %Eats.impl_witness.loc26_51.2 (constants.%Eats.impl_witness.9746ac.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Eats.type.loc26_49.1
// CHECK:STDOUT:     witness = %Eats.impl_witness.loc26_51.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Animal.impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Animal.ref
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
// CHECK:STDOUT: generic fn @Feed(%Food.loc31_13.2: %Edible.type, %T.loc31_25.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.ab0)) {
// CHECK:STDOUT:   %Food.loc31_13.1: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_13.1 (constants.%Food.ffc)]
// CHECK:STDOUT:   %Food.as_type.loc31_37.1: type = facet_access_type %Food.loc31_13.1 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.7d8)]
// CHECK:STDOUT:   %Eats.type.loc31_37.1: type = facet_type <@Eats, @Eats(%Food.as_type.loc31_37.1)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.ab0)]
// CHECK:STDOUT:   %T.loc31_25.1: @Feed.%Eats.type.loc31_37.1 (%Eats.type.ab0) = symbolic_binding T, 1 [symbolic = %T.loc31_25.1 (constants.%T.57c)]
// CHECK:STDOUT:   %pattern_type.loc31_25: type = pattern_type %Eats.type.loc31_37.1 [symbolic = %pattern_type.loc31_25 (constants.%pattern_type.f0e)]
// CHECK:STDOUT:   %T.as_type.loc31_50.1: type = facet_access_type %T.loc31_25.1 [symbolic = %T.as_type.loc31_50.1 (constants.%T.as_type.54c)]
// CHECK:STDOUT:   %pattern_type.loc31_48: type = pattern_type %T.as_type.loc31_50.1 [symbolic = %pattern_type.loc31_48 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %pattern_type.loc31_64: type = pattern_type %Food.as_type.loc31_37.1 [symbolic = %pattern_type.loc31_64 (constants.%pattern_type.c7d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48: <witness> = require_complete_type %T.as_type.loc31_50.1 [symbolic = %require_complete.loc31_48 (constants.%require_complete.bb7)]
// CHECK:STDOUT:   %require_complete.loc31_64: <witness> = require_complete_type %Food.as_type.loc31_37.1 [symbolic = %require_complete.loc31_64 (constants.%require_complete.082)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc31_50.1 (%T.as_type.54c), %food.param: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.7d8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%A.loc32_18.2: %Animal.type, %Food.loc32_33.2: %Edible.type) {
// CHECK:STDOUT:   %A.loc32_18.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_18.1 (constants.%A)]
// CHECK:STDOUT:   %Food.loc32_33.1: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_33.1 (constants.%Food.7a5)]
// CHECK:STDOUT:   %A.as_type.loc32_47.1: type = facet_access_type %A.loc32_18.1 [symbolic = %A.as_type.loc32_47.1 (constants.%A.as_type)]
// CHECK:STDOUT:   %pattern_type.loc32_45: type = pattern_type %A.as_type.loc32_47.1 [symbolic = %pattern_type.loc32_45 (constants.%pattern_type.106)]
// CHECK:STDOUT:   %Food.as_type.loc32_56.1: type = facet_access_type %Food.loc32_33.1 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.a50)]
// CHECK:STDOUT:   %pattern_type.loc32_54: type = pattern_type %Food.as_type.loc32_56.1 [symbolic = %pattern_type.loc32_54 (constants.%pattern_type.fc4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45: <witness> = require_complete_type %A.as_type.loc32_47.1 [symbolic = %require_complete.loc32_45 (constants.%require_complete.284)]
// CHECK:STDOUT:   %require_complete.loc32_54: <witness> = require_complete_type %Food.as_type.loc32_56.1 [symbolic = %require_complete.loc32_54 (constants.%require_complete.2a2)]
// CHECK:STDOUT:   %.loc32_76.3: require_specific_def_type = require_specific_def @T.as_type.as.Eats.impl(%A.loc32_18.1, %Food.loc32_33.1) [symbolic = %.loc32_76.3 (constants.%.bdc)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.as_type.loc32_56.1)> [symbolic = %Eats.type (constants.%Eats.type.837543.2)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc32_18.1, @Eats, @Eats(%Food.as_type.loc32_56.1) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.837543.2) = facet_value %A.as_type.loc32_47.1, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.f1f)]
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc32_33.1, %Eats.facet.loc32_76.2) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.1f3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type), %food.param: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.a50)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%A.as_type.loc32_47.1 (%A.as_type) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.a50) = name_ref food, %food
// CHECK:STDOUT:     %.loc32_76.1: %Edible.type = converted constants.%Food.as_type.a50, constants.%Food.7a5 [symbolic = %Food.loc32_33.1 (constants.%Food.7a5)]
// CHECK:STDOUT:     %Eats.facet.loc32_76.1: @HandleAnimal.%Eats.type (%Eats.type.837543.2) = facet_value constants.%A.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.f1f)]
// CHECK:STDOUT:     %.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.837543.2) = converted constants.%A.as_type, %Eats.facet.loc32_76.1 [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.f1f)]
// CHECK:STDOUT:     %Feed.specific_fn.loc32_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.7a5, constants.%Eats.facet.f1f) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.1f3)]
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
// CHECK:STDOUT:   %.loc35_19.2: ref %Goat = temporary %.loc35_17.2, %.loc35_19.1 [concrete = constants.%.385]
// CHECK:STDOUT:   %.loc35_19.3: %Goat = acquire_value %.loc35_19.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc35_31.2: ref %Grass = temporary %.loc35_29.2, %.loc35_31.1 [concrete = constants.%.79b]
// CHECK:STDOUT:   %.loc35_31.3: %Grass = acquire_value %.loc35_31.2 [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc35_19.3, %.loc35_31.3)
// CHECK:STDOUT:   %Destroy.Op.call.loc35_31: init %empty_tuple.type = call constants.%Destroy.Op.bound.4fe(constants.%.79b)
// CHECK:STDOUT:   %Destroy.Op.call.loc35_19: init %empty_tuple.type = call constants.%Destroy.Op.bound.ce0(constants.%.385)
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
// CHECK:STDOUT: specific @Edible.WithSelf(constants.%Self.33f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Edible.WithSelf(constants.%Edible.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Self.d73) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.67d) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Food.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Food.67d, constants.%Self.1ca) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.as_type) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.837543.1
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.13a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%T.123, constants.%U) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%T.123
// CHECK:STDOUT:   %U.loc26_27.2 => constants.%U
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%T.as_type.6eb
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%U.as_type
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.837543.1
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.9746ac.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%U.as_type, constants.%Self.1ca) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%U.as_type, constants.%Eats.facet.430) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Animal.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.7d8) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Food.as_type.7d8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.ab0
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.a74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Food.as_type.7d8, constants.%Self.1ca) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.ffc, constants.%T.57c) {
// CHECK:STDOUT:   %Food.loc31_13.1 => constants.%Food.ffc
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Food.as_type.7d8
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.ab0
// CHECK:STDOUT:   %T.loc31_25.1 => constants.%T.57c
// CHECK:STDOUT:   %pattern_type.loc31_25 => constants.%pattern_type.f0e
// CHECK:STDOUT:   %T.as_type.loc31_50.1 => constants.%T.as_type.54c
// CHECK:STDOUT:   %pattern_type.loc31_48 => constants.%pattern_type.20f
// CHECK:STDOUT:   %pattern_type.loc31_64 => constants.%pattern_type.c7d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%A, constants.%Food.7a5) {
// CHECK:STDOUT:   %A.loc32_18.1 => constants.%A
// CHECK:STDOUT:   %Food.loc32_33.1 => constants.%Food.7a5
// CHECK:STDOUT:   %A.as_type.loc32_47.1 => constants.%A.as_type
// CHECK:STDOUT:   %pattern_type.loc32_45 => constants.%pattern_type.106
// CHECK:STDOUT:   %Food.as_type.loc32_56.1 => constants.%Food.as_type.a50
// CHECK:STDOUT:   %pattern_type.loc32_54 => constants.%pattern_type.fc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.a50) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Food.as_type.a50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%A, constants.%Food.7a5) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%A
// CHECK:STDOUT:   %U.loc26_27.2 => constants.%Food.7a5
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%A.as_type
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%Food.as_type.a50
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.837543.2
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.9746ac.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8fa479.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.7a5, constants.%Eats.facet.f1f) {
// CHECK:STDOUT:   %Food.loc31_13.1 => constants.%Food.7a5
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Food.as_type.a50
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.837543.2
// CHECK:STDOUT:   %T.loc31_25.1 => constants.%Eats.facet.f1f
// CHECK:STDOUT:   %pattern_type.loc31_25 => constants.%pattern_type.ba3
// CHECK:STDOUT:   %T.as_type.loc31_50.1 => constants.%A.as_type
// CHECK:STDOUT:   %pattern_type.loc31_48 => constants.%pattern_type.106
// CHECK:STDOUT:   %pattern_type.loc31_64 => constants.%pattern_type.fc4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48 => constants.%require_complete.284
// CHECK:STDOUT:   %require_complete.loc31_64 => constants.%require_complete.2a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %A.loc32_18.1 => constants.%Animal.facet
// CHECK:STDOUT:   %Food.loc32_33.1 => constants.%Edible.facet
// CHECK:STDOUT:   %A.as_type.loc32_47.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc32_45 => constants.%pattern_type.332
// CHECK:STDOUT:   %Food.as_type.loc32_56.1 => constants.%Grass
// CHECK:STDOUT:   %pattern_type.loc32_54 => constants.%pattern_type.413
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc32_54 => constants.%complete_type.357
// CHECK:STDOUT:   %.loc32_76.3 => constants.%.ed3
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.04c
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%Eats.impl_witness.931
// CHECK:STDOUT:   %Eats.facet.loc32_76.2 => constants.%Eats.facet.a1f
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2 => constants.%Feed.specific_fn.b2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %U.loc26_27.2 => constants.%Edible.facet
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%Goat
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.04c
// CHECK:STDOUT:   %Eats.impl_witness.loc26_51.2 => constants.%Eats.impl_witness.931
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc21_20.1 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.04c
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.c37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Grass, constants.%Self.1ca) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.a1f) {
// CHECK:STDOUT:   %Food.loc31_13.1 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.04c
// CHECK:STDOUT:   %T.loc31_25.1 => constants.%Eats.facet.a1f
// CHECK:STDOUT:   %pattern_type.loc31_25 => constants.%pattern_type.593
// CHECK:STDOUT:   %T.as_type.loc31_50.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc31_48 => constants.%pattern_type.332
// CHECK:STDOUT:   %pattern_type.loc31_64 => constants.%pattern_type.413
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_64 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

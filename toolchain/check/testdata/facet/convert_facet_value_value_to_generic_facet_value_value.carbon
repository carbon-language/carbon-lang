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
fn HandleAnimal[T:! Animal, Food:! Edible](a: T, food: Food) { Feed(a, food); }

fn F() {
  HandleAnimal({} as Goat, {} as Grass);
}

// CHECK:STDOUT: --- convert_facet_value_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Edible.type: type = facet_type <@Edible> [concrete]
// CHECK:STDOUT:   %Self.9af: %Edible.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Grass: type = class_type @Grass [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.826: <witness> = impl_witness @Grass.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.2bd: type = ptr_type %Grass [concrete]
// CHECK:STDOUT:   %pattern_type.4f0: type = pattern_type %ptr.2bd [concrete]
// CHECK:STDOUT:   %Grass.as.Destroy.impl.Op.type: type = fn_type @Grass.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Grass.as.Destroy.impl.Op: %Grass.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness file.%Edible.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Food.8b3: type = bind_symbolic_name Food, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Eats.type.ba2: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.6c0: type = facet_type <@Eats, @Eats(%Food.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.4eb: %Eats.type.6c0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.fd4: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b0: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %U: %Edible.type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.939: type = pattern_type %Edible.type [concrete]
// CHECK:STDOUT:   %T.as_type.2ad: type = facet_access_type %T.fd4 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %Eats.type.f54c3d.1: type = facet_type <@Eats, @Eats(%U.as_type)> [symbolic]
// CHECK:STDOUT:   %Self.c30: %Eats.type.f54c3d.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.42532a.1: <witness> = require_complete_type %Eats.type.f54c3d.1 [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.fabf92.1: <witness> = impl_witness file.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.fd4, %U) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.943: <witness> = impl_witness @Goat.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.940: type = ptr_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.396: type = pattern_type %ptr.940 [concrete]
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.type: type = fn_type @Goat.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op: %Goat.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness file.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Food.9af: %Edible.type = bind_symbolic_name Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.as_type.952: type = facet_access_type %Food.9af [symbolic]
// CHECK:STDOUT:   %Eats.type.b39: type = facet_type <@Eats, @Eats(%Food.as_type.952)> [symbolic]
// CHECK:STDOUT:   %T.223: %Eats.type.b39 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ed7: type = pattern_type %Eats.type.b39 [symbolic]
// CHECK:STDOUT:   %T.as_type.212: type = facet_access_type %T.223 [symbolic]
// CHECK:STDOUT:   %pattern_type.1a1: type = pattern_type %T.as_type.212 [symbolic]
// CHECK:STDOUT:   %pattern_type.54f: type = pattern_type %Food.as_type.952 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fe6: <witness> = require_complete_type %T.as_type.212 [symbolic]
// CHECK:STDOUT:   %require_complete.005: <witness> = require_complete_type %Food.as_type.952 [symbolic]
// CHECK:STDOUT:   %Food.5fe: %Edible.type = bind_symbolic_name Food, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.36a: type = pattern_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %Food.as_type.fae: type = facet_access_type %Food.5fe [symbolic]
// CHECK:STDOUT:   %pattern_type.f86: type = pattern_type %Food.as_type.fae [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.234: <witness> = require_complete_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %require_complete.444: <witness> = require_complete_type %Food.as_type.fae [symbolic]
// CHECK:STDOUT:   %Eats.type.f54c3d.2: type = facet_type <@Eats, @Eats(%Food.as_type.fae)> [symbolic]
// CHECK:STDOUT:   %require_complete.42532a.2: <witness> = require_complete_type %Eats.type.f54c3d.2 [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.fabf92.2: <witness> = impl_witness file.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.fd4, %Food.5fe) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.fd4, @Eats, @Eats(%Food.as_type.fae) [symbolic]
// CHECK:STDOUT:   %Eats.facet.512: %Eats.type.f54c3d.2 = facet_value %T.as_type.2ad, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.bb9: type = pattern_type %Eats.type.f54c3d.2 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.ea3: <specific function> = specific_function %Feed, @Feed(%Food.5fe, %Eats.facet.512) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value %Grass, (%Edible.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.ab7: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.aff: type = pattern_type %Grass [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.type.1ae: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Self.016: %Eats.type.1ae = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.004: <witness> = complete_type_witness %Eats.type.1ae [concrete]
// CHECK:STDOUT:   %Eats.impl_witness.1cf: <witness> = impl_witness file.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.cb1: %Eats.type.1ae = facet_value %Goat, (%Eats.impl_witness.1cf) [concrete]
// CHECK:STDOUT:   %pattern_type.d77: type = pattern_type %Eats.type.1ae [concrete]
// CHECK:STDOUT:   %Feed.specific_fn.86c: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.cb1) [concrete]
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
// CHECK:STDOUT:   %Edible.impl_witness_table = impl_witness_table (), @Grass.as.Edible.impl [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness %Edible.impl_witness_table [concrete = constants.%Edible.impl_witness]
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Eats.decl: %Eats.type.ba2 = interface_decl @Eats [concrete = constants.%Eats.generic] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.98f = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Food.loc21_16.2: type = bind_symbolic_name Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.939 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %T.as_type.loc26_38.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc26_38: type = converted %T.ref, %T.as_type.loc26_38.1 [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc26_26.1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc26_49.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc26_49: type = converted %U.ref, %U.as_type.loc26_49.1 [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %Eats.type.loc26_49.1: type = facet_type <@Eats, @Eats(constants.%U.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.f54c3d.1)]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc26_14.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     %U.loc26_26.1: %Edible.type = bind_symbolic_name U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (), @T.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table, @T.as_type.as.Eats.impl(constants.%T.fd4, constants.%U) [symbolic = @T.as_type.as.Eats.impl.%Eats.impl_witness (constants.%Eats.impl_witness.fabf92.1)]
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.939 = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @Feed.%pattern_type.loc31_24 (%pattern_type.ed7) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type.loc31_40 (%pattern_type.1a1) = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type.loc31_40 (%pattern_type.1a1) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:     %food.patt: @Feed.%pattern_type.loc31_46 (%pattern_type.54f) = binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @Feed.%pattern_type.loc31_46 (%pattern_type.54f) = value_param_pattern %food.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     %Food.loc31_9.2: %Edible.type = bind_symbolic_name Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.9af)]
// CHECK:STDOUT:     %.loc31_37.1: type = splice_block %Eats.type.loc31_37.2 [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.b39)] {
// CHECK:STDOUT:       %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc31_33: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.9af)]
// CHECK:STDOUT:       %Food.as_type.loc31_37.2: type = facet_access_type %Food.ref.loc31_33 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.952)]
// CHECK:STDOUT:       %.loc31_37.2: type = converted %Food.ref.loc31_33, %Food.as_type.loc31_37.2 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.952)]
// CHECK:STDOUT:       %Eats.type.loc31_37.2: type = facet_type <@Eats, @Eats(constants.%Food.as_type.952)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.b39)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.b39) = bind_symbolic_name T, 1 [symbolic = %T.loc31_24.1 (constants.%T.223)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc31_43.1 (%T.as_type.212) = value_param call_param0
// CHECK:STDOUT:     %.loc31_43.1: type = splice_block %.loc31_43.2 [symbolic = %T.as_type.loc31_43.1 (constants.%T.as_type.212)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc31_37.1 (%Eats.type.b39) = name_ref T, %T.loc31_24.2 [symbolic = %T.loc31_24.1 (constants.%T.223)]
// CHECK:STDOUT:       %T.as_type.loc31_43.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc31_43.1 (constants.%T.as_type.212)]
// CHECK:STDOUT:       %.loc31_43.2: type = converted %T.ref, %T.as_type.loc31_43.2 [symbolic = %T.as_type.loc31_43.1 (constants.%T.as_type.212)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc31_43.1 (%T.as_type.212) = bind_name e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.952) = value_param call_param1
// CHECK:STDOUT:     %.loc31_52.1: type = splice_block %.loc31_52.2 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.952)] {
// CHECK:STDOUT:       %Food.ref.loc31_52: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.9af)]
// CHECK:STDOUT:       %Food.as_type.loc31_52: type = facet_access_type %Food.ref.loc31_52 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.952)]
// CHECK:STDOUT:       %.loc31_52.2: type = converted %Food.ref.loc31_52, %Food.as_type.loc31_52 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.952)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.952) = bind_name food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %Food.patt: %pattern_type.939 = symbolic_binding_pattern Food, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.36a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.36a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.f86) = binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.f86) = value_param_pattern %food.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc32_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc32_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     %Food.loc32_29.2: %Edible.type = bind_symbolic_name Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.5fe)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc32_47.1 (%T.as_type.2ad) = value_param call_param0
// CHECK:STDOUT:     %.loc32_47.1: type = splice_block %.loc32_47.2 [symbolic = %T.as_type.loc32_47.1 (constants.%T.as_type.2ad)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc32_17.2 [symbolic = %T.loc32_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:       %T.as_type.loc32_47.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc32_47.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:       %.loc32_47.2: type = converted %T.ref, %T.as_type.loc32_47.2 [symbolic = %T.as_type.loc32_47.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc32_47.1 (%T.as_type.2ad) = bind_name a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.fae) = value_param call_param1
// CHECK:STDOUT:     %.loc32_56.1: type = splice_block %.loc32_56.2 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.fae)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc32_29.2 [symbolic = %Food.loc32_29.1 (constants.%Food.5fe)]
// CHECK:STDOUT:       %Food.as_type.loc32_56.2: type = facet_access_type %Food.ref [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.fae)]
// CHECK:STDOUT:       %.loc32_56.2: type = converted %Food.ref, %Food.as_type.loc32_56.2 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.fae)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.fae) = bind_name food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value constants.%Grass, (constants.%Edible.impl_witness) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc32_76.1: %Edible.type = converted constants.%Grass, %Edible.facet [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc32_76.2: %Animal.type = converted constants.%Goat, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Edible {
// CHECK:STDOUT:   %Self: %Edible.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9af]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fd4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Eats(%Food.loc21_16.2: type) {
// CHECK:STDOUT:   %Food.loc21_16.1: type = bind_symbolic_name Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc21_16.1)> [symbolic = %Eats.type (constants.%Eats.type.6c0)]
// CHECK:STDOUT:   %Self.2: @Eats.%Eats.type (%Eats.type.6c0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Eats.%Eats.type (%Eats.type.6c0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Grass.as.Destroy.impl: constants.%Grass as constants.%Destroy.type {
// CHECK:STDOUT:   %Grass.as.Destroy.impl.Op.decl: %Grass.as.Destroy.impl.Op.type = fn_decl @Grass.as.Destroy.impl.Op [concrete = constants.%Grass.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.4f0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4f0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.2bd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Grass [concrete = constants.%Grass]
// CHECK:STDOUT:     %self: %ptr.2bd = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Grass.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Grass.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Grass.as.Edible.impl: %Grass.ref as %Edible.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Edible.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.Eats.impl(%T.loc26_14.1: %Animal.type, %U.loc26_26.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc26_14.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %U.loc26_26.2: %Edible.type = bind_symbolic_name U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   %T.as_type.loc26_38.2: type = facet_access_type %T.loc26_14.2 [symbolic = %T.as_type.loc26_38.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %U.as_type.loc26_49.2: type = facet_access_type %U.loc26_26.2 [symbolic = %U.as_type.loc26_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %Eats.type.loc26_49.2: type = facet_type <@Eats, @Eats(%U.as_type.loc26_49.2)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.f54c3d.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Eats.type.loc26_49.2 [symbolic = %require_complete (constants.%require_complete.42532a.1)]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table, @T.as_type.as.Eats.impl(%T.loc26_14.2, %U.loc26_26.2) [symbolic = %Eats.impl_witness (constants.%Eats.impl_witness.fabf92.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc26_38 as %Eats.type.loc26_49.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Eats.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Destroy.impl: constants.%Goat as constants.%Destroy.type {
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.decl: %Goat.as.Destroy.impl.Op.type = fn_decl @Goat.as.Destroy.impl.Op [concrete = constants.%Goat.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.396 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.396 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc28: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.940 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:     %self: %ptr.940 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Goat.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Goat.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Animal.impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Animal.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Grass {
// CHECK:STDOUT:   impl_decl @Grass.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Grass.as.Destroy.impl.%Grass.as.Destroy.impl.Op.decl), @Grass.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.826]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Grass
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   impl_decl @Goat.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Goat.as.Destroy.impl.%Goat.as.Destroy.impl.Op.decl), @Goat.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.943]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Grass.as.Destroy.impl.Op(%self.param: %ptr.2bd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Goat.as.Destroy.impl.Op(%self.param: %ptr.940) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%Food.loc31_9.2: %Edible.type, %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.b39)) {
// CHECK:STDOUT:   %Food.loc31_9.1: %Edible.type = bind_symbolic_name Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.9af)]
// CHECK:STDOUT:   %Food.as_type.loc31_37.1: type = facet_access_type %Food.loc31_9.1 [symbolic = %Food.as_type.loc31_37.1 (constants.%Food.as_type.952)]
// CHECK:STDOUT:   %Eats.type.loc31_37.1: type = facet_type <@Eats, @Eats(%Food.as_type.loc31_37.1)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.b39)]
// CHECK:STDOUT:   %T.loc31_24.1: @Feed.%Eats.type.loc31_37.1 (%Eats.type.b39) = bind_symbolic_name T, 1 [symbolic = %T.loc31_24.1 (constants.%T.223)]
// CHECK:STDOUT:   %pattern_type.loc31_24: type = pattern_type %Eats.type.loc31_37.1 [symbolic = %pattern_type.loc31_24 (constants.%pattern_type.ed7)]
// CHECK:STDOUT:   %T.as_type.loc31_43.1: type = facet_access_type %T.loc31_24.1 [symbolic = %T.as_type.loc31_43.1 (constants.%T.as_type.212)]
// CHECK:STDOUT:   %pattern_type.loc31_40: type = pattern_type %T.as_type.loc31_43.1 [symbolic = %pattern_type.loc31_40 (constants.%pattern_type.1a1)]
// CHECK:STDOUT:   %pattern_type.loc31_46: type = pattern_type %Food.as_type.loc31_37.1 [symbolic = %pattern_type.loc31_46 (constants.%pattern_type.54f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41: <witness> = require_complete_type %T.as_type.loc31_43.1 [symbolic = %require_complete.loc31_41 (constants.%require_complete.fe6)]
// CHECK:STDOUT:   %require_complete.loc31_50: <witness> = require_complete_type %Food.as_type.loc31_37.1 [symbolic = %require_complete.loc31_50 (constants.%require_complete.005)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc31_43.1 (%T.as_type.212), %food.param: @Feed.%Food.as_type.loc31_37.1 (%Food.as_type.952)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc32_17.2: %Animal.type, %Food.loc32_29.2: %Edible.type) {
// CHECK:STDOUT:   %T.loc32_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc32_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:   %Food.loc32_29.1: %Edible.type = bind_symbolic_name Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.5fe)]
// CHECK:STDOUT:   %T.as_type.loc32_47.1: type = facet_access_type %T.loc32_17.1 [symbolic = %T.as_type.loc32_47.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %pattern_type.loc32_44: type = pattern_type %T.as_type.loc32_47.1 [symbolic = %pattern_type.loc32_44 (constants.%pattern_type.36a)]
// CHECK:STDOUT:   %Food.as_type.loc32_56.1: type = facet_access_type %Food.loc32_29.1 [symbolic = %Food.as_type.loc32_56.1 (constants.%Food.as_type.fae)]
// CHECK:STDOUT:   %pattern_type.loc32_50: type = pattern_type %Food.as_type.loc32_56.1 [symbolic = %pattern_type.loc32_50 (constants.%pattern_type.f86)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45: <witness> = require_complete_type %T.as_type.loc32_47.1 [symbolic = %require_complete.loc32_45 (constants.%require_complete.234)]
// CHECK:STDOUT:   %require_complete.loc32_54: <witness> = require_complete_type %Food.as_type.loc32_56.1 [symbolic = %require_complete.loc32_54 (constants.%require_complete.444)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc32_17.1, @Eats, @Eats(%Food.as_type.loc32_56.1) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.as_type.loc32_56.1)> [symbolic = %Eats.type (constants.%Eats.type.f54c3d.2)]
// CHECK:STDOUT:   %Eats.facet.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.f54c3d.2) = facet_value %T.as_type.loc32_47.1, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.512)]
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc32_29.1, %Eats.facet.loc32_76.2) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.ea3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc32_47.1 (%T.as_type.2ad), %food.param: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.fae)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc32_47.1 (%T.as_type.2ad) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.as_type.loc32_56.1 (%Food.as_type.fae) = name_ref food, %food
// CHECK:STDOUT:     %.loc32_76.1: %Edible.type = converted constants.%Food.as_type.fae, constants.%Food.5fe [symbolic = %Food.loc32_29.1 (constants.%Food.5fe)]
// CHECK:STDOUT:     %.loc32_76.2: %Edible.type = converted constants.%Food.as_type.fae, constants.%Food.5fe [symbolic = %Food.loc32_29.1 (constants.%Food.5fe)]
// CHECK:STDOUT:     %.loc32_76.3: %Edible.type = converted constants.%Food.as_type.fae, constants.%Food.5fe [symbolic = %Food.loc32_29.1 (constants.%Food.5fe)]
// CHECK:STDOUT:     %T.as_type.loc32_76: type = facet_access_type constants.%T.fd4 [symbolic = %T.as_type.loc32_47.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc32_76.4: type = converted constants.%T.fd4, %T.as_type.loc32_76 [symbolic = %T.as_type.loc32_47.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc32_76.5: %Animal.type = converted %.loc32_76.4, constants.%T.fd4 [symbolic = %T.loc32_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:     %Eats.facet.loc32_76.1: @HandleAnimal.%Eats.type (%Eats.type.f54c3d.2) = facet_value constants.%T.as_type.2ad, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.512)]
// CHECK:STDOUT:     %.loc32_76.6: @HandleAnimal.%Eats.type (%Eats.type.f54c3d.2) = converted constants.%T.as_type.2ad, %Eats.facet.loc32_76.1 [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.512)]
// CHECK:STDOUT:     %Feed.specific_fn.loc32_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.5fe, constants.%Eats.facet.512) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.ea3)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc32_64.1(%a.ref, %food.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc35_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc35_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc35_17.3: init %Goat = class_init (), %.loc35_17.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc35_17.4: ref %Goat = temporary %.loc35_17.2, %.loc35_17.3
// CHECK:STDOUT:   %.loc35_19.1: ref %Goat = converted %.loc35_17.1, %.loc35_17.4
// CHECK:STDOUT:   %.loc35_29.1: %empty_struct_type = struct_literal ()
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
// CHECK:STDOUT:   %.loc35_19.2: %Goat = bind_value %.loc35_19.1
// CHECK:STDOUT:   %.loc35_31.2: %Grass = bind_value %.loc35_31.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc35_19.2, %.loc35_31.2)
// CHECK:STDOUT:   %Grass.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc35_29.2, constants.%Grass.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc35_29: %ptr.2bd = addr_of %.loc35_29.2
// CHECK:STDOUT:   %Grass.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Grass.as.Destroy.impl.Op.bound(%addr.loc35_29)
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc35_17.2, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc35_17: %ptr.940 = addr_of %.loc35_17.2
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound(%addr.loc35_17)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.8b3) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.as_type) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.f54c3d.1
// CHECK:STDOUT:   %Self.2 => constants.%Self.c30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%T.fd4, constants.%U) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%U
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%U.as_type
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.f54c3d.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.42532a.1
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.fabf92.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.952) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.as_type.952
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.9af, constants.%T.223) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.9af
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Food.as_type.952
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.b39
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%T.223
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.ed7
// CHECK:STDOUT:   %T.as_type.loc31_43.1 => constants.%T.as_type.212
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.1a1
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.54f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.fd4, constants.%Food.5fe) {
// CHECK:STDOUT:   %T.loc32_17.1 => constants.%T.fd4
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Food.5fe
// CHECK:STDOUT:   %T.as_type.loc32_47.1 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.36a
// CHECK:STDOUT:   %Food.as_type.loc32_56.1 => constants.%Food.as_type.fae
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.f86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.fae) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.as_type.fae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%T.fd4, constants.%Food.5fe) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Food.5fe
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%Food.as_type.fae
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.f54c3d.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.42532a.2
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.fabf92.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.5fe, constants.%Eats.facet.512) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.5fe
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Food.as_type.fae
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.f54c3d.2
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.512
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.bb9
// CHECK:STDOUT:   %T.as_type.loc31_43.1 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.36a
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.f86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41 => constants.%require_complete.234
// CHECK:STDOUT:   %require_complete.loc31_50 => constants.%require_complete.444
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc32_17.1 => constants.%Animal.facet
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Edible.facet
// CHECK:STDOUT:   %T.as_type.loc32_47.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.ab7
// CHECK:STDOUT:   %Food.as_type.loc32_56.1 => constants.%Grass
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.aff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc32_54 => constants.%complete_type.357
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%Eats.impl_witness.1cf
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1ae
// CHECK:STDOUT:   %Eats.facet.loc32_76.2 => constants.%Eats.facet.cb1
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2 => constants.%Feed.specific_fn.86c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1ae
// CHECK:STDOUT:   %Self.2 => constants.%Self.016
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.Eats.impl(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%Animal.facet
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Edible.facet
// CHECK:STDOUT:   %T.as_type.loc26_38.2 => constants.%Goat
// CHECK:STDOUT:   %U.as_type.loc26_49.2 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.1ae
// CHECK:STDOUT:   %require_complete => constants.%complete_type.004
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.1cf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.cb1) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.as_type.loc31_37.1 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.1ae
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.cb1
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.d77
// CHECK:STDOUT:   %T.as_type.loc31_43.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.ab7
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.aff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_50 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

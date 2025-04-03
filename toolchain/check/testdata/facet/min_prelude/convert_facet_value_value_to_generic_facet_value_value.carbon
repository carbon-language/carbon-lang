// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_value_to_generic_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_value_to_generic_facet_value_value.carbon

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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Food.8b3: type = bind_symbolic_name Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.patt.e01: type = symbolic_binding_pattern Food, 0 [symbolic]
// CHECK:STDOUT:   %Eats.type.ba2: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.6c0: type = facet_type <@Eats, @Eats(%Food.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.4eb: %Eats.type.6c0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.fd4: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.a9c: %Animal.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: %Edible.type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: %Edible.type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.2ad: type = facet_access_type %T.fd4 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %Eats.type.f54c3d.1: type = facet_type <@Eats, @Eats(%U.as_type)> [symbolic]
// CHECK:STDOUT:   %Self.c30: %Eats.type.f54c3d.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.42532a.1: <witness> = require_complete_type %Eats.type.f54c3d.1 [symbolic]
// CHECK:STDOUT:   %impl_witness.c7c36b.1: <witness> = impl_witness (), @impl.009(%T.fd4, %U) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Food.9af: %Edible.type = bind_symbolic_name Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.patt.0b7: %Edible.type = symbolic_binding_pattern Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.as_type.952: type = facet_access_type %Food.9af [symbolic]
// CHECK:STDOUT:   %Eats.type.b39: type = facet_type <@Eats, @Eats(%Food.as_type.952)> [symbolic]
// CHECK:STDOUT:   %T.223: %Eats.type.b39 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.f09: %Eats.type.b39 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.212: type = facet_access_type %T.223 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fe6: <witness> = require_complete_type %T.as_type.212 [symbolic]
// CHECK:STDOUT:   %require_complete.005: <witness> = require_complete_type %Food.as_type.952 [symbolic]
// CHECK:STDOUT:   %Food.5fe: %Edible.type = bind_symbolic_name Food, 1 [symbolic]
// CHECK:STDOUT:   %Food.patt.551: %Edible.type = symbolic_binding_pattern Food, 1 [symbolic]
// CHECK:STDOUT:   %Food.as_type.fae: type = facet_access_type %Food.5fe [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.234: <witness> = require_complete_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %require_complete.444: <witness> = require_complete_type %Food.as_type.fae [symbolic]
// CHECK:STDOUT:   %Eats.type.f54c3d.2: type = facet_type <@Eats, @Eats(%Food.as_type.fae)> [symbolic]
// CHECK:STDOUT:   %require_complete.42532a.2: <witness> = require_complete_type %Eats.type.f54c3d.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.c7c36b.2: <witness> = impl_witness (), @impl.009(%T.fd4, %Food.5fe) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.fd4, @Eats, @Eats(%Food.as_type.fae) [symbolic]
// CHECK:STDOUT:   %Eats.facet.512: %Eats.type.f54c3d.2 = facet_value %T.as_type.2ad, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %T.patt.1f1: %Eats.type.f54c3d.2 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.ea3: <specific function> = specific_function %Feed, @Feed(%Food.5fe, %Eats.facet.512) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%impl_witness.1bc) [concrete]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value %Grass, (%impl_witness.1bc) [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.type.1ae: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Self.016: %Eats.type.1ae = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.004: <witness> = complete_type_witness %Eats.type.1ae [concrete]
// CHECK:STDOUT:   %impl_witness.4f1: <witness> = impl_witness (), @impl.009(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.fa6: %Eats.type.1ae = facet_value %Goat, (%impl_witness.4f1) [concrete]
// CHECK:STDOUT:   %T.patt.1ac: %Eats.type.1ae = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.d82: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.fa6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @impl.592 [concrete] {} {
// CHECK:STDOUT:     %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Eats.decl: %Eats.type.ba2 = interface_decl @Eats [concrete = constants.%Eats.generic] {
// CHECK:STDOUT:     %Food.patt.loc20_16.1: type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc20_16.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Food.loc20_16.1: type = bind_symbolic_name Food, 0 [symbolic = %Food.loc20_16.2 (constants.%Food.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.009 [concrete] {
// CHECK:STDOUT:     %T.patt.loc25_14.1: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_14.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:     %U.patt.loc25_26.1: %Edible.type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc25_26.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc25_14.1 [symbolic = %T.loc25_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %T.as_type.loc25_38.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc25_38.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc25_38: type = converted %T.ref, %T.as_type.loc25_38.1 [symbolic = %T.as_type.loc25_38.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc25_26.1 [symbolic = %U.loc25_26.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc25_49.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc25_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc25_49: type = converted %U.ref, %U.as_type.loc25_49.1 [symbolic = %U.as_type.loc25_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %Eats.type.loc25_49.1: type = facet_type <@Eats, @Eats(constants.%U.as_type)> [symbolic = %Eats.type.loc25_49.2 (constants.%Eats.type.f54c3d.1)]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc25_14.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc25_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     %U.loc25_26.1: %Edible.type = bind_symbolic_name U, 1 [symbolic = %U.loc25_26.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (), @impl.009(constants.%T.fd4, constants.%U) [symbolic = @impl.009.%impl_witness (constants.%impl_witness.c7c36b.1)]
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl.27e [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc28: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %Food.patt.loc30_9.1: %Edible.type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc30_9.2 (constants.%Food.patt.0b7)]
// CHECK:STDOUT:     %T.patt.loc30_24.1: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc30_24.2 (constants.%T.patt.f09)]
// CHECK:STDOUT:     %e.patt: @Feed.%T.as_type.loc30_43.2 (%T.as_type.212) = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: @Feed.%T.as_type.loc30_43.2 (%T.as_type.212) = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:     %food.patt: @Feed.%Food.as_type.loc30_37.2 (%Food.as_type.952) = binding_pattern food
// CHECK:STDOUT:     %food.param_patt: @Feed.%Food.as_type.loc30_37.2 (%Food.as_type.952) = value_param_pattern %food.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     %Food.loc30_9.1: %Edible.type = bind_symbolic_name Food, 0 [symbolic = %Food.loc30_9.2 (constants.%Food.9af)]
// CHECK:STDOUT:     %.loc30_37.1: type = splice_block %Eats.type.loc30_37.1 [symbolic = %Eats.type.loc30_37.2 (constants.%Eats.type.b39)] {
// CHECK:STDOUT:       %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc30_33: %Edible.type = name_ref Food, %Food.loc30_9.1 [symbolic = %Food.loc30_9.2 (constants.%Food.9af)]
// CHECK:STDOUT:       %Food.as_type.loc30_37.1: type = facet_access_type %Food.ref.loc30_33 [symbolic = %Food.as_type.loc30_37.2 (constants.%Food.as_type.952)]
// CHECK:STDOUT:       %.loc30_37.2: type = converted %Food.ref.loc30_33, %Food.as_type.loc30_37.1 [symbolic = %Food.as_type.loc30_37.2 (constants.%Food.as_type.952)]
// CHECK:STDOUT:       %Eats.type.loc30_37.1: type = facet_type <@Eats, @Eats(constants.%Food.as_type.952)> [symbolic = %Eats.type.loc30_37.2 (constants.%Eats.type.b39)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc30_24.1: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39) = bind_symbolic_name T, 1 [symbolic = %T.loc30_24.2 (constants.%T.223)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc30_43.2 (%T.as_type.212) = value_param call_param0
// CHECK:STDOUT:     %.loc30_43.1: type = splice_block %.loc30_43.2 [symbolic = %T.as_type.loc30_43.2 (constants.%T.as_type.212)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39) = name_ref T, %T.loc30_24.1 [symbolic = %T.loc30_24.2 (constants.%T.223)]
// CHECK:STDOUT:       %T.as_type.loc30_43.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc30_43.2 (constants.%T.as_type.212)]
// CHECK:STDOUT:       %.loc30_43.2: type = converted %T.ref, %T.as_type.loc30_43.1 [symbolic = %T.as_type.loc30_43.2 (constants.%T.as_type.212)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc30_43.2 (%T.as_type.212) = bind_name e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.as_type.loc30_37.2 (%Food.as_type.952) = value_param call_param1
// CHECK:STDOUT:     %.loc30_52.1: type = splice_block %.loc30_52.2 [symbolic = %Food.as_type.loc30_37.2 (constants.%Food.as_type.952)] {
// CHECK:STDOUT:       %Food.ref.loc30_52: %Edible.type = name_ref Food, %Food.loc30_9.1 [symbolic = %Food.loc30_9.2 (constants.%Food.9af)]
// CHECK:STDOUT:       %Food.as_type.loc30_52: type = facet_access_type %Food.ref.loc30_52 [symbolic = %Food.as_type.loc30_37.2 (constants.%Food.as_type.952)]
// CHECK:STDOUT:       %.loc30_52.2: type = converted %Food.ref.loc30_52, %Food.as_type.loc30_52 [symbolic = %Food.as_type.loc30_37.2 (constants.%Food.as_type.952)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.as_type.loc30_37.2 (%Food.as_type.952) = bind_name food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt.loc31_17.1: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_17.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:     %Food.patt.loc31_29.1: %Edible.type = symbolic_binding_pattern Food, 1 [symbolic = %Food.patt.loc31_29.2 (constants.%Food.patt.551)]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae) = binding_pattern food
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae) = value_param_pattern %food.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc31_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc31_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     %Food.loc31_29.1: %Edible.type = bind_symbolic_name Food, 1 [symbolic = %Food.loc31_29.2 (constants.%Food.5fe)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad) = value_param call_param0
// CHECK:STDOUT:     %.loc31_47.1: type = splice_block %.loc31_47.2 [symbolic = %T.as_type.loc31_47.2 (constants.%T.as_type.2ad)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc31_17.1 [symbolic = %T.loc31_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:       %T.as_type.loc31_47.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc31_47.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:       %.loc31_47.2: type = converted %T.ref, %T.as_type.loc31_47.1 [symbolic = %T.as_type.loc31_47.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad) = bind_name a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae) = value_param call_param1
// CHECK:STDOUT:     %.loc31_56.1: type = splice_block %.loc31_56.2 [symbolic = %Food.as_type.loc31_56.2 (constants.%Food.as_type.fae)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc31_29.1 [symbolic = %Food.loc31_29.2 (constants.%Food.5fe)]
// CHECK:STDOUT:       %Food.as_type.loc31_56.1: type = facet_access_type %Food.ref [symbolic = %Food.as_type.loc31_56.2 (constants.%Food.as_type.fae)]
// CHECK:STDOUT:       %.loc31_56.2: type = converted %Food.ref, %Food.as_type.loc31_56.1 [symbolic = %Food.as_type.loc31_56.2 (constants.%Food.as_type.fae)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae) = bind_name food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value constants.%Grass, (constants.%impl_witness.1bc) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc31_76.1: %Edible.type = converted constants.%Grass, %Edible.facet [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness.1bc) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc31_76.2: %Animal.type = converted constants.%Goat, %Animal.facet [concrete = constants.%Animal.facet]
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
// CHECK:STDOUT: generic interface @Eats(%Food.loc20_16.1: type) {
// CHECK:STDOUT:   %Food.loc20_16.2: type = bind_symbolic_name Food, 0 [symbolic = %Food.loc20_16.2 (constants.%Food.8b3)]
// CHECK:STDOUT:   %Food.patt.loc20_16.2: type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc20_16.2 (constants.%Food.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc20_16.2)> [symbolic = %Eats.type (constants.%Eats.type.6c0)]
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
// CHECK:STDOUT: impl @impl.592: %Grass.ref as %Edible.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.009(%T.loc25_14.1: %Animal.type, %U.loc25_26.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc25_14.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc25_14.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.patt.loc25_14.2: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_14.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:   %U.loc25_26.2: %Edible.type = bind_symbolic_name U, 1 [symbolic = %U.loc25_26.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc25_26.2: %Edible.type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc25_26.2 (constants.%U.patt)]
// CHECK:STDOUT:   %T.as_type.loc25_38.2: type = facet_access_type %T.loc25_14.2 [symbolic = %T.as_type.loc25_38.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %U.as_type.loc25_49.2: type = facet_access_type %U.loc25_26.2 [symbolic = %U.as_type.loc25_49.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %Eats.type.loc25_49.2: type = facet_type <@Eats, @Eats(%U.as_type.loc25_49.2)> [symbolic = %Eats.type.loc25_49.2 (constants.%Eats.type.f54c3d.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.009.%Eats.type.loc25_49.2 (%Eats.type.f54c3d.1) [symbolic = %require_complete (constants.%require_complete.42532a.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.009(%T.loc25_14.2, %U.loc25_26.2) [symbolic = %impl_witness (constants.%impl_witness.c7c36b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc25_38 as %Eats.type.loc25_49.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.27e: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Grass {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Grass
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%Food.loc30_9.1: %Edible.type, %T.loc30_24.1: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39)) {
// CHECK:STDOUT:   %Food.loc30_9.2: %Edible.type = bind_symbolic_name Food, 0 [symbolic = %Food.loc30_9.2 (constants.%Food.9af)]
// CHECK:STDOUT:   %Food.patt.loc30_9.2: %Edible.type = symbolic_binding_pattern Food, 0 [symbolic = %Food.patt.loc30_9.2 (constants.%Food.patt.0b7)]
// CHECK:STDOUT:   %Food.as_type.loc30_37.2: type = facet_access_type %Food.loc30_9.2 [symbolic = %Food.as_type.loc30_37.2 (constants.%Food.as_type.952)]
// CHECK:STDOUT:   %Eats.type.loc30_37.2: type = facet_type <@Eats, @Eats(%Food.as_type.loc30_37.2)> [symbolic = %Eats.type.loc30_37.2 (constants.%Eats.type.b39)]
// CHECK:STDOUT:   %T.loc30_24.2: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39) = bind_symbolic_name T, 1 [symbolic = %T.loc30_24.2 (constants.%T.223)]
// CHECK:STDOUT:   %T.patt.loc30_24.2: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc30_24.2 (constants.%T.patt.f09)]
// CHECK:STDOUT:   %T.as_type.loc30_43.2: type = facet_access_type %T.loc30_24.2 [symbolic = %T.as_type.loc30_43.2 (constants.%T.as_type.212)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30_41: <witness> = require_complete_type @Feed.%T.as_type.loc30_43.2 (%T.as_type.212) [symbolic = %require_complete.loc30_41 (constants.%require_complete.fe6)]
// CHECK:STDOUT:   %require_complete.loc30_50: <witness> = require_complete_type @Feed.%Food.as_type.loc30_37.2 (%Food.as_type.952) [symbolic = %require_complete.loc30_50 (constants.%require_complete.005)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%Food.patt.loc30_9.1: %Edible.type, %T.patt.loc30_24.1: @Feed.%Eats.type.loc30_37.2 (%Eats.type.b39)](%e.param_patt: @Feed.%T.as_type.loc30_43.2 (%T.as_type.212), %food.param_patt: @Feed.%Food.as_type.loc30_37.2 (%Food.as_type.952)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc31_17.1: %Animal.type, %Food.loc31_29.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc31_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc31_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.patt.loc31_17.2: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_17.2 (constants.%T.patt.a9c)]
// CHECK:STDOUT:   %Food.loc31_29.2: %Edible.type = bind_symbolic_name Food, 1 [symbolic = %Food.loc31_29.2 (constants.%Food.5fe)]
// CHECK:STDOUT:   %Food.patt.loc31_29.2: %Edible.type = symbolic_binding_pattern Food, 1 [symbolic = %Food.patt.loc31_29.2 (constants.%Food.patt.551)]
// CHECK:STDOUT:   %T.as_type.loc31_47.2: type = facet_access_type %T.loc31_17.2 [symbolic = %T.as_type.loc31_47.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %Food.as_type.loc31_56.2: type = facet_access_type %Food.loc31_29.2 [symbolic = %Food.as_type.loc31_56.2 (constants.%Food.as_type.fae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_45: <witness> = require_complete_type @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad) [symbolic = %require_complete.loc31_45 (constants.%require_complete.234)]
// CHECK:STDOUT:   %require_complete.loc31_54: <witness> = require_complete_type @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae) [symbolic = %require_complete.loc31_54 (constants.%require_complete.444)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.as_type.loc31_56.2)> [symbolic = %Eats.type (constants.%Eats.type.f54c3d.2)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc31_17.2, @Eats, @Eats(%Food.as_type.loc31_56.2) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc31_76.2: @HandleAnimal.%Eats.type (%Eats.type.f54c3d.2) = facet_value %T.as_type.loc31_47.2, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc31_76.2 (constants.%Eats.facet.512)]
// CHECK:STDOUT:   %Feed.specific_fn.loc31_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc31_29.2, %Eats.facet.loc31_76.2) [symbolic = %Feed.specific_fn.loc31_64.2 (constants.%Feed.specific_fn.ea3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc31_17.1: %Animal.type, %Food.patt.loc31_29.1: %Edible.type](%a.param_patt: @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad), %food.param_patt: @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc31_47.2 (%T.as_type.2ad) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.as_type.loc31_56.2 (%Food.as_type.fae) = name_ref food, %food
// CHECK:STDOUT:     %.loc31_76.1: %Edible.type = converted constants.%Food.as_type.fae, constants.%Food.5fe [symbolic = %Food.loc31_29.2 (constants.%Food.5fe)]
// CHECK:STDOUT:     %.loc31_76.2: %Edible.type = converted constants.%Food.as_type.fae, constants.%Food.5fe [symbolic = %Food.loc31_29.2 (constants.%Food.5fe)]
// CHECK:STDOUT:     %.loc31_76.3: %Edible.type = converted constants.%Food.as_type.fae, constants.%Food.5fe [symbolic = %Food.loc31_29.2 (constants.%Food.5fe)]
// CHECK:STDOUT:     %T.as_type.loc31_76: type = facet_access_type constants.%T.fd4 [symbolic = %T.as_type.loc31_47.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc31_76.4: type = converted constants.%T.fd4, %T.as_type.loc31_76 [symbolic = %T.as_type.loc31_47.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc31_76.5: %Animal.type = converted %.loc31_76.4, constants.%T.fd4 [symbolic = %T.loc31_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %Eats.facet.loc31_76.1: @HandleAnimal.%Eats.type (%Eats.type.f54c3d.2) = facet_value constants.%T.as_type.2ad, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc31_76.2 (constants.%Eats.facet.512)]
// CHECK:STDOUT:     %.loc31_76.6: @HandleAnimal.%Eats.type (%Eats.type.f54c3d.2) = converted constants.%T.as_type.2ad, %Eats.facet.loc31_76.1 [symbolic = %Eats.facet.loc31_76.2 (constants.%Eats.facet.512)]
// CHECK:STDOUT:     %Feed.specific_fn.loc31_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.5fe, constants.%Eats.facet.512) [symbolic = %Feed.specific_fn.loc31_64.2 (constants.%Feed.specific_fn.ea3)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc31_64.1(%a.ref, %food.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc34_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc34_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc34_17.3: init %Goat = class_init (), %.loc34_17.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc34_17.4: ref %Goat = temporary %.loc34_17.2, %.loc34_17.3
// CHECK:STDOUT:   %.loc34_19.1: ref %Goat = converted %.loc34_17.1, %.loc34_17.4
// CHECK:STDOUT:   %.loc34_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Grass.ref: type = name_ref Grass, file.%Grass.decl [concrete = constants.%Grass]
// CHECK:STDOUT:   %.loc34_29.2: ref %Grass = temporary_storage
// CHECK:STDOUT:   %.loc34_29.3: init %Grass = class_init (), %.loc34_29.2 [concrete = constants.%Grass.val]
// CHECK:STDOUT:   %.loc34_29.4: ref %Grass = temporary %.loc34_29.2, %.loc34_29.3
// CHECK:STDOUT:   %.loc34_31.1: ref %Grass = converted %.loc34_29.1, %.loc34_29.4
// CHECK:STDOUT:   %Animal.facet.loc34_39.1: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness.1bc) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc34_39.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc34_39.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc34_39.2: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness.1bc) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc34_39.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc34_39.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Edible.facet.loc34_39.1: %Edible.type = facet_value constants.%Grass, (constants.%impl_witness.1bc) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc34_39.3: %Edible.type = converted constants.%Grass, %Edible.facet.loc34_39.1 [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %Edible.facet.loc34_39.2: %Edible.type = facet_value constants.%Grass, (constants.%impl_witness.1bc) [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %.loc34_39.4: %Edible.type = converted constants.%Grass, %Edible.facet.loc34_39.2 [concrete = constants.%Edible.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc34_19.2: %Goat = bind_value %.loc34_19.1
// CHECK:STDOUT:   %.loc34_31.2: %Grass = bind_value %.loc34_31.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc34_19.2, %.loc34_31.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.8b3) {
// CHECK:STDOUT:   %Food.loc20_16.2 => constants.%Food.8b3
// CHECK:STDOUT:   %Food.patt.loc20_16.2 => constants.%Food.patt.e01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(%Food.loc20_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.as_type) {
// CHECK:STDOUT:   %Food.loc20_16.2 => constants.%U.as_type
// CHECK:STDOUT:   %Food.patt.loc20_16.2 => constants.%Food.patt.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.f54c3d.1
// CHECK:STDOUT:   %Self.2 => constants.%Self.c30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.009(constants.%T.fd4, constants.%U) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%T.patt.a9c
// CHECK:STDOUT:   %U.loc25_26.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc25_26.2 => constants.%U.patt
// CHECK:STDOUT:   %T.as_type.loc25_38.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %U.as_type.loc25_49.2 => constants.%U.as_type
// CHECK:STDOUT:   %Eats.type.loc25_49.2 => constants.%Eats.type.f54c3d.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.42532a.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.c7c36b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(@impl.009.%U.as_type.loc25_49.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.009(%T.loc25_14.2, %U.loc25_26.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.952) {
// CHECK:STDOUT:   %Food.loc20_16.2 => constants.%Food.as_type.952
// CHECK:STDOUT:   %Food.patt.loc20_16.2 => constants.%Food.patt.e01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.9af, constants.%T.223) {
// CHECK:STDOUT:   %Food.loc30_9.2 => constants.%Food.9af
// CHECK:STDOUT:   %Food.patt.loc30_9.2 => constants.%Food.patt.0b7
// CHECK:STDOUT:   %Food.as_type.loc30_37.2 => constants.%Food.as_type.952
// CHECK:STDOUT:   %Eats.type.loc30_37.2 => constants.%Eats.type.b39
// CHECK:STDOUT:   %T.loc30_24.2 => constants.%T.223
// CHECK:STDOUT:   %T.patt.loc30_24.2 => constants.%T.patt.f09
// CHECK:STDOUT:   %T.as_type.loc30_43.2 => constants.%T.as_type.212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(@Feed.%Food.as_type.loc30_37.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.fd4, constants.%Food.5fe) {
// CHECK:STDOUT:   %T.loc31_17.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.patt.loc31_17.2 => constants.%T.patt.a9c
// CHECK:STDOUT:   %Food.loc31_29.2 => constants.%Food.5fe
// CHECK:STDOUT:   %Food.patt.loc31_29.2 => constants.%Food.patt.551
// CHECK:STDOUT:   %T.as_type.loc31_47.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %Food.as_type.loc31_56.2 => constants.%Food.as_type.fae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.as_type.fae) {
// CHECK:STDOUT:   %Food.loc20_16.2 => constants.%Food.as_type.fae
// CHECK:STDOUT:   %Food.patt.loc20_16.2 => constants.%Food.patt.e01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.009(constants.%T.fd4, constants.%Food.5fe) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%T.patt.a9c
// CHECK:STDOUT:   %U.loc25_26.2 => constants.%Food.5fe
// CHECK:STDOUT:   %U.patt.loc25_26.2 => constants.%U.patt
// CHECK:STDOUT:   %T.as_type.loc25_38.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %U.as_type.loc25_49.2 => constants.%Food.as_type.fae
// CHECK:STDOUT:   %Eats.type.loc25_49.2 => constants.%Eats.type.f54c3d.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.42532a.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.c7c36b.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.5fe, constants.%Eats.facet.512) {
// CHECK:STDOUT:   %Food.loc30_9.2 => constants.%Food.5fe
// CHECK:STDOUT:   %Food.patt.loc30_9.2 => constants.%Food.patt.0b7
// CHECK:STDOUT:   %Food.as_type.loc30_37.2 => constants.%Food.as_type.fae
// CHECK:STDOUT:   %Eats.type.loc30_37.2 => constants.%Eats.type.f54c3d.2
// CHECK:STDOUT:   %T.loc30_24.2 => constants.%Eats.facet.512
// CHECK:STDOUT:   %T.patt.loc30_24.2 => constants.%T.patt.1f1
// CHECK:STDOUT:   %T.as_type.loc30_43.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30_41 => constants.%require_complete.234
// CHECK:STDOUT:   %require_complete.loc30_50 => constants.%require_complete.444
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(@HandleAnimal.%Food.as_type.loc31_56.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(@HandleAnimal.%Food.loc31_29.2, @HandleAnimal.%Eats.facet.loc31_76.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc31_17.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.patt.loc31_17.2 => constants.%T.patt.a9c
// CHECK:STDOUT:   %Food.loc31_29.2 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.patt.loc31_29.2 => constants.%Food.patt.551
// CHECK:STDOUT:   %T.as_type.loc31_47.2 => constants.%Goat
// CHECK:STDOUT:   %Food.as_type.loc31_56.2 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_45 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_54 => constants.%complete_type.357
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1ae
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%impl_witness.4f1
// CHECK:STDOUT:   %Eats.facet.loc31_76.2 => constants.%Eats.facet.fa6
// CHECK:STDOUT:   %Feed.specific_fn.loc31_64.2 => constants.%Feed.specific_fn.d82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc20_16.2 => constants.%Grass
// CHECK:STDOUT:   %Food.patt.loc20_16.2 => constants.%Food.patt.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.1ae
// CHECK:STDOUT:   %Self.2 => constants.%Self.016
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.009(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%T.patt.a9c
// CHECK:STDOUT:   %U.loc25_26.2 => constants.%Edible.facet
// CHECK:STDOUT:   %U.patt.loc25_26.2 => constants.%U.patt
// CHECK:STDOUT:   %T.as_type.loc25_38.2 => constants.%Goat
// CHECK:STDOUT:   %U.as_type.loc25_49.2 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc25_49.2 => constants.%Eats.type.1ae
// CHECK:STDOUT:   %require_complete => constants.%complete_type.004
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4f1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.fa6) {
// CHECK:STDOUT:   %Food.loc30_9.2 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.patt.loc30_9.2 => constants.%Food.patt.0b7
// CHECK:STDOUT:   %Food.as_type.loc30_37.2 => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc30_37.2 => constants.%Eats.type.1ae
// CHECK:STDOUT:   %T.loc30_24.2 => constants.%Eats.facet.fa6
// CHECK:STDOUT:   %T.patt.loc30_24.2 => constants.%T.patt.1ac
// CHECK:STDOUT:   %T.as_type.loc30_43.2 => constants.%Goat
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30_41 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc30_50 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc8_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc8_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc8_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc8_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc12_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc12_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc12_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc12_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc8_14.1: type) {
// CHECK:STDOUT:   %Dest.loc8_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc8_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc8_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc8_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc8_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc8_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc9_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc8_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc9_20.1: type = splice_block %.loc9_20.3 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc9_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc9_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc9_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc9_20.3: type = converted %Self.ref, %Self.as_type.loc9_20.2 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc9_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc12_22.1: type) {
// CHECK:STDOUT:   %Dest.loc12_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc12_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc12_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc12_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc12_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc12_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc12_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc8_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc9_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc12_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc8_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc8_14.2 => constants.%Dest.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc12_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc12_22.2 => constants.%Dest.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc12_22.2) {}
// CHECK:STDOUT:

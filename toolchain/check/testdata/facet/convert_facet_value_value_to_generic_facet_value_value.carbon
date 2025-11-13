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
// CHECK:STDOUT:   %Self.17b: %Edible.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Grass: type = class_type @Grass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Edible.impl_witness: <witness> = impl_witness file.%Edible.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.84c: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Food.d9f: type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Eats.type.ba2: type = generic_interface_type @Eats [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.generic: %Eats.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.type.c18: type = facet_type <@Eats, @Eats(%Food.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.4e3: %Eats.type.c18 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.84c: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %U: %Edible.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ae0: type = pattern_type %Edible.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.5cd: type = symbolic_binding_type T, 0, %T.84c [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %Eats.type.98f93b.1: type = facet_type <@Eats, @Eats(%U.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %Self.9eb: %Eats.type.98f93b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.cc38c8.1: <witness> = require_complete_type %Eats.type.98f93b.1 [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.42a0e2.1: <witness> = impl_witness file.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%T.84c, %U) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness file.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Food.17b: %Edible.type = symbolic_binding Food, 0 [symbolic]
// CHECK:STDOUT:   %Food.binding.as_type.121: type = symbolic_binding_type Food, 0, %Food.17b [symbolic]
// CHECK:STDOUT:   %Eats.type.3cc: type = facet_type <@Eats, @Eats(%Food.binding.as_type.121)> [symbolic]
// CHECK:STDOUT:   %T.7e0: %Eats.type.3cc = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.67b: type = pattern_type %Eats.type.3cc [symbolic]
// CHECK:STDOUT:   %Self.7e0: %Eats.type.3cc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.97d: type = symbolic_binding_type T, 1, %T.7e0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ce: type = pattern_type %T.binding.as_type.97d [symbolic]
// CHECK:STDOUT:   %pattern_type.25b: type = pattern_type %Food.binding.as_type.121 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1cf: <witness> = require_complete_type %T.binding.as_type.97d [symbolic]
// CHECK:STDOUT:   %require_complete.8f8: <witness> = require_complete_type %Food.binding.as_type.121 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %Food.e7a: %Edible.type = symbolic_binding Food, 1 [symbolic]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A [symbolic]
// CHECK:STDOUT:   %pattern_type.01b: type = pattern_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %Food.binding.as_type.86a: type = symbolic_binding_type Food, 1, %Food.e7a [symbolic]
// CHECK:STDOUT:   %pattern_type.9f3: type = pattern_type %Food.binding.as_type.86a [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c0f: <witness> = require_complete_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.74d: <witness> = require_complete_type %Food.binding.as_type.86a [symbolic]
// CHECK:STDOUT:   %Eats.type.98f93b.2: type = facet_type <@Eats, @Eats(%Food.binding.as_type.86a)> [symbolic]
// CHECK:STDOUT:   %require_complete.cc38c8.2: <witness> = require_complete_type %Eats.type.98f93b.2 [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.42a0e2.2: <witness> = impl_witness file.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%A, %Food.e7a) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A, @Eats, @Eats(%Food.binding.as_type.86a) [symbolic]
// CHECK:STDOUT:   %Eats.facet.bda: %Eats.type.98f93b.2 = facet_value %A.binding.as_type, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.5a9: type = pattern_type %Eats.type.98f93b.2 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn.8aa: <specific function> = specific_function %Feed, @Feed(%Food.e7a, %Eats.facet.bda) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Grass.val: %Grass = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Edible.facet: %Edible.type = facet_value %Grass, (%Edible.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.ab7: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.aff: type = pattern_type %Grass [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.c0f: %type_where = facet_value %Grass, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.85f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.c0f) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.325: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.85f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2bd: type = ptr_type %Grass [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4fb: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.325, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.c0f) [concrete]
// CHECK:STDOUT:   %facet_value.8dd: %type_where = facet_value %Goat, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5a8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.8dd) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9fd: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5a8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.940: type = ptr_type %Goat [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3f8: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.9fd, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.8dd) [concrete]
// CHECK:STDOUT:   %Eats.type.cee: type = facet_type <@Eats, @Eats(%Grass)> [concrete]
// CHECK:STDOUT:   %Self.026: %Eats.type.cee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.eba: <witness> = complete_type_witness %Eats.type.cee [concrete]
// CHECK:STDOUT:   %Eats.impl_witness.7f7: <witness> = impl_witness file.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%Animal.facet, %Edible.facet) [concrete]
// CHECK:STDOUT:   %Eats.facet.179: %Eats.type.cee = facet_value %Goat, (%Eats.impl_witness.7f7) [concrete]
// CHECK:STDOUT:   %pattern_type.cf8: type = pattern_type %Eats.type.cee [concrete]
// CHECK:STDOUT:   %Feed.specific_fn.577: <specific function> = specific_function %Feed, @Feed(%Edible.facet, %Eats.facet.179) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Food.loc21_16.2: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3c3 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.ae0 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Animal.type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T.84c)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:     %.loc26_38: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:     %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:     %U.ref: %Edible.type = name_ref U, %U.loc26_26.1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %.loc26_49: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %Eats.type.loc26_49.1: type = facet_type <@Eats, @Eats(constants.%U.binding.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.98f93b.1)]
// CHECK:STDOUT:     %.loc26_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_14.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T.84c)]
// CHECK:STDOUT:     %.loc26_30: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc26_26.1: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(constants.%T.84c, constants.%U) [symbolic = @T.binding.as_type.as.Eats.impl.%Eats.impl_witness (constants.%Eats.impl_witness.42a0e2.1)]
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %Food.patt: %pattern_type.ae0 = symbolic_binding_pattern Food, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @Feed.%pattern_type.loc31_24 (%pattern_type.67b) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type.loc31_40 (%pattern_type.3ce) = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type.loc31_40 (%pattern_type.3ce) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:     %food.patt: @Feed.%pattern_type.loc31_46 (%pattern_type.25b) = value_binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @Feed.%pattern_type.loc31_46 (%pattern_type.25b) = value_param_pattern %food.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_16: type = splice_block %Edible.ref [concrete = constants.%Edible.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Edible.ref: type = name_ref Edible, file.%Edible.decl [concrete = constants.%Edible.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Food.loc31_9.2: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.17b)]
// CHECK:STDOUT:     %.loc31_37.1: type = splice_block %Eats.type.loc31_37.2 [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.3cc)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: %Eats.type.ba2 = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.generic]
// CHECK:STDOUT:       %Food.ref.loc31_33: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.17b)]
// CHECK:STDOUT:       %Food.as_type.loc31_37: type = facet_access_type %Food.ref.loc31_33 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.121)]
// CHECK:STDOUT:       %.loc31_37.2: type = converted %Food.ref.loc31_33, %Food.as_type.loc31_37 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.121)]
// CHECK:STDOUT:       %Eats.type.loc31_37.2: type = facet_type <@Eats, @Eats(constants.%Food.binding.as_type.121)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.3cc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3cc) = symbolic_binding T, 1 [symbolic = %T.loc31_24.1 (constants.%T.7e0)]
// CHECK:STDOUT:     %e.param: @Feed.%T.binding.as_type (%T.binding.as_type.97d) = value_param call_param0
// CHECK:STDOUT:     %.loc31_43.1: type = splice_block %.loc31_43.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.97d)] {
// CHECK:STDOUT:       %T.ref: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3cc) = name_ref T, %T.loc31_24.2 [symbolic = %T.loc31_24.1 (constants.%T.7e0)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.97d)]
// CHECK:STDOUT:       %.loc31_43.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.97d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.binding.as_type (%T.binding.as_type.97d) = value_binding e, %e.param
// CHECK:STDOUT:     %food.param: @Feed.%Food.binding.as_type (%Food.binding.as_type.121) = value_param call_param1
// CHECK:STDOUT:     %.loc31_52.1: type = splice_block %.loc31_52.2 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.121)] {
// CHECK:STDOUT:       %Food.ref.loc31_52: %Edible.type = name_ref Food, %Food.loc31_9.2 [symbolic = %Food.loc31_9.1 (constants.%Food.17b)]
// CHECK:STDOUT:       %Food.as_type.loc31_52: type = facet_access_type %Food.ref.loc31_52 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.121)]
// CHECK:STDOUT:       %.loc31_52.2: type = converted %Food.ref.loc31_52, %Food.as_type.loc31_52 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.121)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @Feed.%Food.binding.as_type (%Food.binding.as_type.121) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %A.patt: %pattern_type.3c3 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %Food.patt: %pattern_type.ae0 = symbolic_binding_pattern Food, 1 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.01b) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type.loc32_44 (%pattern_type.01b) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %food.patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.9f3) = value_binding_pattern food [concrete]
// CHECK:STDOUT:     %food.param_patt: @HandleAnimal.%pattern_type.loc32_50 (%pattern_type.9f3) = value_param_pattern %food.patt, call_param1 [concrete]
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
// CHECK:STDOUT:     %Food.loc32_29.2: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.e7a)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc32_47.1: type = splice_block %.loc32_47.2 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)] {
// CHECK:STDOUT:       %A.ref: %Animal.type = name_ref A, %A.loc32_17.2 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type: type = facet_access_type %A.ref [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:       %.loc32_47.2: type = converted %A.ref, %A.as_type [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = value_binding a, %a.param
// CHECK:STDOUT:     %food.param: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.86a) = value_param call_param1
// CHECK:STDOUT:     %.loc32_56.1: type = splice_block %.loc32_56.2 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.86a)] {
// CHECK:STDOUT:       %Food.ref: %Edible.type = name_ref Food, %Food.loc32_29.2 [symbolic = %Food.loc32_29.1 (constants.%Food.e7a)]
// CHECK:STDOUT:       %Food.as_type: type = facet_access_type %Food.ref [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.86a)]
// CHECK:STDOUT:       %.loc32_56.2: type = converted %Food.ref, %Food.as_type [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.86a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %food: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.86a) = value_binding food, %food.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Edible {
// CHECK:STDOUT:   %Self: %Edible.type = symbolic_binding Self, 0 [symbolic = constants.%Self.17b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.84c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Eats(%Food.loc21_16.2: type) {
// CHECK:STDOUT:   %Food.loc21_16.1: type = symbolic_binding Food, 0 [symbolic = %Food.loc21_16.1 (constants.%Food.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.loc21_16.1)> [symbolic = %Eats.type (constants.%Eats.type.c18)]
// CHECK:STDOUT:   %Self.loc21_29.2: @Eats.%Eats.type (%Eats.type.c18) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.4e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc21_29.1: @Eats.%Eats.type (%Eats.type.c18) = symbolic_binding Self, 1 [symbolic = %Self.loc21_29.2 (constants.%Self.4e3)]
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
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Edible.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Eats.impl(%T.loc26_14.1: %Animal.type, %U.loc26_26.1: %Edible.type) {
// CHECK:STDOUT:   %T.loc26_14.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T.84c)]
// CHECK:STDOUT:   %U.loc26_26.2: %Edible.type = symbolic_binding U, 1 [symbolic = %U.loc26_26.2 (constants.%U)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc26_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc26_26.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %Eats.type.loc26_49.2: type = facet_type <@Eats, @Eats(%U.binding.as_type)> [symbolic = %Eats.type.loc26_49.2 (constants.%Eats.type.98f93b.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Eats.type.loc26_49.2 [symbolic = %require_complete (constants.%require_complete.cc38c8.1)]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table, @T.binding.as_type.as.Eats.impl(%T.loc26_14.2, %U.loc26_26.2) [symbolic = %Eats.impl_witness (constants.%Eats.impl_witness.42a0e2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc26_38 as %Eats.type.loc26_49.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Eats.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Animal.impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Animal.impl_witness
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
// CHECK:STDOUT: generic fn @Feed(%Food.loc31_9.2: %Edible.type, %T.loc31_24.2: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3cc)) {
// CHECK:STDOUT:   %Food.loc31_9.1: %Edible.type = symbolic_binding Food, 0 [symbolic = %Food.loc31_9.1 (constants.%Food.17b)]
// CHECK:STDOUT:   %Food.binding.as_type: type = symbolic_binding_type Food, 0, %Food.loc31_9.1 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.121)]
// CHECK:STDOUT:   %Eats.type.loc31_37.1: type = facet_type <@Eats, @Eats(%Food.binding.as_type)> [symbolic = %Eats.type.loc31_37.1 (constants.%Eats.type.3cc)]
// CHECK:STDOUT:   %T.loc31_24.1: @Feed.%Eats.type.loc31_37.1 (%Eats.type.3cc) = symbolic_binding T, 1 [symbolic = %T.loc31_24.1 (constants.%T.7e0)]
// CHECK:STDOUT:   %pattern_type.loc31_24: type = pattern_type %Eats.type.loc31_37.1 [symbolic = %pattern_type.loc31_24 (constants.%pattern_type.67b)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 1, %T.loc31_24.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.97d)]
// CHECK:STDOUT:   %pattern_type.loc31_40: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc31_40 (constants.%pattern_type.3ce)]
// CHECK:STDOUT:   %pattern_type.loc31_46: type = pattern_type %Food.binding.as_type [symbolic = %pattern_type.loc31_46 (constants.%pattern_type.25b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc31_41 (constants.%require_complete.1cf)]
// CHECK:STDOUT:   %require_complete.loc31_50: <witness> = require_complete_type %Food.binding.as_type [symbolic = %require_complete.loc31_50 (constants.%require_complete.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.binding.as_type (%T.binding.as_type.97d), %food.param: @Feed.%Food.binding.as_type (%Food.binding.as_type.121)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%A.loc32_17.2: %Animal.type, %Food.loc32_29.2: %Edible.type) {
// CHECK:STDOUT:   %A.loc32_17.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc32_17.1 (constants.%A)]
// CHECK:STDOUT:   %Food.loc32_29.1: %Edible.type = symbolic_binding Food, 1 [symbolic = %Food.loc32_29.1 (constants.%Food.e7a)]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A.loc32_17.1 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc32_44: type = pattern_type %A.binding.as_type [symbolic = %pattern_type.loc32_44 (constants.%pattern_type.01b)]
// CHECK:STDOUT:   %Food.binding.as_type: type = symbolic_binding_type Food, 1, %Food.loc32_29.1 [symbolic = %Food.binding.as_type (constants.%Food.binding.as_type.86a)]
// CHECK:STDOUT:   %pattern_type.loc32_50: type = pattern_type %Food.binding.as_type [symbolic = %pattern_type.loc32_50 (constants.%pattern_type.9f3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45: <witness> = require_complete_type %A.binding.as_type [symbolic = %require_complete.loc32_45 (constants.%require_complete.c0f)]
// CHECK:STDOUT:   %require_complete.loc32_54: <witness> = require_complete_type %Food.binding.as_type [symbolic = %require_complete.loc32_54 (constants.%require_complete.74d)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc32_17.1, @Eats, @Eats(%Food.binding.as_type) [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats, @Eats(%Food.binding.as_type)> [symbolic = %Eats.type (constants.%Eats.type.98f93b.2)]
// CHECK:STDOUT:   %Eats.facet.loc32_76.2: @HandleAnimal.%Eats.type (%Eats.type.98f93b.2) = facet_value %A.binding.as_type, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.bda)]
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2: <specific function> = specific_function constants.%Feed, @Feed(%Food.loc32_29.1, %Eats.facet.loc32_76.2) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.8aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%A.binding.as_type (%A.binding.as_type), %food.param: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.86a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%A.binding.as_type (%A.binding.as_type) = name_ref a, %a
// CHECK:STDOUT:     %food.ref: @HandleAnimal.%Food.binding.as_type (%Food.binding.as_type.86a) = name_ref food, %food
// CHECK:STDOUT:     %.loc32_76.1: %Edible.type = converted constants.%Food.binding.as_type.86a, constants.%Food.e7a [symbolic = %Food.loc32_29.1 (constants.%Food.e7a)]
// CHECK:STDOUT:     %.loc32_76.2: %Edible.type = converted constants.%Food.binding.as_type.86a, constants.%Food.e7a [symbolic = %Food.loc32_29.1 (constants.%Food.e7a)]
// CHECK:STDOUT:     %Eats.facet.loc32_76.1: @HandleAnimal.%Eats.type (%Eats.type.98f93b.2) = facet_value constants.%A.binding.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.bda)]
// CHECK:STDOUT:     %.loc32_76.3: @HandleAnimal.%Eats.type (%Eats.type.98f93b.2) = converted constants.%A.binding.as_type, %Eats.facet.loc32_76.1 [symbolic = %Eats.facet.loc32_76.2 (constants.%Eats.facet.bda)]
// CHECK:STDOUT:     %Feed.specific_fn.loc32_64.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Food.e7a, constants.%Eats.facet.bda) [symbolic = %Feed.specific_fn.loc32_64.2 (constants.%Feed.specific_fn.8aa)]
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc35_29: <bound method> = bound_method %.loc35_29.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.325
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.325, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.c0f) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4fb]
// CHECK:STDOUT:   %bound_method.loc35_29: <bound method> = bound_method %.loc35_29.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc35_29: %ptr.2bd = addr_of %.loc35_29.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc35_29: init %empty_tuple.type = call %bound_method.loc35_29(%addr.loc35_29)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc35_17: <bound method> = bound_method %.loc35_17.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9fd
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9fd, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.8dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3f8]
// CHECK:STDOUT:   %bound_method.loc35_17: <bound method> = bound_method %.loc35_17.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc35_17: %ptr.940 = addr_of %.loc35_17.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc35_17: init %empty_tuple.type = call %bound_method.loc35_17(%addr.loc35_17)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.d9f) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%U.binding.as_type) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%U.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.98f93b.1
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.9eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%T.84c, constants.%U) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T.84c
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%U
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.5cd
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.98f93b.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cc38c8.1
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.42a0e2.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.binding.as_type.121) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.binding.as_type.121
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.3cc
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.7e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.17b, constants.%T.7e0) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.17b
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.121
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.3cc
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%T.7e0
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.67b
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.97d
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.3ce
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.25b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%A, constants.%Food.e7a) {
// CHECK:STDOUT:   %A.loc32_17.1 => constants.%A
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Food.e7a
// CHECK:STDOUT:   %A.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.01b
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.86a
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.9f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Food.binding.as_type.86a) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Food.binding.as_type.86a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%A, constants.%Food.e7a) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%A
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Food.e7a
// CHECK:STDOUT:   %T.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %U.binding.as_type => constants.%Food.binding.as_type.86a
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.98f93b.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cc38c8.2
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.42a0e2.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Food.e7a, constants.%Eats.facet.bda) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Food.e7a
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Food.binding.as_type.86a
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.98f93b.2
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.bda
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.5a9
// CHECK:STDOUT:   %T.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.01b
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.9f3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41 => constants.%require_complete.c0f
// CHECK:STDOUT:   %require_complete.loc31_50 => constants.%require_complete.74d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %A.loc32_17.1 => constants.%Animal.facet
// CHECK:STDOUT:   %Food.loc32_29.1 => constants.%Edible.facet
// CHECK:STDOUT:   %A.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc32_44 => constants.%pattern_type.ab7
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %pattern_type.loc32_50 => constants.%pattern_type.aff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32_45 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc32_54 => constants.%complete_type.357
// CHECK:STDOUT:   %Eats.lookup_impl_witness => constants.%Eats.impl_witness.7f7
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.cee
// CHECK:STDOUT:   %Eats.facet.loc32_76.2 => constants.%Eats.facet.179
// CHECK:STDOUT:   %Feed.specific_fn.loc32_64.2 => constants.%Feed.specific_fn.577
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats(constants.%Grass) {
// CHECK:STDOUT:   %Food.loc21_16.1 => constants.%Grass
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Eats.type => constants.%Eats.type.cee
// CHECK:STDOUT:   %Self.loc21_29.2 => constants.%Self.026
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Eats.impl(constants.%Animal.facet, constants.%Edible.facet) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%Animal.facet
// CHECK:STDOUT:   %U.loc26_26.2 => constants.%Edible.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %U.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc26_49.2 => constants.%Eats.type.cee
// CHECK:STDOUT:   %require_complete => constants.%complete_type.eba
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.7f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Edible.facet, constants.%Eats.facet.179) {
// CHECK:STDOUT:   %Food.loc31_9.1 => constants.%Edible.facet
// CHECK:STDOUT:   %Food.binding.as_type => constants.%Grass
// CHECK:STDOUT:   %Eats.type.loc31_37.1 => constants.%Eats.type.cee
// CHECK:STDOUT:   %T.loc31_24.1 => constants.%Eats.facet.179
// CHECK:STDOUT:   %pattern_type.loc31_24 => constants.%pattern_type.cf8
// CHECK:STDOUT:   %T.binding.as_type => constants.%Goat
// CHECK:STDOUT:   %pattern_type.loc31_40 => constants.%pattern_type.ab7
// CHECK:STDOUT:   %pattern_type.loc31_46 => constants.%pattern_type.aff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_41 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_50 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

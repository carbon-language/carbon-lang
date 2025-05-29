// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_to_narrowed_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_to_narrowed_facet_type.carbon

// --- convert_to_narrowed_facet_type.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}

fn Feed[T:! Eats](e: T) {}

fn HandleAnimal[U:! Animal & Eats](a: U) { Feed(a); }

// --- bigger.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}
interface Tame {}

fn FeedTame[V:! Tame & Eats](v: V) {}

fn HandleTameAnimal[W:! Eats & Animal & Tame](w: W) {
  FeedTame(w);
}

// --- with_blanket.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}
interface Tame {}

impl forall [A:! Animal] A as Eats {}

fn FeedTame2[V:! Tame & Eats](v: V) {}

fn HandleTameAnimal2[W:! Animal & Tame](w: W) {
  FeedTame2(w);
}

// --- equivalent.carbon
library "[[@TEST_NAME]]";

// This is testing `FindWitnessInFacet` from impl_lookup.cpp

interface A {}

fn TakesA[T:! A](x: T) {}

fn WithExtraWhere[U:! A where .Self impls type](y: U) {
  TakesA(y);
}

// --- no_interfaces_success.carbon
library "[[@TEST_NAME]]";

fn TakesTypeDeduced[T:! type](x: T) {}
fn CallsWithExtraWhere[U:! type where .Self impls type](y: U) {
  TakesTypeDeduced(y);
}

fn TakesTypeExplicit(T:! type) {}
fn CallsWithExtraWhereExplicit(U:! type where .Self impls type) {
  TakesTypeExplicit(U);
}

// --- no_interfaces.carbon
library "[[@TEST_NAME]]";

fn TakesExtraWhereDeduced[T:! type where .Self impls type](x: T) {}
fn CallsWithType[U:! type](y: U) {
  TakesExtraWhereDeduced(y);
}

fn TakesExtraWhereExplicit(T:! type where .Self impls type) {}
fn CallsWithTypeExplicit(U:! type) {
  TakesExtraWhereExplicit(U);
}

// CHECK:STDOUT: --- convert_to_narrowed_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.1b5: %Eats.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a05: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.1b5 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b4: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c75: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness.0e5: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness.0e5) [concrete]
// CHECK:STDOUT:   %.518: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %Animal.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %Animal.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Eats & @Animal> [concrete]
// CHECK:STDOUT:   %U: %facet_type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.787: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.56f: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.680: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %U, @Eats [symbolic]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %U.as_type, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.1e6), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a05 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.2b4) = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.2b4) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %T.loc6_9.1: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc6_22.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.2 [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc6_9.1 [symbolic = %T.loc6_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:       %T.as_type.loc6_22.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_22.2: type = converted %T.ref, %T.as_type.loc6_22.1 [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc6_22.2 (%T.as_type) = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %U.patt: %pattern_type.787 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.56f) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.56f) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_28.1: type = splice_block %.loc8_28.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %impl.elem0: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc8_28.1: <bound method> = bound_method %Animal.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_28.2: <bound method> = bound_method %Animal.ref, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %type.and: init type = call %bound_method.loc8_28.2(%Animal.ref, %Eats.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc8_28.2: type = value_of_initializer %type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc8_28.3: type = converted %type.and, %.loc8_28.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_17.1: %facet_type = bind_symbolic_name U, 0 [symbolic = %U.loc8_17.2 (constants.%U)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%U.as_type.loc8_39.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_39.1: type = splice_block %.loc8_39.2 [symbolic = %U.as_type.loc8_39.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type = name_ref U, %U.loc8_17.1 [symbolic = %U.loc8_17.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc8_39.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc8_39.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc8_39.2: type = converted %U.ref, %U.as_type.loc8_39.1 [symbolic = %U.as_type.loc8_39.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%U.as_type.loc8_39.2 (%U.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
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
// CHECK:STDOUT: generic fn @Feed(%T.loc6_9.1: %Eats.type) {
// CHECK:STDOUT:   %T.loc6_9.2: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:   %T.as_type.loc6_22.2: type = facet_access_type %T.loc6_9.2 [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc6_22.2 [symbolic = %pattern_type (constants.%pattern_type.2b4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_22.2 [symbolic = %require_complete (constants.%require_complete.c75)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc6_22.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%U.loc8_17.1: %facet_type) {
// CHECK:STDOUT:   %U.loc8_17.2: %facet_type = bind_symbolic_name U, 0 [symbolic = %U.loc8_17.2 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc8_39.2: type = facet_access_type %U.loc8_17.2 [symbolic = %U.as_type.loc8_39.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc8_39.2 [symbolic = %pattern_type (constants.%pattern_type.56f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc8_39.2 [symbolic = %require_complete (constants.%require_complete.680)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc8_17.2, @Eats [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc8_50.3: %Eats.type = facet_value %U.as_type.loc8_39.2, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc8_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:   %Feed.specific_fn.loc8_44.2: <specific function> = specific_function constants.%Feed, @Feed(%Eats.facet.loc8_50.3) [symbolic = %Feed.specific_fn.loc8_44.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%U.as_type.loc8_39.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%U.as_type.loc8_39.2 (%U.as_type) = name_ref a, %a
// CHECK:STDOUT:     %Eats.facet.loc8_50.1: %Eats.type = facet_value constants.%U.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc8_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc8_50.1: %Eats.type = converted constants.%U.as_type, %Eats.facet.loc8_50.1 [symbolic = %Eats.facet.loc8_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %Eats.facet.loc8_50.2: %Eats.type = facet_value constants.%U.as_type, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc8_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc8_50.2: %Eats.type = converted constants.%U.as_type, %Eats.facet.loc8_50.2 [symbolic = %Eats.facet.loc8_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %Feed.specific_fn.loc8_44.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [symbolic = %Feed.specific_fn.loc8_44.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc8_44.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.1b5) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T.1b5
// CHECK:STDOUT:   %T.as_type.loc6_22.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%U) {
// CHECK:STDOUT:   %U.loc8_17.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc8_39.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.56f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%Eats.facet
// CHECK:STDOUT:   %T.as_type.loc6_22.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.56f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.680
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bigger.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Tame.type: type = facet_type <@Tame> [concrete]
// CHECK:STDOUT:   %Self.7ee: %Tame.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness.0e5: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness.0e5) [concrete]
// CHECK:STDOUT:   %.518: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound.9f8: <bound method> = bound_method %Tame.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method.a56: <bound method> = bound_method %Tame.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.6ff: type = facet_type <@Eats & @Tame> [concrete]
// CHECK:STDOUT:   %V: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9f3: type = pattern_type %facet_type.6ff [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.760: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %FeedTame.type: type = fn_type @FeedTame [concrete]
// CHECK:STDOUT:   %FeedTame: %FeedTame.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.940: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %Op.bound.d46: <bound method> = bound_method %Eats.type, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.f8f: <bound method> = bound_method %Eats.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.c3f: type = facet_type <@Eats & @Animal> [concrete]
// CHECK:STDOUT:   %Op.bound.c0a: <bound method> = bound_method %facet_type.c3f, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.7fc: <bound method> = bound_method %facet_type.c3f, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.a95: type = facet_type <@Eats & @Animal & @Tame> [concrete]
// CHECK:STDOUT:   %W: %facet_type.a95 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1b: type = pattern_type %facet_type.a95 [concrete]
// CHECK:STDOUT:   %W.as_type: type = facet_access_type %W [symbolic]
// CHECK:STDOUT:   %pattern_type.002: type = pattern_type %W.as_type [symbolic]
// CHECK:STDOUT:   %HandleTameAnimal.type: type = fn_type @HandleTameAnimal [concrete]
// CHECK:STDOUT:   %HandleTameAnimal: %HandleTameAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.40f: <witness> = require_complete_type %W.as_type [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %W, @Eats [symbolic]
// CHECK:STDOUT:   %Tame.lookup_impl_witness: <witness> = lookup_impl_witness %W, @Tame [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.6ff = facet_value %W.as_type, (%Eats.lookup_impl_witness, %Tame.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %FeedTame.specific_fn: <specific function> = specific_function %FeedTame, @FeedTame(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.1e6), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Tame = %Tame.decl
// CHECK:STDOUT:     .FeedTame = %FeedTame.decl
// CHECK:STDOUT:     .HandleTameAnimal = %HandleTameAnimal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Tame.decl: type = interface_decl @Tame [concrete = constants.%Tame.type] {} {}
// CHECK:STDOUT:   %FeedTame.decl: %FeedTame.type = fn_decl @FeedTame [concrete = constants.%FeedTame] {
// CHECK:STDOUT:     %V.patt: %pattern_type.9f3 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %v.patt: @FeedTame.%pattern_type (%pattern_type.760) = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @FeedTame.%pattern_type (%pattern_type.760) = value_param_pattern %v.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_22.1: type = splice_block %.loc7_22.3 [concrete = constants.%facet_type.6ff] {
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %impl.elem0: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc7_22.1: <bound method> = bound_method %Tame.ref, %impl.elem0 [concrete = constants.%Op.bound.9f8]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc7_22.2: <bound method> = bound_method %Tame.ref, %specific_fn [concrete = constants.%bound_method.a56]
// CHECK:STDOUT:       %type.and: init type = call %bound_method.loc7_22.2(%Tame.ref, %Eats.ref) [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:       %.loc7_22.2: type = value_of_initializer %type.and [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:       %.loc7_22.3: type = converted %type.and, %.loc7_22.2 [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc7_13.1: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic = %V.loc7_13.2 (constants.%V)]
// CHECK:STDOUT:     %v.param: @FeedTame.%V.as_type.loc7_33.2 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_33.1: type = splice_block %.loc7_33.2 [symbolic = %V.as_type.loc7_33.2 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %facet_type.6ff = name_ref V, %V.loc7_13.1 [symbolic = %V.loc7_13.2 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc7_33.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc7_33.2 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc7_33.2: type = converted %V.ref, %V.as_type.loc7_33.1 [symbolic = %V.as_type.loc7_33.2 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @FeedTame.%V.as_type.loc7_33.2 (%V.as_type) = bind_name v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleTameAnimal.decl: %HandleTameAnimal.type = fn_decl @HandleTameAnimal [concrete = constants.%HandleTameAnimal] {
// CHECK:STDOUT:     %W.patt: %pattern_type.f1b = symbolic_binding_pattern W, 0 [concrete]
// CHECK:STDOUT:     %w.patt: @HandleTameAnimal.%pattern_type (%pattern_type.002) = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.param_patt: @HandleTameAnimal.%pattern_type (%pattern_type.002) = value_param_pattern %w.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_39.1: type = splice_block %.loc9_39.3 [concrete = constants.%facet_type.a95] {
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %impl.elem0.loc9_30: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc9_30.1: <bound method> = bound_method %Eats.ref, %impl.elem0.loc9_30 [concrete = constants.%Op.bound.d46]
// CHECK:STDOUT:       %specific_fn.loc9_30: <specific function> = specific_function %impl.elem0.loc9_30, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc9_30.2: <bound method> = bound_method %Eats.ref, %specific_fn.loc9_30 [concrete = constants.%bound_method.f8f]
// CHECK:STDOUT:       %type.and.loc9_30: init type = call %bound_method.loc9_30.2(%Eats.ref, %Animal.ref) [concrete = constants.%facet_type.c3f]
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %impl.elem0.loc9_39: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc9_39.1: <bound method> = bound_method %type.and.loc9_30, %impl.elem0.loc9_39 [concrete = constants.%Op.bound.c0a]
// CHECK:STDOUT:       %specific_fn.loc9_39: <specific function> = specific_function %impl.elem0.loc9_39, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc9_39.2: <bound method> = bound_method %type.and.loc9_30, %specific_fn.loc9_39 [concrete = constants.%bound_method.7fc]
// CHECK:STDOUT:       %.loc9_30.1: type = value_of_initializer %type.and.loc9_30 [concrete = constants.%facet_type.c3f]
// CHECK:STDOUT:       %.loc9_30.2: type = converted %type.and.loc9_30, %.loc9_30.1 [concrete = constants.%facet_type.c3f]
// CHECK:STDOUT:       %type.and.loc9_39: init type = call %bound_method.loc9_39.2(%.loc9_30.2, %Tame.ref) [concrete = constants.%facet_type.a95]
// CHECK:STDOUT:       %.loc9_39.2: type = value_of_initializer %type.and.loc9_39 [concrete = constants.%facet_type.a95]
// CHECK:STDOUT:       %.loc9_39.3: type = converted %type.and.loc9_39, %.loc9_39.2 [concrete = constants.%facet_type.a95]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc9_21.1: %facet_type.a95 = bind_symbolic_name W, 0 [symbolic = %W.loc9_21.2 (constants.%W)]
// CHECK:STDOUT:     %w.param: @HandleTameAnimal.%W.as_type.loc9_50.2 (%W.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_50.1: type = splice_block %.loc9_50.2 [symbolic = %W.as_type.loc9_50.2 (constants.%W.as_type)] {
// CHECK:STDOUT:       %W.ref: %facet_type.a95 = name_ref W, %W.loc9_21.1 [symbolic = %W.loc9_21.2 (constants.%W)]
// CHECK:STDOUT:       %W.as_type.loc9_50.1: type = facet_access_type %W.ref [symbolic = %W.as_type.loc9_50.2 (constants.%W.as_type)]
// CHECK:STDOUT:       %.loc9_50.2: type = converted %W.ref, %W.as_type.loc9_50.1 [symbolic = %W.as_type.loc9_50.2 (constants.%W.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: @HandleTameAnimal.%W.as_type.loc9_50.2 (%W.as_type) = bind_name w, %w.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
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
// CHECK:STDOUT: interface @Tame {
// CHECK:STDOUT:   %Self: %Tame.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedTame(%V.loc7_13.1: %facet_type.6ff) {
// CHECK:STDOUT:   %V.loc7_13.2: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic = %V.loc7_13.2 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc7_33.2: type = facet_access_type %V.loc7_13.2 [symbolic = %V.as_type.loc7_33.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc7_33.2 [symbolic = %pattern_type (constants.%pattern_type.760)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.as_type.loc7_33.2 [symbolic = %require_complete (constants.%require_complete.940)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%v.param: @FeedTame.%V.as_type.loc7_33.2 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleTameAnimal(%W.loc9_21.1: %facet_type.a95) {
// CHECK:STDOUT:   %W.loc9_21.2: %facet_type.a95 = bind_symbolic_name W, 0 [symbolic = %W.loc9_21.2 (constants.%W)]
// CHECK:STDOUT:   %W.as_type.loc9_50.2: type = facet_access_type %W.loc9_21.2 [symbolic = %W.as_type.loc9_50.2 (constants.%W.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %W.as_type.loc9_50.2 [symbolic = %pattern_type (constants.%pattern_type.002)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %W.as_type.loc9_50.2 [symbolic = %require_complete (constants.%require_complete.40f)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %W.loc9_21.2, @Eats [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Tame.lookup_impl_witness: <witness> = lookup_impl_witness %W.loc9_21.2, @Tame [symbolic = %Tame.lookup_impl_witness (constants.%Tame.lookup_impl_witness)]
// CHECK:STDOUT:   %facet_value.loc10_13.3: %facet_type.6ff = facet_value %W.as_type.loc9_50.2, (%Eats.lookup_impl_witness, %Tame.lookup_impl_witness) [symbolic = %facet_value.loc10_13.3 (constants.%facet_value)]
// CHECK:STDOUT:   %FeedTame.specific_fn.loc10_3.2: <specific function> = specific_function constants.%FeedTame, @FeedTame(%facet_value.loc10_13.3) [symbolic = %FeedTame.specific_fn.loc10_3.2 (constants.%FeedTame.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%w.param: @HandleTameAnimal.%W.as_type.loc9_50.2 (%W.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedTame.ref: %FeedTame.type = name_ref FeedTame, file.%FeedTame.decl [concrete = constants.%FeedTame]
// CHECK:STDOUT:     %w.ref: @HandleTameAnimal.%W.as_type.loc9_50.2 (%W.as_type) = name_ref w, %w
// CHECK:STDOUT:     %facet_value.loc10_13.1: %facet_type.6ff = facet_value constants.%W.as_type, (constants.%Eats.lookup_impl_witness, constants.%Tame.lookup_impl_witness) [symbolic = %facet_value.loc10_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc10_13.1: %facet_type.6ff = converted constants.%W.as_type, %facet_value.loc10_13.1 [symbolic = %facet_value.loc10_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %facet_value.loc10_13.2: %facet_type.6ff = facet_value constants.%W.as_type, (constants.%Eats.lookup_impl_witness, constants.%Tame.lookup_impl_witness) [symbolic = %facet_value.loc10_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc10_13.2: %facet_type.6ff = converted constants.%W.as_type, %facet_value.loc10_13.2 [symbolic = %facet_value.loc10_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %FeedTame.specific_fn.loc10_3.1: <specific function> = specific_function %FeedTame.ref, @FeedTame(constants.%facet_value) [symbolic = %FeedTame.specific_fn.loc10_3.2 (constants.%FeedTame.specific_fn)]
// CHECK:STDOUT:     %FeedTame.call: init %empty_tuple.type = call %FeedTame.specific_fn.loc10_3.1(%w.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame(constants.%V) {
// CHECK:STDOUT:   %V.loc7_13.2 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc7_33.2 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.760
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleTameAnimal(constants.%W) {
// CHECK:STDOUT:   %W.loc9_21.2 => constants.%W
// CHECK:STDOUT:   %W.as_type.loc9_50.2 => constants.%W.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.002
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame(constants.%facet_value) {
// CHECK:STDOUT:   %V.loc7_13.2 => constants.%facet_value
// CHECK:STDOUT:   %V.as_type.loc7_33.2 => constants.%W.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.40f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_blanket.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Tame.type: type = facet_type <@Tame> [concrete]
// CHECK:STDOUT:   %Self.7ee: %Tame.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b0: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.8ab: <witness> = impl_witness file.%Eats.impl_witness_table, @impl.e7b(%A) [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness.0e5: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness.0e5) [concrete]
// CHECK:STDOUT:   %.518: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound.9f8: <bound method> = bound_method %Tame.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method.a56: <bound method> = bound_method %Tame.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.6ff: type = facet_type <@Eats & @Tame> [concrete]
// CHECK:STDOUT:   %V: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9f3: type = pattern_type %facet_type.6ff [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.760: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %FeedTame2.type: type = fn_type @FeedTame2 [concrete]
// CHECK:STDOUT:   %FeedTame2: %FeedTame2.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.940: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %Op.bound.fe3: <bound method> = bound_method %Animal.type, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.a26: <bound method> = bound_method %Animal.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.65c: type = facet_type <@Animal & @Tame> [concrete]
// CHECK:STDOUT:   %W: %facet_type.65c = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6e8: type = pattern_type %facet_type.65c [concrete]
// CHECK:STDOUT:   %W.as_type: type = facet_access_type %W [symbolic]
// CHECK:STDOUT:   %pattern_type.5d1: type = pattern_type %W.as_type [symbolic]
// CHECK:STDOUT:   %HandleTameAnimal2.type: type = fn_type @HandleTameAnimal2 [concrete]
// CHECK:STDOUT:   %HandleTameAnimal2: %HandleTameAnimal2.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ba9: <witness> = require_complete_type %W.as_type [symbolic]
// CHECK:STDOUT:   %Animal.lookup_impl_witness: <witness> = lookup_impl_witness %W, @Animal [symbolic]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %W.as_type, (%Animal.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.e5d: <witness> = impl_witness file.%Eats.impl_witness_table, @impl.e7b(%Animal.facet) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %W, @Eats [symbolic]
// CHECK:STDOUT:   %Tame.lookup_impl_witness: <witness> = lookup_impl_witness %W, @Tame [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.6ff = facet_value %W.as_type, (%Eats.lookup_impl_witness, %Tame.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %FeedTame2.specific_fn: <specific function> = specific_function %FeedTame2, @FeedTame2(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.1e6), @impl.f92 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Tame = %Tame.decl
// CHECK:STDOUT:     .FeedTame2 = %FeedTame2.decl
// CHECK:STDOUT:     .HandleTameAnimal2 = %HandleTameAnimal2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Tame.decl: type = interface_decl @Tame [concrete = constants.%Tame.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.e7b [concrete] {
// CHECK:STDOUT:     %A.patt: %pattern_type.3b0 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %Animal.type = name_ref A, %A.loc7_14.1 [symbolic = %A.loc7_14.2 (constants.%A)]
// CHECK:STDOUT:     %A.as_type.loc7_26.1: type = facet_access_type %A.ref [symbolic = %A.as_type.loc7_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %.loc7: type = converted %A.ref, %A.as_type.loc7_26.1 [symbolic = %A.as_type.loc7_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %A.loc7_14.1: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc7_14.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (), @impl.e7b [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table, @impl.e7b(constants.%A) [symbolic = @impl.e7b.%Eats.impl_witness (constants.%Eats.impl_witness.8ab)]
// CHECK:STDOUT:   %FeedTame2.decl: %FeedTame2.type = fn_decl @FeedTame2 [concrete = constants.%FeedTame2] {
// CHECK:STDOUT:     %V.patt: %pattern_type.9f3 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %v.patt: @FeedTame2.%pattern_type (%pattern_type.760) = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @FeedTame2.%pattern_type (%pattern_type.760) = value_param_pattern %v.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %.loc9_23.3 [concrete = constants.%facet_type.6ff] {
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %impl.elem0: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc9_23.1: <bound method> = bound_method %Tame.ref, %impl.elem0 [concrete = constants.%Op.bound.9f8]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc9_23.2: <bound method> = bound_method %Tame.ref, %specific_fn [concrete = constants.%bound_method.a56]
// CHECK:STDOUT:       %type.and: init type = call %bound_method.loc9_23.2(%Tame.ref, %Eats.ref) [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:       %.loc9_23.2: type = value_of_initializer %type.and [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:       %.loc9_23.3: type = converted %type.and, %.loc9_23.2 [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc9_14.1: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic = %V.loc9_14.2 (constants.%V)]
// CHECK:STDOUT:     %v.param: @FeedTame2.%V.as_type.loc9_34.2 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_34.1: type = splice_block %.loc9_34.2 [symbolic = %V.as_type.loc9_34.2 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %facet_type.6ff = name_ref V, %V.loc9_14.1 [symbolic = %V.loc9_14.2 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc9_34.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc9_34.2 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc9_34.2: type = converted %V.ref, %V.as_type.loc9_34.1 [symbolic = %V.as_type.loc9_34.2 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @FeedTame2.%V.as_type.loc9_34.2 (%V.as_type) = bind_name v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleTameAnimal2.decl: %HandleTameAnimal2.type = fn_decl @HandleTameAnimal2 [concrete = constants.%HandleTameAnimal2] {
// CHECK:STDOUT:     %W.patt: %pattern_type.6e8 = symbolic_binding_pattern W, 0 [concrete]
// CHECK:STDOUT:     %w.patt: @HandleTameAnimal2.%pattern_type (%pattern_type.5d1) = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.param_patt: @HandleTameAnimal2.%pattern_type (%pattern_type.5d1) = value_param_pattern %w.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_33.1: type = splice_block %.loc11_33.3 [concrete = constants.%facet_type.65c] {
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %impl.elem0: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc11_33.1: <bound method> = bound_method %Animal.ref, %impl.elem0 [concrete = constants.%Op.bound.fe3]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc11_33.2: <bound method> = bound_method %Animal.ref, %specific_fn [concrete = constants.%bound_method.a26]
// CHECK:STDOUT:       %type.and: init type = call %bound_method.loc11_33.2(%Animal.ref, %Tame.ref) [concrete = constants.%facet_type.65c]
// CHECK:STDOUT:       %.loc11_33.2: type = value_of_initializer %type.and [concrete = constants.%facet_type.65c]
// CHECK:STDOUT:       %.loc11_33.3: type = converted %type.and, %.loc11_33.2 [concrete = constants.%facet_type.65c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc11_22.1: %facet_type.65c = bind_symbolic_name W, 0 [symbolic = %W.loc11_22.2 (constants.%W)]
// CHECK:STDOUT:     %w.param: @HandleTameAnimal2.%W.as_type.loc11_44.2 (%W.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_44.1: type = splice_block %.loc11_44.2 [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)] {
// CHECK:STDOUT:       %W.ref: %facet_type.65c = name_ref W, %W.loc11_22.1 [symbolic = %W.loc11_22.2 (constants.%W)]
// CHECK:STDOUT:       %W.as_type.loc11_44.1: type = facet_access_type %W.ref [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:       %.loc11_44.2: type = converted %W.ref, %W.as_type.loc11_44.1 [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: @HandleTameAnimal2.%W.as_type.loc11_44.2 (%W.as_type) = bind_name w, %w.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
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
// CHECK:STDOUT: interface @Tame {
// CHECK:STDOUT:   %Self: %Tame.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.e7b(%A.loc7_14.1: %Animal.type) {
// CHECK:STDOUT:   %A.loc7_14.2: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc7_14.2 (constants.%A)]
// CHECK:STDOUT:   %A.as_type.loc7_26.2: type = facet_access_type %A.loc7_14.2 [symbolic = %A.as_type.loc7_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table, @impl.e7b(%A.loc7_14.2) [symbolic = %Eats.impl_witness (constants.%Eats.impl_witness.8ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7 as %Eats.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Eats.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedTame2(%V.loc9_14.1: %facet_type.6ff) {
// CHECK:STDOUT:   %V.loc9_14.2: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic = %V.loc9_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc9_34.2: type = facet_access_type %V.loc9_14.2 [symbolic = %V.as_type.loc9_34.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc9_34.2 [symbolic = %pattern_type (constants.%pattern_type.760)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.as_type.loc9_34.2 [symbolic = %require_complete (constants.%require_complete.940)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%v.param: @FeedTame2.%V.as_type.loc9_34.2 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleTameAnimal2(%W.loc11_22.1: %facet_type.65c) {
// CHECK:STDOUT:   %W.loc11_22.2: %facet_type.65c = bind_symbolic_name W, 0 [symbolic = %W.loc11_22.2 (constants.%W)]
// CHECK:STDOUT:   %W.as_type.loc11_44.2: type = facet_access_type %W.loc11_22.2 [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %W.as_type.loc11_44.2 [symbolic = %pattern_type (constants.%pattern_type.5d1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %W.as_type.loc11_44.2 [symbolic = %require_complete (constants.%require_complete.ba9)]
// CHECK:STDOUT:   %Animal.lookup_impl_witness: <witness> = lookup_impl_witness %W.loc11_22.2, @Animal [symbolic = %Animal.lookup_impl_witness (constants.%Animal.lookup_impl_witness)]
// CHECK:STDOUT:   %Animal.facet.loc12_14.3: %Animal.type = facet_value %W.as_type.loc11_44.2, (%Animal.lookup_impl_witness) [symbolic = %Animal.facet.loc12_14.3 (constants.%Animal.facet)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %W.loc11_22.2, @Eats [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Tame.lookup_impl_witness: <witness> = lookup_impl_witness %W.loc11_22.2, @Tame [symbolic = %Tame.lookup_impl_witness (constants.%Tame.lookup_impl_witness)]
// CHECK:STDOUT:   %facet_value.loc12_14.3: %facet_type.6ff = facet_value %W.as_type.loc11_44.2, (%Eats.lookup_impl_witness, %Tame.lookup_impl_witness) [symbolic = %facet_value.loc12_14.3 (constants.%facet_value)]
// CHECK:STDOUT:   %FeedTame2.specific_fn.loc12_3.2: <specific function> = specific_function constants.%FeedTame2, @FeedTame2(%facet_value.loc12_14.3) [symbolic = %FeedTame2.specific_fn.loc12_3.2 (constants.%FeedTame2.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%w.param: @HandleTameAnimal2.%W.as_type.loc11_44.2 (%W.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedTame2.ref: %FeedTame2.type = name_ref FeedTame2, file.%FeedTame2.decl [concrete = constants.%FeedTame2]
// CHECK:STDOUT:     %w.ref: @HandleTameAnimal2.%W.as_type.loc11_44.2 (%W.as_type) = name_ref w, %w
// CHECK:STDOUT:     %W.as_type.loc12_14.1: type = facet_access_type constants.%W [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:     %.loc12_14.1: type = converted constants.%W, %W.as_type.loc12_14.1 [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:     %Animal.facet.loc12_14.1: %Animal.type = facet_value constants.%W.as_type, (constants.%Animal.lookup_impl_witness) [symbolic = %Animal.facet.loc12_14.3 (constants.%Animal.facet)]
// CHECK:STDOUT:     %.loc12_14.2: %Animal.type = converted %.loc12_14.1, %Animal.facet.loc12_14.1 [symbolic = %Animal.facet.loc12_14.3 (constants.%Animal.facet)]
// CHECK:STDOUT:     %facet_value.loc12_14.1: %facet_type.6ff = facet_value constants.%W.as_type, (constants.%Eats.lookup_impl_witness, constants.%Tame.lookup_impl_witness) [symbolic = %facet_value.loc12_14.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc12_14.3: %facet_type.6ff = converted constants.%W.as_type, %facet_value.loc12_14.1 [symbolic = %facet_value.loc12_14.3 (constants.%facet_value)]
// CHECK:STDOUT:     %W.as_type.loc12_14.2: type = facet_access_type constants.%W [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:     %.loc12_14.4: type = converted constants.%W, %W.as_type.loc12_14.2 [symbolic = %W.as_type.loc11_44.2 (constants.%W.as_type)]
// CHECK:STDOUT:     %Animal.facet.loc12_14.2: %Animal.type = facet_value constants.%W.as_type, (constants.%Animal.lookup_impl_witness) [symbolic = %Animal.facet.loc12_14.3 (constants.%Animal.facet)]
// CHECK:STDOUT:     %.loc12_14.5: %Animal.type = converted %.loc12_14.4, %Animal.facet.loc12_14.2 [symbolic = %Animal.facet.loc12_14.3 (constants.%Animal.facet)]
// CHECK:STDOUT:     %facet_value.loc12_14.2: %facet_type.6ff = facet_value constants.%W.as_type, (constants.%Eats.lookup_impl_witness, constants.%Tame.lookup_impl_witness) [symbolic = %facet_value.loc12_14.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc12_14.6: %facet_type.6ff = converted constants.%W.as_type, %facet_value.loc12_14.2 [symbolic = %facet_value.loc12_14.3 (constants.%facet_value)]
// CHECK:STDOUT:     %FeedTame2.specific_fn.loc12_3.1: <specific function> = specific_function %FeedTame2.ref, @FeedTame2(constants.%facet_value) [symbolic = %FeedTame2.specific_fn.loc12_3.2 (constants.%FeedTame2.specific_fn)]
// CHECK:STDOUT:     %FeedTame2.call: init %empty_tuple.type = call %FeedTame2.specific_fn.loc12_3.1(%w.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.e7b(constants.%A) {
// CHECK:STDOUT:   %A.loc7_14.2 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc7_26.2 => constants.%A.as_type
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.8ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame2(constants.%V) {
// CHECK:STDOUT:   %V.loc9_14.2 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc9_34.2 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.760
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleTameAnimal2(constants.%W) {
// CHECK:STDOUT:   %W.loc11_22.2 => constants.%W
// CHECK:STDOUT:   %W.as_type.loc11_44.2 => constants.%W.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.e7b(constants.%Animal.facet) {
// CHECK:STDOUT:   %A.loc7_14.2 => constants.%Animal.facet
// CHECK:STDOUT:   %A.as_type.loc7_26.2 => constants.%W.as_type
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.e5d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame2(constants.%facet_value) {
// CHECK:STDOUT:   %V.loc9_14.2 => constants.%facet_value
// CHECK:STDOUT:   %V.as_type.loc9_34.2 => constants.%W.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5d1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ba9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a2b: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.f9844e.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf45b7.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %U: %A.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.f9844e.2: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %WithExtraWhere.type: type = fn_type @WithExtraWhere [concrete]
// CHECK:STDOUT:   %WithExtraWhere: %WithExtraWhere.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf45b7.2: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %TakesA.specific_fn: <specific function> = specific_function %TakesA, @TakesA(%U) [symbolic]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .WithExtraWhere = %WithExtraWhere.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a2b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @TakesA.%pattern_type (%pattern_type.f9844e.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @TakesA.%pattern_type (%pattern_type.f9844e.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %T.loc7_11.1: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_11.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_21.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_21.1: type = splice_block %.loc7_21.2 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_11.1 [symbolic = %T.loc7_11.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_21.2: type = converted %T.ref, %T.as_type.loc7_21.1 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_21.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithExtraWhere.decl: %WithExtraWhere.type = fn_decl @WithExtraWhere [concrete = constants.%WithExtraWhere] {
// CHECK:STDOUT:     %U.patt: %pattern_type.a2b = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %y.patt: @WithExtraWhere.%pattern_type (%pattern_type.f9844e.2) = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @WithExtraWhere.%pattern_type (%pattern_type.f9844e.2) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_25.1: type = splice_block %.loc9_25.2 [concrete = constants.%A.type] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self: %A.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_31: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_25.2: type = where_expr %.Self [concrete = constants.%A.type] {
// CHECK:STDOUT:         requirement_impls %.loc9_31, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_19.1: %A.type = bind_symbolic_name U, 0 [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_52.1: type = splice_block %.loc9_52.2 [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %A.type = name_ref U, %U.loc9_19.1 [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc9_52.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc9_52.2: type = converted %U.ref, %U.as_type.loc9_52.1 [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_11.1: %A.type) {
// CHECK:STDOUT:   %T.loc7_11.2: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_11.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_21.2: type = facet_access_type %T.loc7_11.2 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_21.2 [symbolic = %pattern_type (constants.%pattern_type.f9844e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_21.2 [symbolic = %require_complete (constants.%require_complete.cf45b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.as_type.loc7_21.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithExtraWhere(%U.loc9_19.1: %A.type) {
// CHECK:STDOUT:   %U.loc9_19.2: %A.type = bind_symbolic_name U, 0 [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc9_52.2: type = facet_access_type %U.loc9_19.2 [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc9_52.2 [symbolic = %pattern_type (constants.%pattern_type.f9844e.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc9_52.2 [symbolic = %require_complete (constants.%require_complete.cf45b7.2)]
// CHECK:STDOUT:   %TakesA.specific_fn.loc10_3.2: <specific function> = specific_function constants.%TakesA, @TakesA(%U.loc9_19.2) [symbolic = %TakesA.specific_fn.loc10_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %y.ref: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = name_ref y, %y
// CHECK:STDOUT:     %.loc10_11.1: %A.type = converted constants.%U.as_type, constants.%U [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:     %.loc10_11.2: %A.type = converted constants.%U.as_type, constants.%U [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:     %TakesA.specific_fn.loc10_3.1: <specific function> = specific_function %TakesA.ref, @TakesA(constants.%U) [symbolic = %TakesA.specific_fn.loc10_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:     %TakesA.call: init %empty_tuple.type = call %TakesA.specific_fn.loc10_3.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_11.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f9844e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithExtraWhere(constants.%U) {
// CHECK:STDOUT:   %U.loc9_19.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc9_52.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f9844e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%U) {
// CHECK:STDOUT:   %T.loc7_11.2 => constants.%U
// CHECK:STDOUT:   %T.as_type.loc7_21.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f9844e.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cf45b7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_interfaces_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %TakesTypeDeduced.type: type = fn_type @TakesTypeDeduced [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesTypeDeduced: %TakesTypeDeduced.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %U: %type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.d06: type = pattern_type %type [concrete]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.772: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %CallsWithExtraWhere.type: type = fn_type @CallsWithExtraWhere [concrete]
// CHECK:STDOUT:   %CallsWithExtraWhere: %CallsWithExtraWhere.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5eb: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %TakesTypeDeduced.specific_fn: <specific function> = specific_function %TakesTypeDeduced, @TakesTypeDeduced(%U.as_type) [symbolic]
// CHECK:STDOUT:   %TakesTypeExplicit.type: type = fn_type @TakesTypeExplicit [concrete]
// CHECK:STDOUT:   %TakesTypeExplicit: %TakesTypeExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallsWithExtraWhereExplicit.type: type = fn_type @CallsWithExtraWhereExplicit [concrete]
// CHECK:STDOUT:   %CallsWithExtraWhereExplicit: %CallsWithExtraWhereExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakesTypeExplicit.specific_fn: <specific function> = specific_function %TakesTypeExplicit, @TakesTypeExplicit(%U.as_type) [symbolic]
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
// CHECK:STDOUT:     .TakesTypeDeduced = %TakesTypeDeduced.decl
// CHECK:STDOUT:     .CallsWithExtraWhere = %CallsWithExtraWhere.decl
// CHECK:STDOUT:     .TakesTypeExplicit = %TakesTypeExplicit.decl
// CHECK:STDOUT:     .CallsWithExtraWhereExplicit = %CallsWithExtraWhereExplicit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TakesTypeDeduced.decl: %TakesTypeDeduced.type = fn_decl @TakesTypeDeduced [concrete = constants.%TakesTypeDeduced] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @TakesTypeDeduced.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @TakesTypeDeduced.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_21.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesTypeDeduced.%T.loc3_21.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc3_21.1 [symbolic = %T.loc3_21.2 (constants.%T)]
// CHECK:STDOUT:     %x: @TakesTypeDeduced.%T.loc3_21.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithExtraWhere.decl: %CallsWithExtraWhere.type = fn_decl @CallsWithExtraWhere [concrete = constants.%CallsWithExtraWhere] {
// CHECK:STDOUT:     %U.patt: %pattern_type.d06 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %y.patt: @CallsWithExtraWhere.%pattern_type (%pattern_type.772) = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @CallsWithExtraWhere.%pattern_type (%pattern_type.772) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.2 [concrete = constants.%type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_33.2: type = where_expr %.Self [concrete = constants.%type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc4_24.1: %type = bind_symbolic_name U, 0 [symbolic = %U.loc4_24.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_60.1: type = splice_block %.loc4_60.2 [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %type = name_ref U, %U.loc4_24.1 [symbolic = %U.loc4_24.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc4_60.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc4_60.2: type = converted %U.ref, %U.as_type.loc4_60.1 [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesTypeExplicit.decl: %TakesTypeExplicit.type = fn_decl @TakesTypeExplicit [concrete = constants.%TakesTypeExplicit] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithExtraWhereExplicit.decl: %CallsWithExtraWhereExplicit.type = fn_decl @CallsWithExtraWhereExplicit [concrete = constants.%CallsWithExtraWhereExplicit] {
// CHECK:STDOUT:     %U.patt: %pattern_type.d06 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_41.1: type = splice_block %.loc9_41.2 [concrete = constants.%type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_41.2: type = where_expr %.Self [concrete = constants.%type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_32.1: %type = bind_symbolic_name U, 0 [symbolic = %U.loc9_32.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesTypeDeduced(%T.loc3_21.1: type) {
// CHECK:STDOUT:   %T.loc3_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_21.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc3_21.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc3_21.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesTypeDeduced.%T.loc3_21.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithExtraWhere(%U.loc4_24.1: %type) {
// CHECK:STDOUT:   %U.loc4_24.2: %type = bind_symbolic_name U, 0 [symbolic = %U.loc4_24.2 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc4_60.2: type = facet_access_type %U.loc4_24.2 [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc4_60.2 [symbolic = %pattern_type (constants.%pattern_type.772)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc4_60.2 [symbolic = %require_complete (constants.%require_complete.5eb)]
// CHECK:STDOUT:   %TakesTypeDeduced.specific_fn.loc5_3.2: <specific function> = specific_function constants.%TakesTypeDeduced, @TakesTypeDeduced(%U.as_type.loc4_60.2) [symbolic = %TakesTypeDeduced.specific_fn.loc5_3.2 (constants.%TakesTypeDeduced.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesTypeDeduced.ref: %TakesTypeDeduced.type = name_ref TakesTypeDeduced, file.%TakesTypeDeduced.decl [concrete = constants.%TakesTypeDeduced]
// CHECK:STDOUT:     %y.ref: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = name_ref y, %y
// CHECK:STDOUT:     %TakesTypeDeduced.specific_fn.loc5_3.1: <specific function> = specific_function %TakesTypeDeduced.ref, @TakesTypeDeduced(constants.%U.as_type) [symbolic = %TakesTypeDeduced.specific_fn.loc5_3.2 (constants.%TakesTypeDeduced.specific_fn)]
// CHECK:STDOUT:     %TakesTypeDeduced.call: init %empty_tuple.type = call %TakesTypeDeduced.specific_fn.loc5_3.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesTypeExplicit(%T.loc8_22.1: type) {
// CHECK:STDOUT:   %T.loc8_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithExtraWhereExplicit(%U.loc9_32.1: %type) {
// CHECK:STDOUT:   %U.loc9_32.2: %type = bind_symbolic_name U, 0 [symbolic = %U.loc9_32.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.as_type.loc10_22.2: type = facet_access_type %U.loc9_32.2 [symbolic = %U.as_type.loc10_22.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %TakesTypeExplicit.specific_fn.loc10_3.2: <specific function> = specific_function constants.%TakesTypeExplicit, @TakesTypeExplicit(%U.as_type.loc10_22.2) [symbolic = %TakesTypeExplicit.specific_fn.loc10_3.2 (constants.%TakesTypeExplicit.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesTypeExplicit.ref: %TakesTypeExplicit.type = name_ref TakesTypeExplicit, file.%TakesTypeExplicit.decl [concrete = constants.%TakesTypeExplicit]
// CHECK:STDOUT:     %U.ref: %type = name_ref U, %U.loc9_32.1 [symbolic = %U.loc9_32.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc10_22.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc10_22.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc10: type = converted %U.ref, %U.as_type.loc10_22.1 [symbolic = %U.as_type.loc10_22.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %TakesTypeExplicit.specific_fn.loc10_3.1: <specific function> = specific_function %TakesTypeExplicit.ref, @TakesTypeExplicit(constants.%U.as_type) [symbolic = %TakesTypeExplicit.specific_fn.loc10_3.2 (constants.%TakesTypeExplicit.specific_fn)]
// CHECK:STDOUT:     %TakesTypeExplicit.call: init %empty_tuple.type = call %TakesTypeExplicit.specific_fn.loc10_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc3_21.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithExtraWhere(constants.%U) {
// CHECK:STDOUT:   %U.loc4_24.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc4_60.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.772
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeDeduced(constants.%U.as_type) {
// CHECK:STDOUT:   %T.loc3_21.2 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.772
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeExplicit(constants.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithExtraWhereExplicit(constants.%U) {
// CHECK:STDOUT:   %U.loc9_32.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeExplicit(constants.%U.as_type) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T: %type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.d06: type = pattern_type %type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.772: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %TakesExtraWhereDeduced.type: type = fn_type @TakesExtraWhereDeduced [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesExtraWhereDeduced: %TakesExtraWhereDeduced.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5eb: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %CallsWithType.type: type = fn_type @CallsWithType [concrete]
// CHECK:STDOUT:   %CallsWithType: %CallsWithType.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %facet_value: %type = facet_value %U, () [symbolic]
// CHECK:STDOUT:   %TakesExtraWhereDeduced.specific_fn: <specific function> = specific_function %TakesExtraWhereDeduced, @TakesExtraWhereDeduced(%facet_value) [symbolic]
// CHECK:STDOUT:   %TakesExtraWhereExplicit.type: type = fn_type @TakesExtraWhereExplicit [concrete]
// CHECK:STDOUT:   %TakesExtraWhereExplicit: %TakesExtraWhereExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallsWithTypeExplicit.type: type = fn_type @CallsWithTypeExplicit [concrete]
// CHECK:STDOUT:   %CallsWithTypeExplicit: %CallsWithTypeExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakesExtraWhereExplicit.specific_fn: <specific function> = specific_function %TakesExtraWhereExplicit, @TakesExtraWhereExplicit(%facet_value) [symbolic]
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
// CHECK:STDOUT:     .TakesExtraWhereDeduced = %TakesExtraWhereDeduced.decl
// CHECK:STDOUT:     .CallsWithType = %CallsWithType.decl
// CHECK:STDOUT:     .TakesExtraWhereExplicit = %TakesExtraWhereExplicit.decl
// CHECK:STDOUT:     .CallsWithTypeExplicit = %CallsWithTypeExplicit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TakesExtraWhereDeduced.decl: %TakesExtraWhereDeduced.type = fn_decl @TakesExtraWhereDeduced [concrete = constants.%TakesExtraWhereDeduced] {
// CHECK:STDOUT:     %T.patt: %pattern_type.d06 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @TakesExtraWhereDeduced.%pattern_type (%pattern_type.772) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @TakesExtraWhereDeduced.%pattern_type (%pattern_type.772) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_36.1: type = splice_block %.loc3_36.2 [concrete = constants.%type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_36.2: type = where_expr %.Self [concrete = constants.%type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_27.1: %type = bind_symbolic_name T, 0 [symbolic = %T.loc3_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc3_63.1: type = splice_block %.loc3_63.2 [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %type = name_ref T, %T.loc3_27.1 [symbolic = %T.loc3_27.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc3_63.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc3_63.2: type = converted %T.ref, %T.as_type.loc3_63.1 [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithType.decl: %CallsWithType.type = fn_decl @CallsWithType [concrete = constants.%CallsWithType] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %y.patt: @CallsWithType.%pattern_type (%pattern_type.7dc) = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @CallsWithType.%pattern_type (%pattern_type.7dc) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc4_18.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_18.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @CallsWithType.%U.loc4_18.2 (%U) = value_param call_param0
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc4_18.1 [symbolic = %U.loc4_18.2 (constants.%U)]
// CHECK:STDOUT:     %y: @CallsWithType.%U.loc4_18.2 (%U) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesExtraWhereExplicit.decl: %TakesExtraWhereExplicit.type = fn_decl @TakesExtraWhereExplicit [concrete = constants.%TakesExtraWhereExplicit] {
// CHECK:STDOUT:     %T.patt: %pattern_type.d06 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_37.1: type = splice_block %.loc8_37.2 [concrete = constants.%type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_37.2: type = where_expr %.Self [concrete = constants.%type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_28.1: %type = bind_symbolic_name T, 0 [symbolic = %T.loc8_28.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithTypeExplicit.decl: %CallsWithTypeExplicit.type = fn_decl @CallsWithTypeExplicit [concrete = constants.%CallsWithTypeExplicit] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc9_26.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc9_26.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesExtraWhereDeduced(%T.loc3_27.1: %type) {
// CHECK:STDOUT:   %T.loc3_27.2: %type = bind_symbolic_name T, 0 [symbolic = %T.loc3_27.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc3_63.2: type = facet_access_type %T.loc3_27.2 [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc3_63.2 [symbolic = %pattern_type (constants.%pattern_type.772)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc3_63.2 [symbolic = %require_complete (constants.%require_complete.5eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithType(%U.loc4_18.1: type) {
// CHECK:STDOUT:   %U.loc4_18.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_18.2 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.loc4_18.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.loc4_18.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %facet_value.loc5_27.3: %type = facet_value %U.loc4_18.2, () [symbolic = %facet_value.loc5_27.3 (constants.%facet_value)]
// CHECK:STDOUT:   %TakesExtraWhereDeduced.specific_fn.loc5_3.2: <specific function> = specific_function constants.%TakesExtraWhereDeduced, @TakesExtraWhereDeduced(%facet_value.loc5_27.3) [symbolic = %TakesExtraWhereDeduced.specific_fn.loc5_3.2 (constants.%TakesExtraWhereDeduced.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @CallsWithType.%U.loc4_18.2 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesExtraWhereDeduced.ref: %TakesExtraWhereDeduced.type = name_ref TakesExtraWhereDeduced, file.%TakesExtraWhereDeduced.decl [concrete = constants.%TakesExtraWhereDeduced]
// CHECK:STDOUT:     %y.ref: @CallsWithType.%U.loc4_18.2 (%U) = name_ref y, %y
// CHECK:STDOUT:     %facet_value.loc5_27.1: %type = facet_value constants.%U, () [symbolic = %facet_value.loc5_27.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc5_27.1: %type = converted constants.%U, %facet_value.loc5_27.1 [symbolic = %facet_value.loc5_27.3 (constants.%facet_value)]
// CHECK:STDOUT:     %facet_value.loc5_27.2: %type = facet_value constants.%U, () [symbolic = %facet_value.loc5_27.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc5_27.2: %type = converted constants.%U, %facet_value.loc5_27.2 [symbolic = %facet_value.loc5_27.3 (constants.%facet_value)]
// CHECK:STDOUT:     %TakesExtraWhereDeduced.specific_fn.loc5_3.1: <specific function> = specific_function %TakesExtraWhereDeduced.ref, @TakesExtraWhereDeduced(constants.%facet_value) [symbolic = %TakesExtraWhereDeduced.specific_fn.loc5_3.2 (constants.%TakesExtraWhereDeduced.specific_fn)]
// CHECK:STDOUT:     %TakesExtraWhereDeduced.call: init %empty_tuple.type = call %TakesExtraWhereDeduced.specific_fn.loc5_3.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesExtraWhereExplicit(%T.loc8_28.1: %type) {
// CHECK:STDOUT:   %T.loc8_28.2: %type = bind_symbolic_name T, 0 [symbolic = %T.loc8_28.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithTypeExplicit(%U.loc9_26.1: type) {
// CHECK:STDOUT:   %U.loc9_26.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc9_26.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %facet_value.loc10_28.2: %type = facet_value %U.loc9_26.2, () [symbolic = %facet_value.loc10_28.2 (constants.%facet_value)]
// CHECK:STDOUT:   %TakesExtraWhereExplicit.specific_fn.loc10_3.2: <specific function> = specific_function constants.%TakesExtraWhereExplicit, @TakesExtraWhereExplicit(%facet_value.loc10_28.2) [symbolic = %TakesExtraWhereExplicit.specific_fn.loc10_3.2 (constants.%TakesExtraWhereExplicit.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesExtraWhereExplicit.ref: %TakesExtraWhereExplicit.type = name_ref TakesExtraWhereExplicit, file.%TakesExtraWhereExplicit.decl [concrete = constants.%TakesExtraWhereExplicit]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc9_26.1 [symbolic = %U.loc9_26.2 (constants.%U)]
// CHECK:STDOUT:     %facet_value.loc10_28.1: %type = facet_value constants.%U, () [symbolic = %facet_value.loc10_28.2 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc10: %type = converted %U.ref, %facet_value.loc10_28.1 [symbolic = %facet_value.loc10_28.2 (constants.%facet_value)]
// CHECK:STDOUT:     %TakesExtraWhereExplicit.specific_fn.loc10_3.1: <specific function> = specific_function %TakesExtraWhereExplicit.ref, @TakesExtraWhereExplicit(constants.%facet_value) [symbolic = %TakesExtraWhereExplicit.specific_fn.loc10_3.2 (constants.%TakesExtraWhereExplicit.specific_fn)]
// CHECK:STDOUT:     %TakesExtraWhereExplicit.call: init %empty_tuple.type = call %TakesExtraWhereExplicit.specific_fn.loc10_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesExtraWhereDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc3_27.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc3_63.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.772
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithType(constants.%U) {
// CHECK:STDOUT:   %U.loc4_18.2 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesExtraWhereDeduced(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc3_27.2 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc3_63.2 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesExtraWhereExplicit(constants.%T) {
// CHECK:STDOUT:   %T.loc8_28.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithTypeExplicit(constants.%U) {
// CHECK:STDOUT:   %U.loc9_26.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesExtraWhereExplicit(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc8_28.2 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

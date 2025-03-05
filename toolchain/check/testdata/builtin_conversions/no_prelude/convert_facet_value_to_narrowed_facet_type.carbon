// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_facet_value_to_narrowed_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_facet_value_to_narrowed_facet_type.carbon

// --- core.carbon
package Core;

interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}

impl forall [T:! type] T as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}

// --- convert_to_narrowed_facet_type.carbon
library "[[@TEST_NAME]]";

import Core;

interface Eats {}
interface Animal {}

fn Feed[T:! Eats](e: T) {}

fn HandleAnimal[U:! Animal & Eats](a: U) { Feed(a); }

// --- bigger.carbon
library "[[@TEST_NAME]]";

import Core;

interface Eats {}
interface Animal {}
interface Tame {}

fn FeedTame[V:! Tame & Eats](v: V) {}

fn HandleTameAnimal[W:! Eats & Animal & Tame](w: W) {
  FeedTame(w);
}

// --- with_blanket.carbon
library "[[@TEST_NAME]]";

import Core;

interface Eats {}
interface Animal {}
interface Tame {}

impl forall [A:! Animal] A as Eats {}

fn FeedTame2[V:! Tame & Eats](v: V) {}

fn HandleTameAnimal2[W:! Animal & Tame](w: W) {
  FeedTame2(w);
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc4_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc4_37: type = facet_access_type %Self.ref.loc4_37 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc4_37: type = converted %Self.ref.loc4_37, %Self.as_type.loc4_37 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc4_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_15.2: type = facet_access_type %Self.ref.loc4_15 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref.loc4_15, %Self.as_type.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc4_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_28: type = facet_access_type %Self.ref.loc4_28 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_28.2: type = converted %Self.ref.loc4_28, %Self.as_type.loc4_28 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc7_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc7_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc8_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc8_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc8_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type)) -> @Op.1.%Self.as_type.loc4_15.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_to_narrowed_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.1b5: %Eats.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6be: %Eats.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.1b5 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c75: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.1, @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.1, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, %impl_witness.3ea [concrete]
// CHECK:STDOUT:   %.d4d: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %Animal.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.bound, @Op.1(type) [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Eats & @Animal> [concrete]
// CHECK:STDOUT:   %U: %facet_type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %facet_type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.680: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %U.as_wit: <witness> = facet_access_witness %U [symbolic]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %U.as_type, %U.as_wit [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//default, loc4_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//default, loc7_36, loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc7_14, loaded [symbolic = @impl.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//default, loc7_24, loaded [symbolic = @impl.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//default, loc7_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.%Op.type (%Op.type.f99) = import_ref Core//default, loc8_42, loaded [symbolic = @impl.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc7_14, loaded [symbolic = @impl.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//default, inst15 [no loc], loaded [symbolic = constants.%Self.25f]
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
// CHECK:STDOUT:     %T.patt.loc8_9.1: %Eats.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt.6be)]
// CHECK:STDOUT:     %e.patt: @Feed.%T.as_type.loc8_22.2 (%T.as_type) = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: @Feed.%T.as_type.loc8_22.2 (%T.as_type) = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %T.loc8_9.1: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc8_22.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_22.1: type = splice_block %.loc8_22.2 [symbolic = %T.as_type.loc8_22.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc8_9.1 [symbolic = %T.loc8_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:       %T.as_type.loc8_22.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_22.2: type = converted %T.ref, %T.as_type.loc8_22.1 [symbolic = %T.as_type.loc8_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc8_22.2 (%T.as_type) = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %U.patt.loc10_17.1: %facet_type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc10_17.2 (constants.%U.patt)]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_28.1: type = splice_block %.loc10_28.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %impl.elem0: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %Animal.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %bound_method, @Op.1(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %type.and: init type = call %specific_fn(%Animal.ref, %Eats.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc10_28.2: type = value_of_initializer %type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc10_28.3: type = converted %type.and, %.loc10_28.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_17.1: %facet_type = bind_symbolic_name U, 0 [symbolic = %U.loc10_17.2 (constants.%U)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc10_39.1: type = splice_block %.loc10_39.2 [symbolic = %U.as_type.loc10_39.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type = name_ref U, %U.loc10_17.1 [symbolic = %U.loc10_17.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc10_39.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc10_39.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc10_39.2: type = converted %U.ref, %U.as_type.loc10_39.1 [symbolic = %U.as_type.loc10_39.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type) = bind_name a, %a.param
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
// CHECK:STDOUT: interface @BitAnd [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @impl(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T (%T.8b3) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc8_9.1: %Eats.type) {
// CHECK:STDOUT:   %T.loc8_9.2: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:   %T.patt.loc8_9.2: %Eats.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt.6be)]
// CHECK:STDOUT:   %T.as_type.loc8_22.2: type = facet_access_type %T.loc8_9.2 [symbolic = %T.as_type.loc8_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Feed.%T.as_type.loc8_22.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.c75)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_9.1: %Eats.type](%e.param_patt: @Feed.%T.as_type.loc8_22.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.5ab3ec.2: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%T (%T.8b3)](%other.param_patt: @Op.1.%T (%T.8b3)) -> @Op.1.%T (%T.8b3) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.040: %BitAnd.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.2.%Self.as_type (%Self.as_type)) -> @Op.2.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%U.loc10_17.1: %facet_type) {
// CHECK:STDOUT:   %U.loc10_17.2: %facet_type = bind_symbolic_name U, 0 [symbolic = %U.loc10_17.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc10_17.2: %facet_type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc10_17.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.as_type.loc10_39.2: type = facet_access_type %U.loc10_17.2 [symbolic = %U.as_type.loc10_39.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type) [symbolic = %require_complete (constants.%require_complete.680)]
// CHECK:STDOUT:   %U.as_wit.loc10_50.3: <witness> = facet_access_witness %U.loc10_17.2 [symbolic = %U.as_wit.loc10_50.3 (constants.%U.as_wit)]
// CHECK:STDOUT:   %Eats.facet.loc10_50.3: %Eats.type = facet_value %U.as_type.loc10_39.2, %U.as_wit.loc10_50.3 [symbolic = %Eats.facet.loc10_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:   %Feed.specific_fn.loc10_44.2: <specific function> = specific_function constants.%Feed, @Feed(%Eats.facet.loc10_50.3) [symbolic = %Feed.specific_fn.loc10_44.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc10_17.1: %facet_type](%a.param_patt: @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%U.as_type.loc10_39.2 (%U.as_type) = name_ref a, %a
// CHECK:STDOUT:     %U.as_wit.loc10_50.1: <witness> = facet_access_witness constants.%U [symbolic = %U.as_wit.loc10_50.3 (constants.%U.as_wit)]
// CHECK:STDOUT:     %Eats.facet.loc10_50.1: %Eats.type = facet_value constants.%U.as_type, %U.as_wit.loc10_50.1 [symbolic = %Eats.facet.loc10_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc10_50.1: %Eats.type = converted constants.%U.as_type, %Eats.facet.loc10_50.1 [symbolic = %Eats.facet.loc10_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %U.as_wit.loc10_50.2: <witness> = facet_access_witness constants.%U [symbolic = %U.as_wit.loc10_50.3 (constants.%U.as_wit)]
// CHECK:STDOUT:     %Eats.facet.loc10_50.2: %Eats.type = facet_value constants.%U.as_type, %U.as_wit.loc10_50.2 [symbolic = %Eats.facet.loc10_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc10_50.2: %Eats.type = converted constants.%U.as_type, %Eats.facet.loc10_50.2 [symbolic = %Eats.facet.loc10_50.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %Feed.specific_fn.loc10_44.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [symbolic = %Feed.specific_fn.loc10_44.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc10_44.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.1b5) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%T.1b5
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T.1b5
// CHECK:STDOUT:   %T.as_type.loc8_22.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%U) {
// CHECK:STDOUT:   %U.loc10_17.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc10_17.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc10_39.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%Eats.facet
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%Eats.facet
// CHECK:STDOUT:   %T.as_type.loc8_22.2 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.680
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(@HandleAnimal.%Eats.facet.loc10_50.3) {}
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
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.1, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.1, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, %impl_witness.3ea [concrete]
// CHECK:STDOUT:   %.d4d: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound.9f8: <bound method> = bound_method %Tame.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.163: <specific function> = specific_function %Op.bound.9f8, @Op.1(type) [concrete]
// CHECK:STDOUT:   %facet_type.6ff: type = facet_type <@Eats & @Tame> [concrete]
// CHECK:STDOUT:   %V: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %facet_type.6ff = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %FeedTame.type: type = fn_type @FeedTame [concrete]
// CHECK:STDOUT:   %FeedTame: %FeedTame.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.940: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %Op.bound.d46: <bound method> = bound_method %Eats.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8e0: <specific function> = specific_function %Op.bound.d46, @Op.1(type) [concrete]
// CHECK:STDOUT:   %facet_type.c3f: type = facet_type <@Eats & @Animal> [concrete]
// CHECK:STDOUT:   %Op.bound.c0a: <bound method> = bound_method %facet_type.c3f, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.394: <specific function> = specific_function %Op.bound.c0a, @Op.1(type) [concrete]
// CHECK:STDOUT:   %facet_type.a95: type = facet_type <@Eats & @Animal & @Tame> [concrete]
// CHECK:STDOUT:   %W: %facet_type.a95 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %facet_type.a95 = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %W.as_type: type = facet_access_type %W [symbolic]
// CHECK:STDOUT:   %HandleTameAnimal.type: type = fn_type @HandleTameAnimal [concrete]
// CHECK:STDOUT:   %HandleTameAnimal: %HandleTameAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.40f: <witness> = require_complete_type %W.as_type [symbolic]
// CHECK:STDOUT:   %W.as_wit: <witness> = facet_access_witness %W [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.6ff = facet_value %W.as_type, %W.as_wit [symbolic]
// CHECK:STDOUT:   %FeedTame.specific_fn: <specific function> = specific_function %FeedTame, @FeedTame(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//default, loc4_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//default, loc7_36, loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc7_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//default, loc7_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//default, loc7_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.%Op.type (%Op.type.f99) = import_ref Core//default, loc8_42, loaded [symbolic = @impl.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc7_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//default, inst15 [no loc], loaded [symbolic = constants.%Self.25f]
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
// CHECK:STDOUT:     %V.patt.loc9_13.1: %facet_type.6ff = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_13.2 (constants.%V.patt)]
// CHECK:STDOUT:     %v.patt: @FeedTame.%V.as_type.loc9_33.2 (%V.as_type) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @FeedTame.%V.as_type.loc9_33.2 (%V.as_type) = value_param_pattern %v.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.3 [concrete = constants.%facet_type.6ff] {
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %impl.elem0: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %Tame.ref, %impl.elem0 [concrete = constants.%Op.bound.9f8]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %bound_method, @Op.1(type) [concrete = constants.%Op.specific_fn.163]
// CHECK:STDOUT:       %type.and: init type = call %specific_fn(%Tame.ref, %Eats.ref) [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:       %.loc9_22.2: type = value_of_initializer %type.and [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:       %.loc9_22.3: type = converted %type.and, %.loc9_22.2 [concrete = constants.%facet_type.6ff]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc9_13.1: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic = %V.loc9_13.2 (constants.%V)]
// CHECK:STDOUT:     %v.param: @FeedTame.%V.as_type.loc9_33.2 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_33.1: type = splice_block %.loc9_33.2 [symbolic = %V.as_type.loc9_33.2 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %facet_type.6ff = name_ref V, %V.loc9_13.1 [symbolic = %V.loc9_13.2 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc9_33.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc9_33.2 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc9_33.2: type = converted %V.ref, %V.as_type.loc9_33.1 [symbolic = %V.as_type.loc9_33.2 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @FeedTame.%V.as_type.loc9_33.2 (%V.as_type) = bind_name v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleTameAnimal.decl: %HandleTameAnimal.type = fn_decl @HandleTameAnimal [concrete = constants.%HandleTameAnimal] {
// CHECK:STDOUT:     %W.patt.loc11_21.1: %facet_type.a95 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_21.2 (constants.%W.patt)]
// CHECK:STDOUT:     %w.patt: @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type) = binding_pattern w
// CHECK:STDOUT:     %w.param_patt: @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type) = value_param_pattern %w.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_39.1: type = splice_block %.loc11_39.3 [concrete = constants.%facet_type.a95] {
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %impl.elem0.loc11_30: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc11_30: <bound method> = bound_method %Eats.ref, %impl.elem0.loc11_30 [concrete = constants.%Op.bound.d46]
// CHECK:STDOUT:       %specific_fn.loc11_30: <specific function> = specific_function %bound_method.loc11_30, @Op.1(type) [concrete = constants.%Op.specific_fn.8e0]
// CHECK:STDOUT:       %type.and.loc11_30: init type = call %specific_fn.loc11_30(%Eats.ref, %Animal.ref) [concrete = constants.%facet_type.c3f]
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %impl.elem0.loc11_39: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc11_39: <bound method> = bound_method %type.and.loc11_30, %impl.elem0.loc11_39 [concrete = constants.%Op.bound.c0a]
// CHECK:STDOUT:       %specific_fn.loc11_39: <specific function> = specific_function %bound_method.loc11_39, @Op.1(type) [concrete = constants.%Op.specific_fn.394]
// CHECK:STDOUT:       %.loc11_30.1: type = value_of_initializer %type.and.loc11_30 [concrete = constants.%facet_type.c3f]
// CHECK:STDOUT:       %.loc11_30.2: type = converted %type.and.loc11_30, %.loc11_30.1 [concrete = constants.%facet_type.c3f]
// CHECK:STDOUT:       %type.and.loc11_39: init type = call %specific_fn.loc11_39(%.loc11_30.2, %Tame.ref) [concrete = constants.%facet_type.a95]
// CHECK:STDOUT:       %.loc11_39.2: type = value_of_initializer %type.and.loc11_39 [concrete = constants.%facet_type.a95]
// CHECK:STDOUT:       %.loc11_39.3: type = converted %type.and.loc11_39, %.loc11_39.2 [concrete = constants.%facet_type.a95]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc11_21.1: %facet_type.a95 = bind_symbolic_name W, 0 [symbolic = %W.loc11_21.2 (constants.%W)]
// CHECK:STDOUT:     %w.param: @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_50.1: type = splice_block %.loc11_50.2 [symbolic = %W.as_type.loc11_50.2 (constants.%W.as_type)] {
// CHECK:STDOUT:       %W.ref: %facet_type.a95 = name_ref W, %W.loc11_21.1 [symbolic = %W.loc11_21.2 (constants.%W)]
// CHECK:STDOUT:       %W.as_type.loc11_50.1: type = facet_access_type %W.ref [symbolic = %W.as_type.loc11_50.2 (constants.%W.as_type)]
// CHECK:STDOUT:       %.loc11_50.2: type = converted %W.ref, %W.as_type.loc11_50.1 [symbolic = %W.as_type.loc11_50.2 (constants.%W.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type) = bind_name w, %w.param
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
// CHECK:STDOUT: interface @BitAnd [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @impl(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.5ab3ec.2: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%T (%T)](%other.param_patt: @Op.1.%T (%T)) -> @Op.1.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.040: %BitAnd.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.2.%Self.as_type (%Self.as_type)) -> @Op.2.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedTame(%V.loc9_13.1: %facet_type.6ff) {
// CHECK:STDOUT:   %V.loc9_13.2: %facet_type.6ff = bind_symbolic_name V, 0 [symbolic = %V.loc9_13.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc9_13.2: %facet_type.6ff = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_13.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.as_type.loc9_33.2: type = facet_access_type %V.loc9_13.2 [symbolic = %V.as_type.loc9_33.2 (constants.%V.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @FeedTame.%V.as_type.loc9_33.2 (%V.as_type) [symbolic = %require_complete (constants.%require_complete.940)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%V.patt.loc9_13.1: %facet_type.6ff](%v.param_patt: @FeedTame.%V.as_type.loc9_33.2 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleTameAnimal(%W.loc11_21.1: %facet_type.a95) {
// CHECK:STDOUT:   %W.loc11_21.2: %facet_type.a95 = bind_symbolic_name W, 0 [symbolic = %W.loc11_21.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_21.2: %facet_type.a95 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_21.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.as_type.loc11_50.2: type = facet_access_type %W.loc11_21.2 [symbolic = %W.as_type.loc11_50.2 (constants.%W.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type) [symbolic = %require_complete (constants.%require_complete.40f)]
// CHECK:STDOUT:   %W.as_wit.loc12_13.5: <witness> = facet_access_witness %W.loc11_21.2 [symbolic = %W.as_wit.loc12_13.5 (constants.%W.as_wit)]
// CHECK:STDOUT:   %facet_value.loc12_13.3: %facet_type.6ff = facet_value %W.as_type.loc11_50.2, %W.as_wit.loc12_13.5 [symbolic = %facet_value.loc12_13.3 (constants.%facet_value)]
// CHECK:STDOUT:   %FeedTame.specific_fn.loc12_3.2: <specific function> = specific_function constants.%FeedTame, @FeedTame(%facet_value.loc12_13.3) [symbolic = %FeedTame.specific_fn.loc12_3.2 (constants.%FeedTame.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.patt.loc11_21.1: %facet_type.a95](%w.param_patt: @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedTame.ref: %FeedTame.type = name_ref FeedTame, file.%FeedTame.decl [concrete = constants.%FeedTame]
// CHECK:STDOUT:     %w.ref: @HandleTameAnimal.%W.as_type.loc11_50.2 (%W.as_type) = name_ref w, %w
// CHECK:STDOUT:     %W.as_wit.loc12_13.1: <witness> = facet_access_witness constants.%W [symbolic = %W.as_wit.loc12_13.5 (constants.%W.as_wit)]
// CHECK:STDOUT:     %W.as_wit.loc12_13.2: <witness> = facet_access_witness constants.%W [symbolic = %W.as_wit.loc12_13.5 (constants.%W.as_wit)]
// CHECK:STDOUT:     %facet_value.loc12_13.1: %facet_type.6ff = facet_value constants.%W.as_type, %W.as_wit.loc12_13.1 [symbolic = %facet_value.loc12_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc12_13.1: %facet_type.6ff = converted constants.%W.as_type, %facet_value.loc12_13.1 [symbolic = %facet_value.loc12_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %W.as_wit.loc12_13.3: <witness> = facet_access_witness constants.%W [symbolic = %W.as_wit.loc12_13.5 (constants.%W.as_wit)]
// CHECK:STDOUT:     %W.as_wit.loc12_13.4: <witness> = facet_access_witness constants.%W [symbolic = %W.as_wit.loc12_13.5 (constants.%W.as_wit)]
// CHECK:STDOUT:     %facet_value.loc12_13.2: %facet_type.6ff = facet_value constants.%W.as_type, %W.as_wit.loc12_13.3 [symbolic = %facet_value.loc12_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc12_13.2: %facet_type.6ff = converted constants.%W.as_type, %facet_value.loc12_13.2 [symbolic = %facet_value.loc12_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %FeedTame.specific_fn.loc12_3.1: <specific function> = specific_function %FeedTame.ref, @FeedTame(constants.%facet_value) [symbolic = %FeedTame.specific_fn.loc12_3.2 (constants.%FeedTame.specific_fn)]
// CHECK:STDOUT:     %FeedTame.call: init %empty_tuple.type = call %FeedTame.specific_fn.loc12_3.1(%w.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame(constants.%V) {
// CHECK:STDOUT:   %V.loc9_13.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc9_13.2 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc9_33.2 => constants.%V.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleTameAnimal(constants.%W) {
// CHECK:STDOUT:   %W.loc11_21.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_21.2 => constants.%W
// CHECK:STDOUT:   %W.as_type.loc11_50.2 => constants.%W.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame(constants.%facet_value) {
// CHECK:STDOUT:   %V.loc9_13.2 => constants.%facet_value
// CHECK:STDOUT:   %V.patt.loc9_13.2 => constants.%facet_value
// CHECK:STDOUT:   %V.as_type.loc9_33.2 => constants.%W.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.40f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame(@HandleTameAnimal.%facet_value.loc12_13.3) {}
// CHECK:STDOUT:

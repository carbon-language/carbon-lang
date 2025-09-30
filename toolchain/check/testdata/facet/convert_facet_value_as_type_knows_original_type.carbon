// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_facet_value_as_type_knows_original_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_facet_value_as_type_knows_original_type.carbon

// --- explicit_as_type.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}

class Goat {}
impl Goat as Animal {}
impl Goat as Eats {}

fn Feed(e:! Eats) {}

fn F() {
  //@dump-sem-ir-begin
  Feed((Goat as Animal) as type);
  //@dump-sem-ir-end
}

// --- facet_type_in_type_position.carbon
library "[[@TEST_NAME]]";

interface Eats {
  fn Eat();
}
interface Animal {}

class Goat {
  fn Bleet() {}

  impl as Animal {}
  extend impl as Eats {
    fn Eat() {}
  }
}

fn F() {
  // `Goat as Animal` in the type position retains/recovers the original type
  // Goat, so member lookup can see more than just `Animal`.

  //@dump-sem-ir-begin
  let x: Goat as Animal = {} as Goat;
  x.Bleet();
  x.Eat();

  (({} as Goat) as (Goat as Animal)).Bleet();
  (({} as Goat) as (Goat as Animal)).Eat();
  //@dump-sem-ir-end
}

// --- facet_access_type_converts_back_to_original_facet_value.carbon
library "[[@TEST_NAME]]";

interface J {}

class C(A:! J, B:! A) {}

// The class C is holding a specific with a 2 BindSymbolicNames where one is a
// *symbolic* type of the other.
//
// This means deduce will determine the argument type for `A`, then `B`. Then it
// will traverse into deducing the type of `B` with the type of the parameter
// `B`, which are both `FacetAccessType(A)`. This ultimately tries to deduce the
// value for `A` again, so it needs to match, even though the argument is now
// `FacetAccessType(A)`.
//
// This depends on unwrapping the `FacetAccessType(A)` when converting it to the
// type of `A`, instead of evaluating to a `FacetValue` holding the
// `FacetAccessType(A)` with a witness from the type of `A`. This preserves the
// original facet value instead of constructing a different facet value
// referring to the same symbolic type and with the same facet type.
// Essentially, it allows lossless round trips through a FacetAccessType when
// converting back to the type of its original facet value.
fn G[A:! J, B:! A](x: C(A, B)) {}

fn F[A:! J, B:! A](x: C(A, B)) {
  //@dump-sem-ir-begin
  G(x);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- explicit_as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness file.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table [concrete]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%Eats.impl_witness) [concrete]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat.ref, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc14_14: %Animal.type = converted %Goat.ref, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc14_14 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc14_25: type = converted %.loc14_14, %as_type [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %.loc14_25, (constants.%Eats.impl_witness) [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %.loc14_32: %Eats.type = converted %.loc14_25, %Eats.facet [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [concrete = constants.%Feed.specific_fn]
// CHECK:STDOUT:   %Feed.call: init %empty_tuple.type = call %Feed.specific_fn()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_type_in_type_position.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Eats.Eat.type: type = fn_type @Eats.Eat [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.assoc_type: type = assoc_entity_type @Eats [concrete]
// CHECK:STDOUT:   %assoc0.e43: %Eats.assoc_type = assoc_entity element0, @Eats.%Eats.Eat.decl [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Goat.Bleet.type: type = fn_type @Goat.Bleet [concrete]
// CHECK:STDOUT:   %Goat.Bleet: %Goat.Bleet.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness @Goat.%Eats.impl_witness_table [concrete]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.type: type = fn_type @Goat.as.Eats.impl.Eat [concrete]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat: %Goat.as.Eats.impl.Eat.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%Eats.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.ab7: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %.843: type = fn_type_with_self_type %Eats.Eat.type, %Eats.facet [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Goat, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.f2a: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.353: %AggregateT.as_type.as.Destroy.impl.Op.type.f2a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.940: type = ptr_type %Goat [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ab7 = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc22_33: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc22_28.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc22_28.3: init %Goat = class_init (), %.loc22_28.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc22_28.4: ref %Goat = temporary %.loc22_28.2, %.loc22_28.3
// CHECK:STDOUT:   %.loc22_30.1: ref %Goat = converted %.loc22_28.1, %.loc22_28.4
// CHECK:STDOUT:   %.loc22_15.1: type = splice_block %.loc22_15.3 [concrete = constants.%Goat] {
// CHECK:STDOUT:     %Goat.ref.loc22_10: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref.loc22: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %Animal.facet.loc22: %Animal.type = facet_value %Goat.ref.loc22_10, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %.loc22_15.2: %Animal.type = converted %Goat.ref.loc22_10, %Animal.facet.loc22 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %as_type.loc22: type = facet_access_type %.loc22_15.2 [concrete = constants.%Goat]
// CHECK:STDOUT:     %.loc22_15.3: type = converted %.loc22_15.2, %as_type.loc22 [concrete = constants.%Goat]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_30.2: %Goat = bind_value %.loc22_30.1
// CHECK:STDOUT:   %x: %Goat = bind_name x, %.loc22_30.2
// CHECK:STDOUT:   %x.ref.loc23: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Bleet.ref.loc23: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc23: init %empty_tuple.type = call %Bleet.ref.loc23()
// CHECK:STDOUT:   %x.ref.loc24: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Eat.ref.loc24: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0.e43]
// CHECK:STDOUT:   %impl.elem0.loc24: %.843 = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc24: init %empty_tuple.type = call %impl.elem0.loc24()
// CHECK:STDOUT:   %.loc26_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc26_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc26_6.3: init %Goat = class_init (), %.loc26_6.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc26_6.4: ref %Goat = temporary %.loc26_6.2, %.loc26_6.3
// CHECK:STDOUT:   %.loc26_8: ref %Goat = converted %.loc26_6.1, %.loc26_6.4
// CHECK:STDOUT:   %Goat.ref.loc26_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc26: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc26: %Animal.type = facet_value %Goat.ref.loc26_21, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc26_26: %Animal.type = converted %Goat.ref.loc26_21, %Animal.facet.loc26 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc26: type = facet_access_type %.loc26_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_35: type = converted %.loc26_26, %as_type.loc26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Bleet.ref.loc26: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc26: init %empty_tuple.type = call %Bleet.ref.loc26()
// CHECK:STDOUT:   %.loc27_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc27_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc27_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc27_6.3: init %Goat = class_init (), %.loc27_6.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc27_6.4: ref %Goat = temporary %.loc27_6.2, %.loc27_6.3
// CHECK:STDOUT:   %.loc27_8: ref %Goat = converted %.loc27_6.1, %.loc27_6.4
// CHECK:STDOUT:   %Goat.ref.loc27_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc27: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc27: %Animal.type = facet_value %Goat.ref.loc27_21, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc27_26: %Animal.type = converted %Goat.ref.loc27_21, %Animal.facet.loc27 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc27: type = facet_access_type %.loc27_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc27_35: type = converted %.loc27_26, %as_type.loc27 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eat.ref.loc27: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0.e43]
// CHECK:STDOUT:   %impl.elem0.loc27: %.843 = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc27: init %empty_tuple.type = call %impl.elem0.loc27()
// CHECK:STDOUT:   %facet_value.loc27: %type_where = facet_value constants.%Goat, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc27_6.5: %type_where = converted constants.%Goat, %facet_value.loc27 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc27: <bound method> = bound_method %.loc27_6.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.353
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc27: <bound method> = bound_method %.loc27_6.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc27: %ptr.940 = addr_of %.loc27_6.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc27: init %empty_tuple.type = call %bound_method.loc27(%addr.loc27)
// CHECK:STDOUT:   %facet_value.loc26: %type_where = facet_value constants.%Goat, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc26_6.5: %type_where = converted constants.%Goat, %facet_value.loc26 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %.loc26_6.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.353
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %.loc26_6.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc26: %ptr.940 = addr_of %.loc26_6.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26(%addr.loc26)
// CHECK:STDOUT:   %facet_value.loc22: %type_where = facet_value constants.%Goat, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc22_28.5: %type_where = converted constants.%Goat, %facet_value.loc22 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %.loc22_28.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.353
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %.loc22_28.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc22: %ptr.940 = addr_of %.loc22_28.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_access_type_converts_back_to_original_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %A: %J.type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %B: %A.as_type = bind_symbolic_name B, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.8ba: type = pattern_type %A.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%A, %B) [symbolic]
// CHECK:STDOUT:   %require_complete.8dd: <witness> = require_complete_type %A.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.f19: type = pattern_type %C [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%A, %B) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc25_6.2: %J.type, %B.loc25_13.2: @F.%A.as_type.loc25_17.1 (%A.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %G.specific_fn.loc27_3.2: <specific function> = specific_function constants.%G, @G(%A.loc25_6.1, %B.loc25_13.1) [symbolic = %G.specific_fn.loc27_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%C.loc25_29.1 (%C)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %x.ref: @F.%C.loc25_29.1 (%C) = name_ref x, %x
// CHECK:STDOUT:     %.loc27_6.1: %J.type = converted constants.%A.as_type, constants.%A [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc27_6.2: %J.type = converted constants.%A.as_type, constants.%A [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:     %G.specific_fn.loc27_3.1: <specific function> = specific_function %G.ref, @G(constants.%A, constants.%B) [symbolic = %G.specific_fn.loc27_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.specific_fn.loc27_3.1(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc25_6.1 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc25_17.1 => constants.%A.as_type
// CHECK:STDOUT:   %B.loc25_13.1 => constants.%B
// CHECK:STDOUT:   %pattern_type.loc25_13 => constants.%pattern_type.8ba
// CHECK:STDOUT:   %require_complete.loc25_29 => constants.%require_complete.8dd
// CHECK:STDOUT:   %C.loc25_29.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc25_20 => constants.%pattern_type.f19
// CHECK:STDOUT: }
// CHECK:STDOUT:

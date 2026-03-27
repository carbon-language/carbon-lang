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

fn Feed(unused e:! Eats) {}

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

// The class C is holding a specific with a 2 SymbolicBindings where one is a
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
fn G[A:! J, B:! A](unused x: C(A, B)) {}

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
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness @Goat.as.Eats.impl.%Eats.impl_witness_table [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%Eats.impl_witness) [concrete]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat.ref, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc14_14: %Animal.type = converted %Goat.ref, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc14_28: type = type_literal type [concrete = type]
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.assoc_type: type = assoc_entity_type @Eats [concrete]
// CHECK:STDOUT:   %assoc0.083: %Eats.assoc_type = assoc_entity element0, @Eats.WithSelf.%Eats.WithSelf.Eat.decl [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Goat.Bleet.type: type = fn_type @Goat.Bleet [concrete]
// CHECK:STDOUT:   %Goat.Bleet: %Goat.Bleet.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness @Goat.as.Eats.impl.%Eats.impl_witness_table [concrete]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.type: type = fn_type @Goat.as.Eats.impl.Eat [concrete]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat: %Goat.as.Eats.impl.Eat.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%Eats.impl_witness) [concrete]
// CHECK:STDOUT:   %Eats.WithSelf.Eat.type.b4d: type = fn_type @Eats.WithSelf.Eat, @Eats.WithSelf(%Eats.facet) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %.ca5e0e.1: ref %Goat = temporary invalid, @F.%.loc26_30.1 [concrete]
// CHECK:STDOUT:   %.35f: type = fn_type_with_self_type %Eats.WithSelf.Eat.type.b4d, %Eats.facet [concrete]
// CHECK:STDOUT:   %.ca5e0e.2: ref %Goat = temporary invalid, @F.%.loc34_8.1 [concrete]
// CHECK:STDOUT:   %.ca5e0e.3: ref %Goat = temporary invalid, @F.%.loc39_8.1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref.loc26_33: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_28.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc26_28.3: init %Goat to %.loc26_28.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc26_30.1: init %Goat = converted %.loc26_28.1, %.loc26_28.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc26_15.1: type = splice_block %.loc26_15.3 [concrete = constants.%Goat] {
// CHECK:STDOUT:     %Goat.ref.loc26_10: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref.loc26: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %Animal.facet.loc26: %Animal.type = facet_value %Goat.ref.loc26_10, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %.loc26_15.2: %Animal.type = converted %Goat.ref.loc26_10, %Animal.facet.loc26 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %as_type.loc26: type = facet_access_type %.loc26_15.2 [concrete = constants.%Goat]
// CHECK:STDOUT:     %.loc26_15.3: type = converted %.loc26_15.2, %as_type.loc26 [concrete = constants.%Goat]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26_30.2: ref %Goat = temporary %.loc26_28.2, %.loc26_30.1 [concrete = constants.%.ca5e0e.1]
// CHECK:STDOUT:   %.loc26_30.3: %Goat = acquire_value %.loc26_30.2
// CHECK:STDOUT:   %x: %Goat = value_binding x, %.loc26_30.3
// CHECK:STDOUT:   %x.ref.loc27: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Bleet.ref.loc27: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc27: init %empty_tuple.type = call %Bleet.ref.loc27()
// CHECK:STDOUT:   %x.ref.loc28: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Eat.ref.loc28: %Eats.assoc_type = name_ref Eat, @Eats.WithSelf.%assoc0 [concrete = constants.%assoc0.083]
// CHECK:STDOUT:   %impl.elem0.loc28: %.35f = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc28: init %empty_tuple.type = call %impl.elem0.loc28()
// CHECK:STDOUT:   %.loc34_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref.loc34_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc34_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc34_6.3: init %Goat to %.loc34_6.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc34_8.1: init %Goat = converted %.loc34_6.1, %.loc34_6.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %Goat.ref.loc34_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc34: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc34: %Animal.type = facet_value %Goat.ref.loc34_21, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc34_26: %Animal.type = converted %Goat.ref.loc34_21, %Animal.facet.loc34 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc34: type = facet_access_type %.loc34_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc34_35: type = converted %.loc34_26, %as_type.loc34 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc34_8.2: ref %Goat = temporary %.loc34_6.2, %.loc34_8.1 [concrete = constants.%.ca5e0e.2]
// CHECK:STDOUT:   %Bleet.ref.loc34: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc34: init %empty_tuple.type = call %Bleet.ref.loc34()
// CHECK:STDOUT:   %.loc39_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref.loc39_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc39_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc39_6.3: init %Goat to %.loc39_6.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc39_8.1: init %Goat = converted %.loc39_6.1, %.loc39_6.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %Goat.ref.loc39_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc39: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc39: %Animal.type = facet_value %Goat.ref.loc39_21, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc39_26: %Animal.type = converted %Goat.ref.loc39_21, %Animal.facet.loc39 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc39: type = facet_access_type %.loc39_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc39_35: type = converted %.loc39_26, %as_type.loc39 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc39_8.2: ref %Goat = temporary %.loc39_6.2, %.loc39_8.1 [concrete = constants.%.ca5e0e.3]
// CHECK:STDOUT:   %Eat.ref.loc39: %Eats.assoc_type = name_ref Eat, @Eats.WithSelf.%assoc0 [concrete = constants.%assoc0.083]
// CHECK:STDOUT:   %impl.elem0.loc39: %.35f = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc39: init %empty_tuple.type = call %impl.elem0.loc39()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Goat) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_access_type_converts_back_to_original_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %A: %J.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A [symbolic]
// CHECK:STDOUT:   %pattern_type.14f: type = pattern_type %A.binding.as_type [symbolic]
// CHECK:STDOUT:   %B: %A.binding.as_type = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%A, %B) [symbolic]
// CHECK:STDOUT:   %pattern_type.e04: type = pattern_type %C [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%A, %B) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc25_7.2: %J.type, %B.loc25_14.2: @F.%A.binding.as_type (%A.binding.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %G.specific_fn.loc27_3.2: <specific function> = specific_function constants.%G, @G(%A.loc25_7.1, %B.loc25_14.1) [symbolic = %G.specific_fn.loc27_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%C.loc25_29.1 (%C)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %x.ref: @F.%C.loc25_29.1 (%C) = name_ref x, %x
// CHECK:STDOUT:     %.loc27: %J.type = converted constants.%A.binding.as_type, constants.%A [symbolic = %A.loc25_7.1 (constants.%A)]
// CHECK:STDOUT:     %G.specific_fn.loc27_3.1: <specific function> = specific_function %G.ref, @G(constants.%A, constants.%B) [symbolic = %G.specific_fn.loc27_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.specific_fn.loc27_3.1(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc25_7.1 => constants.%A
// CHECK:STDOUT:   %A.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %B.loc25_14.1 => constants.%B
// CHECK:STDOUT:   %pattern_type.loc25_14 => constants.%pattern_type.14f
// CHECK:STDOUT:   %C.loc25_29.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc25_21 => constants.%pattern_type.e04
// CHECK:STDOUT: }
// CHECK:STDOUT:

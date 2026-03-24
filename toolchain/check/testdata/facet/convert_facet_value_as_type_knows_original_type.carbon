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
// CHECK:STDOUT:   %.loc28_14: %Animal.type = converted %Goat.ref, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc28_28: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc28_14 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc28_25: type = converted %.loc28_14, %as_type [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %.loc28_25, (constants.%Eats.impl_witness) [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %.loc28_32: %Eats.type = converted %.loc28_25, %Eats.facet [concrete = constants.%Eats.facet]
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
// CHECK:STDOUT:   %.35f: type = fn_type_with_self_type %Eats.WithSelf.Eat.type.b4d, %Eats.facet [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc54_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref.loc54_33: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc54_28.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc54_28.3: init %Goat to %.loc54_28.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc54_30.1: init %Goat = converted %.loc54_28.1, %.loc54_28.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc54_15.1: type = splice_block %.loc54_15.3 [concrete = constants.%Goat] {
// CHECK:STDOUT:     %Goat.ref.loc54_10: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref.loc54: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %Animal.facet.loc54: %Animal.type = facet_value %Goat.ref.loc54_10, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %.loc54_15.2: %Animal.type = converted %Goat.ref.loc54_10, %Animal.facet.loc54 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %as_type.loc54: type = facet_access_type %.loc54_15.2 [concrete = constants.%Goat]
// CHECK:STDOUT:     %.loc54_15.3: type = converted %.loc54_15.2, %as_type.loc54 [concrete = constants.%Goat]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc54_30.2: ref %Goat = temporary %.loc54_28.2, %.loc54_30.1
// CHECK:STDOUT:   %.loc54_30.3: %Goat = acquire_value %.loc54_30.2
// CHECK:STDOUT:   %x: %Goat = value_binding x, %.loc54_30.3
// CHECK:STDOUT:   %x.ref.loc55: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Bleet.ref.loc55: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc55: init %empty_tuple.type = call %Bleet.ref.loc55()
// CHECK:STDOUT:   %x.ref.loc56: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Eat.ref.loc56: %Eats.assoc_type = name_ref Eat, @Eats.WithSelf.%assoc0 [concrete = constants.%assoc0.083]
// CHECK:STDOUT:   %impl.elem0.loc56: %.35f = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc56: init %empty_tuple.type = call %impl.elem0.loc56()
// CHECK:STDOUT:   %.loc58_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref.loc58_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc58_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc58_6.3: init %Goat to %.loc58_6.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc58_8.1: init %Goat = converted %.loc58_6.1, %.loc58_6.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %Goat.ref.loc58_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc58: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc58: %Animal.type = facet_value %Goat.ref.loc58_21, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc58_26: %Animal.type = converted %Goat.ref.loc58_21, %Animal.facet.loc58 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc58: type = facet_access_type %.loc58_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc58_35: type = converted %.loc58_26, %as_type.loc58 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc58_8.2: ref %Goat = temporary %.loc58_6.2, %.loc58_8.1
// CHECK:STDOUT:   %Bleet.ref.loc58: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc58: init %empty_tuple.type = call %Bleet.ref.loc58()
// CHECK:STDOUT:   %.loc59_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref.loc59_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc59_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc59_6.3: init %Goat to %.loc59_6.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc59_8.1: init %Goat = converted %.loc59_6.1, %.loc59_6.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %Goat.ref.loc59_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc59: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc59: %Animal.type = facet_value %Goat.ref.loc59_21, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc59_26: %Animal.type = converted %Goat.ref.loc59_21, %Animal.facet.loc59 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc59: type = facet_access_type %.loc59_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc59_35: type = converted %.loc59_26, %as_type.loc59 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc59_8.2: ref %Goat = temporary %.loc59_6.2, %.loc59_8.1
// CHECK:STDOUT:   %Eat.ref.loc59: %Eats.assoc_type = name_ref Eat, @Eats.WithSelf.%assoc0 [concrete = constants.%assoc0.083]
// CHECK:STDOUT:   %impl.elem0.loc59: %.35f = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc59: init %empty_tuple.type = call %impl.elem0.loc59()
// CHECK:STDOUT:   %Destroy.Op.bound.loc59: <bound method> = bound_method %.loc59_8.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc59: init %empty_tuple.type = call %Destroy.Op.bound.loc59(%.loc59_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc58: <bound method> = bound_method %.loc58_8.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc58: init %empty_tuple.type = call %Destroy.Op.bound.loc58(%.loc58_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc54: <bound method> = bound_method %.loc54_30.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc54: init %empty_tuple.type = call %Destroy.Op.bound.loc54(%.loc54_30.2)
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
// CHECK:STDOUT: generic fn @F(%A.loc88_6.2: %J.type, %B.loc88_13.2: @F.%A.binding.as_type (%A.binding.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %G.specific_fn.loc90_3.2: <specific function> = specific_function constants.%G, @G(%A.loc88_6.1, %B.loc88_13.1) [symbolic = %G.specific_fn.loc90_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%C.loc88_29.1 (%C)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %x.ref: @F.%C.loc88_29.1 (%C) = name_ref x, %x
// CHECK:STDOUT:     %.loc90: %J.type = converted constants.%A.binding.as_type, constants.%A [symbolic = %A.loc88_6.1 (constants.%A)]
// CHECK:STDOUT:     %G.specific_fn.loc90_3.1: <specific function> = specific_function %G.ref, @G(constants.%A, constants.%B) [symbolic = %G.specific_fn.loc90_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.specific_fn.loc90_3.1(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc88_6.1 => constants.%A
// CHECK:STDOUT:   %A.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %B.loc88_13.1 => constants.%B
// CHECK:STDOUT:   %pattern_type.loc88_13 => constants.%pattern_type.14f
// CHECK:STDOUT:   %C.loc88_29.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc88_21 => constants.%pattern_type.e04
// CHECK:STDOUT: }
// CHECK:STDOUT:

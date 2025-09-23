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
  Feed((Goat as Animal) as type);
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

  let x: Goat as Animal = {} as Goat;
  x.Bleet();
  x.Eat();

  (({} as Goat) as (Goat as Animal)).Bleet();
  (({} as Goat) as (Goat as Animal)).Eat();
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
  G(x);
}

// CHECK:STDOUT: --- explicit_as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.9e6: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.611: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Goat.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.940: type = ptr_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.396: type = pattern_type %ptr.940 [concrete]
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.type: type = fn_type @Goat.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op: %Goat.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness file.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %e: %Eats.type = bind_symbolic_name e, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.384: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%Eats.impl_witness) [concrete]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [concrete]
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
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:   impl_decl @Goat.as.Eats.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (), @Goat.as.Eats.impl [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table [concrete = constants.%Eats.impl_witness]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %e.patt: %pattern_type.384 = symbolic_binding_pattern e, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e.loc10_9.2: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc10_9.1 (constants.%e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9e6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.611]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Destroy.impl: @Goat.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.decl: %Goat.as.Destroy.impl.Op.type = fn_decl @Goat.as.Destroy.impl.Op [concrete = constants.%Goat.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.396 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.396 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: impl @Goat.as.Eats.impl: %Goat.ref as %Eats.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Eats.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:   impl_decl @Goat.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Goat.as.Destroy.impl.%Goat.as.Destroy.impl.Op.decl), @Goat.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Goat.as.Destroy.impl.Op(%self.param: %ptr.940) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%e.loc10_9.2: %Eats.type) {
// CHECK:STDOUT:   %e.loc10_9.1: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc10_9.1 (constants.%e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc13_14: %Animal.type = converted %Goat.ref, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc13_14 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc13_25: type = converted %.loc13_14, %as_type [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value constants.%Goat, (constants.%Eats.impl_witness) [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %.loc13_32: %Eats.type = converted %.loc13_25, %Eats.facet [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [concrete = constants.%Feed.specific_fn]
// CHECK:STDOUT:   %Feed.call: init %empty_tuple.type = call %Feed.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%e) {
// CHECK:STDOUT:   %e.loc10_9.1 => constants.%e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %e.loc10_9.1 => constants.%Eats.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_type_in_type_position.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.9e6: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Eats.Eat.type: type = fn_type @Eats.Eat [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eats.Eat: %Eats.Eat.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.assoc_type: type = assoc_entity_type @Eats [concrete]
// CHECK:STDOUT:   %assoc0.e43: %Eats.assoc_type = assoc_entity element0, @Eats.%Eats.Eat.decl [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.611: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Goat.Bleet.type: type = fn_type @Goat.Bleet [concrete]
// CHECK:STDOUT:   %Goat.Bleet: %Goat.Bleet.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness @Goat.%Eats.impl_witness_table [concrete]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.type: type = fn_type @Goat.as.Eats.impl.Eat [concrete]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat: %Goat.as.Eats.impl.Eat.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%Eats.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.06d: <witness> = impl_witness @Goat.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.940: type = ptr_type %Goat [concrete]
// CHECK:STDOUT:   %pattern_type.396: type = pattern_type %ptr.940 [concrete]
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.type: type = fn_type @Goat.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op: %Goat.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.ab7: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %.843: type = fn_type_with_self_type %Eats.Eat.type, %Eats.facet [concrete]
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
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9e6]
// CHECK:STDOUT:   %Eats.Eat.decl: %Eats.Eat.type = fn_decl @Eats.Eat [concrete = constants.%Eats.Eat] {} {}
// CHECK:STDOUT:   %assoc0: %Eats.assoc_type = assoc_entity element0, %Eats.Eat.decl [concrete = constants.%assoc0.e43]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Eat = %assoc0
// CHECK:STDOUT:   witness = (%Eats.Eat.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.611]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Animal.impl: %Self.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @Goat.%Animal.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Eats.impl: %Self.ref as %Eats.ref {
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.decl: %Goat.as.Eats.impl.Eat.type = fn_decl @Goat.as.Eats.impl.Eat [concrete = constants.%Goat.as.Eats.impl.Eat] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Eat = %Goat.as.Eats.impl.Eat.decl
// CHECK:STDOUT:   witness = @Goat.%Eats.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Destroy.impl: @Goat.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.decl: %Goat.as.Destroy.impl.Op.type = fn_decl @Goat.as.Destroy.impl.Op [concrete = constants.%Goat.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.396 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.396 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %Goat.Bleet.decl: %Goat.Bleet.type = fn_decl @Goat.Bleet [concrete = constants.%Goat.Bleet] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:   impl_decl @Goat.as.Eats.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (@Goat.as.Eats.impl.%Goat.as.Eats.impl.Eat.decl), @Goat.as.Eats.impl [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table [concrete = constants.%Eats.impl_witness]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:   impl_decl @Goat.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Goat.as.Destroy.impl.%Goat.as.Destroy.impl.Op.decl), @Goat.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.06d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT:   .Bleet = %Goat.Bleet.decl
// CHECK:STDOUT:   .Animal = <poisoned>
// CHECK:STDOUT:   .Eats = <poisoned>
// CHECK:STDOUT:   .Eat = <poisoned>
// CHECK:STDOUT:   extend @Goat.as.Eats.impl.%Eats.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Eats.Eat(@Eats.%Self: %Eats.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Goat.Bleet() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Goat.as.Eats.impl.Eat() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Goat.as.Destroy.impl.Op(%self.param: %ptr.940) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ab7 = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc21_33: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc21_28.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc21_28.3: init %Goat = class_init (), %.loc21_28.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc21_28.4: ref %Goat = temporary %.loc21_28.2, %.loc21_28.3
// CHECK:STDOUT:   %.loc21_30.1: ref %Goat = converted %.loc21_28.1, %.loc21_28.4
// CHECK:STDOUT:   %.loc21_15.1: type = splice_block %.loc21_15.3 [concrete = constants.%Goat] {
// CHECK:STDOUT:     %Goat.ref.loc21_10: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref.loc21: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %Animal.facet.loc21: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %.loc21_15.2: %Animal.type = converted %Goat.ref.loc21_10, %Animal.facet.loc21 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %as_type.loc21: type = facet_access_type %.loc21_15.2 [concrete = constants.%Goat]
// CHECK:STDOUT:     %.loc21_15.3: type = converted %.loc21_15.2, %as_type.loc21 [concrete = constants.%Goat]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_30.2: %Goat = bind_value %.loc21_30.1
// CHECK:STDOUT:   %x: %Goat = bind_name x, %.loc21_30.2
// CHECK:STDOUT:   %x.ref.loc22: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Bleet.ref.loc22: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc22: init %empty_tuple.type = call %Bleet.ref.loc22()
// CHECK:STDOUT:   %x.ref.loc23: %Goat = name_ref x, %x
// CHECK:STDOUT:   %Eat.ref.loc23: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0.e43]
// CHECK:STDOUT:   %impl.elem0.loc23: %.843 = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc23: init %empty_tuple.type = call %impl.elem0.loc23()
// CHECK:STDOUT:   %.loc25_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc25_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc25_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc25_6.3: init %Goat = class_init (), %.loc25_6.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc25_6.4: ref %Goat = temporary %.loc25_6.2, %.loc25_6.3
// CHECK:STDOUT:   %.loc25_8: ref %Goat = converted %.loc25_6.1, %.loc25_6.4
// CHECK:STDOUT:   %Goat.ref.loc25_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc25: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc25: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc25_26: %Animal.type = converted %Goat.ref.loc25_21, %Animal.facet.loc25 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc25: type = facet_access_type %.loc25_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc25_35: type = converted %.loc25_26, %as_type.loc25 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Bleet.ref.loc25: %Goat.Bleet.type = name_ref Bleet, @Goat.%Goat.Bleet.decl [concrete = constants.%Goat.Bleet]
// CHECK:STDOUT:   %Goat.Bleet.call.loc25: init %empty_tuple.type = call %Bleet.ref.loc25()
// CHECK:STDOUT:   %.loc26_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc26_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc26_6.3: init %Goat = class_init (), %.loc26_6.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc26_6.4: ref %Goat = temporary %.loc26_6.2, %.loc26_6.3
// CHECK:STDOUT:   %.loc26_8: ref %Goat = converted %.loc26_6.1, %.loc26_6.4
// CHECK:STDOUT:   %Goat.ref.loc26_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc26: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc26: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc26_26: %Animal.type = converted %Goat.ref.loc26_21, %Animal.facet.loc26 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc26: type = facet_access_type %.loc26_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_35: type = converted %.loc26_26, %as_type.loc26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eat.ref.loc26: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0.e43]
// CHECK:STDOUT:   %impl.elem0.loc26: %.843 = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc26: init %empty_tuple.type = call %impl.elem0.loc26()
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %.loc26_6.4, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc26: %ptr.940 = addr_of %.loc26_6.4
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound.loc26(%addr.loc26)
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound.loc25: <bound method> = bound_method %.loc25_6.4, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc25: %ptr.940 = addr_of %.loc25_6.4
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call.loc25: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound.loc25(%addr.loc25)
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %.loc21_28.4, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc21: %ptr.940 = addr_of %.loc21_28.4
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound.loc21(%addr.loc21)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.Eat(constants.%Self.9e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.Eat(constants.%Eats.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_access_type_converts_back_to_original_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.bf6: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %A: %J.type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %B: %A.as_type = bind_symbolic_name B, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.8ba: type = pattern_type %A.as_type [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%A, %B) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%A, %B) [symbolic]
// CHECK:STDOUT:   %require_complete.8dd: <witness> = require_complete_type %A.as_type [symbolic]
// CHECK:STDOUT:   %ptr.39d: type = ptr_type %C [symbolic]
// CHECK:STDOUT:   %pattern_type.b4f: type = pattern_type %ptr.39d [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%A, %B) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.f19: type = pattern_type %C [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.667: <witness> = require_complete_type %C [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%A, %B) [symbolic]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %A.patt: %pattern_type.84b = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: @C.%pattern_type (%pattern_type.8ba) = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_13: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc5_9.2: %J.type = bind_symbolic_name A, 0 [symbolic = %A.loc5_9.1 (constants.%A)]
// CHECK:STDOUT:     %.loc5_20.1: type = splice_block %.loc5_20.2 [symbolic = %A.as_type.loc5_20.1 (constants.%A.as_type)] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %J.type = name_ref A, %A.loc5_9.2 [symbolic = %A.loc5_9.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type.loc5_20.2: type = facet_access_type %A.ref [symbolic = %A.as_type.loc5_20.1 (constants.%A.as_type)]
// CHECK:STDOUT:       %.loc5_20.2: type = converted %A.ref, %A.as_type.loc5_20.2 [symbolic = %A.as_type.loc5_20.1 (constants.%A.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc5_16.2: @C.%A.as_type.loc5_20.1 (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B.loc5_16.1 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %A.patt: %pattern_type.84b = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: @G.%pattern_type.loc23_13 (%pattern_type.8ba) = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @G.%pattern_type.loc23_20 (%pattern_type.f19) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @G.%pattern_type.loc23_20 (%pattern_type.f19) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_10: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc23_6.2: %J.type = bind_symbolic_name A, 0 [symbolic = %A.loc23_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc23_17.1: type = splice_block %.loc23_17.2 [symbolic = %A.as_type.loc23_17.1 (constants.%A.as_type)] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref.loc23_17: %J.type = name_ref A, %A.loc23_6.2 [symbolic = %A.loc23_6.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type.loc23_17.2: type = facet_access_type %A.ref.loc23_17 [symbolic = %A.as_type.loc23_17.1 (constants.%A.as_type)]
// CHECK:STDOUT:       %.loc23_17.2: type = converted %A.ref.loc23_17, %A.as_type.loc23_17.2 [symbolic = %A.as_type.loc23_17.1 (constants.%A.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc23_13.2: @G.%A.as_type.loc23_17.1 (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B.loc23_13.1 (constants.%B)]
// CHECK:STDOUT:     %x.param: @G.%C.loc23_29.1 (%C) = value_param call_param0
// CHECK:STDOUT:     %.loc23_29.1: type = splice_block %C.loc23_29.2 [symbolic = %C.loc23_29.1 (constants.%C)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %A.ref.loc23_25: %J.type = name_ref A, %A.loc23_6.2 [symbolic = %A.loc23_6.1 (constants.%A)]
// CHECK:STDOUT:       %B.ref: @G.%A.as_type.loc23_17.1 (%A.as_type) = name_ref B, %B.loc23_13.2 [symbolic = %B.loc23_13.1 (constants.%B)]
// CHECK:STDOUT:       %.loc23_29.2: %J.type = converted constants.%A.as_type, constants.%A [symbolic = %A.loc23_6.1 (constants.%A)]
// CHECK:STDOUT:       %C.loc23_29.2: type = class_type @C, @C(constants.%A, constants.%B) [symbolic = %C.loc23_29.1 (constants.%C)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @G.%C.loc23_29.1 (%C) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.84b = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: @F.%pattern_type.loc25_13 (%pattern_type.8ba) = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc25_20 (%pattern_type.f19) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc25_20 (%pattern_type.f19) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_10: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc25_6.2: %J.type = bind_symbolic_name A, 0 [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc25_17.1: type = splice_block %.loc25_17.2 [symbolic = %A.as_type.loc25_17.1 (constants.%A.as_type)] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref.loc25_17: %J.type = name_ref A, %A.loc25_6.2 [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:       %A.as_type.loc25_17.2: type = facet_access_type %A.ref.loc25_17 [symbolic = %A.as_type.loc25_17.1 (constants.%A.as_type)]
// CHECK:STDOUT:       %.loc25_17.2: type = converted %A.ref.loc25_17, %A.as_type.loc25_17.2 [symbolic = %A.as_type.loc25_17.1 (constants.%A.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc25_13.2: @F.%A.as_type.loc25_17.1 (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B.loc25_13.1 (constants.%B)]
// CHECK:STDOUT:     %x.param: @F.%C.loc25_29.1 (%C) = value_param call_param0
// CHECK:STDOUT:     %.loc25_29.1: type = splice_block %C.loc25_29.2 [symbolic = %C.loc25_29.1 (constants.%C)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %A.ref.loc25_25: %J.type = name_ref A, %A.loc25_6.2 [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:       %B.ref: @F.%A.as_type.loc25_17.1 (%A.as_type) = name_ref B, %B.loc25_13.2 [symbolic = %B.loc25_13.1 (constants.%B)]
// CHECK:STDOUT:       %.loc25_29.2: %J.type = converted constants.%A.as_type, constants.%A [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:       %C.loc25_29.2: type = class_type @C, @C(constants.%A, constants.%B) [symbolic = %C.loc25_29.1 (constants.%C)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%C.loc25_29.1 (%C) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bf6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%A.loc5_9.2: %J.type, @C.%B.loc5_16.2: @C.%A.as_type.loc5_20.1 (%A.as_type)) {
// CHECK:STDOUT:   %A: %J.type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic = %A.as_type (constants.%A.as_type)]
// CHECK:STDOUT:   %B: @C.as.Destroy.impl.%A.as_type (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%A, %B) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%A, %B) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%A, %B) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.b4f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.b4f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc5_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.39d) = value_param call_param0
// CHECK:STDOUT:       %.loc5_23.2: type = splice_block %Self.ref [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:         %.loc5_23.3: type = specific_constant constants.%C, @C(constants.%A, constants.%B) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_23.3 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.39d) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%A.loc5_9.2: %J.type, %B.loc5_16.2: @C.%A.as_type.loc5_20.1 (%A.as_type)) {
// CHECK:STDOUT:   %A.loc5_9.1: %J.type = bind_symbolic_name A, 0 [symbolic = %A.loc5_9.1 (constants.%A)]
// CHECK:STDOUT:   %A.as_type.loc5_20.1: type = facet_access_type %A.loc5_9.1 [symbolic = %A.as_type.loc5_20.1 (constants.%A.as_type)]
// CHECK:STDOUT:   %B.loc5_16.1: @C.%A.as_type.loc5_20.1 (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B.loc5_16.1 (constants.%B)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A.as_type.loc5_20.1 [symbolic = %pattern_type (constants.%pattern_type.8ba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.as_type.loc5_20.1 [symbolic = %require_complete (constants.%require_complete.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [symbolic = @C.as.Destroy.impl.%C (constants.%C)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%A, constants.%B) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%A.loc5_9.2: %J.type, @C.%B.loc5_16.2: @C.%A.as_type.loc5_20.1 (%A.as_type)) {
// CHECK:STDOUT:   %A: %J.type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic = %A.as_type (constants.%A.as_type)]
// CHECK:STDOUT:   %B: @C.as.Destroy.impl.Op.%A.as_type (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%A, %B) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.39d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.b4f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.39d)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%A.loc23_6.2: %J.type, %B.loc23_13.2: @G.%A.as_type.loc23_17.1 (%A.as_type)) {
// CHECK:STDOUT:   %A.loc23_6.1: %J.type = bind_symbolic_name A, 0 [symbolic = %A.loc23_6.1 (constants.%A)]
// CHECK:STDOUT:   %A.as_type.loc23_17.1: type = facet_access_type %A.loc23_6.1 [symbolic = %A.as_type.loc23_17.1 (constants.%A.as_type)]
// CHECK:STDOUT:   %B.loc23_13.1: @G.%A.as_type.loc23_17.1 (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B.loc23_13.1 (constants.%B)]
// CHECK:STDOUT:   %pattern_type.loc23_13: type = pattern_type %A.as_type.loc23_17.1 [symbolic = %pattern_type.loc23_13 (constants.%pattern_type.8ba)]
// CHECK:STDOUT:   %require_complete.loc23_29: <witness> = require_complete_type %A.as_type.loc23_17.1 [symbolic = %require_complete.loc23_29 (constants.%require_complete.8dd)]
// CHECK:STDOUT:   %C.loc23_29.1: type = class_type @C, @C(%A.loc23_6.1, %B.loc23_13.1) [symbolic = %C.loc23_29.1 (constants.%C)]
// CHECK:STDOUT:   %pattern_type.loc23_20: type = pattern_type %C.loc23_29.1 [symbolic = %pattern_type.loc23_20 (constants.%pattern_type.f19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc23_21: <witness> = require_complete_type %C.loc23_29.1 [symbolic = %require_complete.loc23_21 (constants.%require_complete.667)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @G.%C.loc23_29.1 (%C)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc25_6.2: %J.type, %B.loc25_13.2: @F.%A.as_type.loc25_17.1 (%A.as_type)) {
// CHECK:STDOUT:   %A.loc25_6.1: %J.type = bind_symbolic_name A, 0 [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:   %A.as_type.loc25_17.1: type = facet_access_type %A.loc25_6.1 [symbolic = %A.as_type.loc25_17.1 (constants.%A.as_type)]
// CHECK:STDOUT:   %B.loc25_13.1: @F.%A.as_type.loc25_17.1 (%A.as_type) = bind_symbolic_name B, 1 [symbolic = %B.loc25_13.1 (constants.%B)]
// CHECK:STDOUT:   %pattern_type.loc25_13: type = pattern_type %A.as_type.loc25_17.1 [symbolic = %pattern_type.loc25_13 (constants.%pattern_type.8ba)]
// CHECK:STDOUT:   %require_complete.loc25_29: <witness> = require_complete_type %A.as_type.loc25_17.1 [symbolic = %require_complete.loc25_29 (constants.%require_complete.8dd)]
// CHECK:STDOUT:   %C.loc25_29.1: type = class_type @C, @C(%A.loc25_6.1, %B.loc25_13.1) [symbolic = %C.loc25_29.1 (constants.%C)]
// CHECK:STDOUT:   %pattern_type.loc25_20: type = pattern_type %C.loc25_29.1 [symbolic = %pattern_type.loc25_20 (constants.%pattern_type.f19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25_21: <witness> = require_complete_type %C.loc25_29.1 [symbolic = %require_complete.loc25_21 (constants.%require_complete.667)]
// CHECK:STDOUT:   %G.specific_fn.loc26_3.2: <specific function> = specific_function constants.%G, @G(%A.loc25_6.1, %B.loc25_13.1) [symbolic = %G.specific_fn.loc26_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%C.loc25_29.1 (%C)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %x.ref: @F.%C.loc25_29.1 (%C) = name_ref x, %x
// CHECK:STDOUT:     %.loc26_6.1: %J.type = converted constants.%A.as_type, constants.%A [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc26_6.2: %J.type = converted constants.%A.as_type, constants.%A [symbolic = %A.loc25_6.1 (constants.%A)]
// CHECK:STDOUT:     %G.specific_fn.loc26_3.1: <specific function> = specific_function %G.ref, @G(constants.%A, constants.%B) [symbolic = %G.specific_fn.loc26_3.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.specific_fn.loc26_3.1(%x.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc5_9.1 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc5_20.1 => constants.%A.as_type
// CHECK:STDOUT:   %B.loc5_16.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8ba
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %A.as_type => constants.%A.as_type
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %A.as_type => constants.%A.as_type
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %ptr => constants.%ptr.39d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc23_6.1 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc23_17.1 => constants.%A.as_type
// CHECK:STDOUT:   %B.loc23_13.1 => constants.%B
// CHECK:STDOUT:   %pattern_type.loc23_13 => constants.%pattern_type.8ba
// CHECK:STDOUT:   %require_complete.loc23_29 => constants.%require_complete.8dd
// CHECK:STDOUT:   %C.loc23_29.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc23_20 => constants.%pattern_type.f19
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc23_21 => constants.%require_complete.667
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

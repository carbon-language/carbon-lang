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
// CHECK:STDOUT:     %.loc11: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e.loc11_9.2: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc11_9.1 (constants.%e)]
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
// CHECK:STDOUT:     %.loc7: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: generic fn @Feed(%e.loc11_9.2: %Eats.type) {
// CHECK:STDOUT:   %e.loc11_9.1: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc11_9.1 (constants.%e)]
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
// CHECK:STDOUT:   %.loc14_14: %Animal.type = converted %Goat.ref, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc14_14 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc14_25: type = converted %.loc14_14, %as_type [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value constants.%Goat, (constants.%Eats.impl_witness) [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %.loc14_32: %Eats.type = converted %.loc14_25, %Eats.facet [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [concrete = constants.%Feed.specific_fn]
// CHECK:STDOUT:   %Feed.call: init %empty_tuple.type = call %Feed.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%e) {
// CHECK:STDOUT:   %e.loc11_9.1 => constants.%e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %e.loc11_9.1 => constants.%Eats.facet
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
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT:   %.loc22_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc22_33: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc22_28.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc22_28.3: init %Goat = class_init (), %.loc22_28.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc22_28.4: ref %Goat = temporary %.loc22_28.2, %.loc22_28.3
// CHECK:STDOUT:   %.loc22_30.1: ref %Goat = converted %.loc22_28.1, %.loc22_28.4
// CHECK:STDOUT:   %.loc22_15.1: type = splice_block %.loc22_15.3 [concrete = constants.%Goat] {
// CHECK:STDOUT:     %Goat.ref.loc22_10: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref.loc22: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %Animal.facet.loc22: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
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
// CHECK:STDOUT:   %Animal.facet.loc26: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
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
// CHECK:STDOUT:   %Animal.facet.loc27: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc27_26: %Animal.type = converted %Goat.ref.loc27_21, %Animal.facet.loc27 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc27: type = facet_access_type %.loc27_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc27_35: type = converted %.loc27_26, %as_type.loc27 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eat.ref.loc27: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0.e43]
// CHECK:STDOUT:   %impl.elem0.loc27: %.843 = impl_witness_access constants.%Eats.impl_witness, element0 [concrete = constants.%Goat.as.Eats.impl.Eat]
// CHECK:STDOUT:   %Goat.as.Eats.impl.Eat.call.loc27: init %empty_tuple.type = call %impl.elem0.loc27()
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound.loc27: <bound method> = bound_method %.loc27_6.4, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc27: %ptr.940 = addr_of %.loc27_6.4
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call.loc27: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound.loc27(%addr.loc27)
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %.loc26_6.4, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc26: %ptr.940 = addr_of %.loc26_6.4
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound.loc26(%addr.loc26)
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %.loc22_28.4, constants.%Goat.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc22: %ptr.940 = addr_of %.loc22_28.4
// CHECK:STDOUT:   %Goat.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %Goat.as.Destroy.impl.Op.bound.loc22(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.Eat(constants.%Self.9e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.Eat(constants.%Eats.facet) {}
// CHECK:STDOUT:

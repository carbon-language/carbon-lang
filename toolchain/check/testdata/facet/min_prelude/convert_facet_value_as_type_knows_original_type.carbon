// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_as_type_knows_original_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_as_type_knows_original_type.carbon

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
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %e: %Eats.type = bind_symbolic_name e, 0 [symbolic]
// CHECK:STDOUT:   %e.patt: %Eats.type = symbolic_binding_pattern e, 0 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%impl_witness) [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%impl_witness) [concrete]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [concrete]
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
// CHECK:STDOUT:   impl_decl @impl.27e [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.b88 [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc9: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %e.patt.loc11_9.1: %Eats.type = symbolic_binding_pattern e, 0 [symbolic = %e.patt.loc11_9.2 (constants.%e.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %e.loc11_9.1: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc11_9.2 (constants.%e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
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
// CHECK:STDOUT: impl @impl.27e: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b88: %Goat.ref as %Eats.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%e.loc11_9.1: %Eats.type) {
// CHECK:STDOUT:   %e.loc11_9.2: %Eats.type = bind_symbolic_name e, 0 [symbolic = %e.loc11_9.2 (constants.%e)]
// CHECK:STDOUT:   %e.patt.loc11_9.2: %Eats.type = symbolic_binding_pattern e, 0 [symbolic = %e.patt.loc11_9.2 (constants.%e.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.patt.loc11_9.1: %Eats.type) {
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
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc14_14: %Animal.type = converted %Goat.ref, %Animal.facet [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc14_14 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc14_25: type = converted %.loc14_14, %as_type [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value constants.%Goat, (constants.%impl_witness) [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %.loc14_32: %Eats.type = converted %.loc14_25, %Eats.facet [concrete = constants.%Eats.facet]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [concrete = constants.%Feed.specific_fn]
// CHECK:STDOUT:   %Feed.call: init %empty_tuple.type = call %Feed.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%e) {
// CHECK:STDOUT:   %e.loc11_9.2 => constants.%e
// CHECK:STDOUT:   %e.patt.loc11_9.2 => constants.%e.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %e.loc11_9.2 => constants.%Eats.facet
// CHECK:STDOUT:   %e.patt.loc11_9.2 => constants.%e.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_type_in_type_position.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Eat.type.e5d: type = fn_type @Eat.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Eat.7cd: %Eat.type.e5d = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.assoc_type: type = assoc_entity_type %Eats.type [concrete]
// CHECK:STDOUT:   %assoc0: %Eats.assoc_type = assoc_entity element0, @Eats.%Eat.decl [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %Bleet.type: type = fn_type @Bleet [concrete]
// CHECK:STDOUT:   %Bleet: %Bleet.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %impl_witness.9a5: <witness> = impl_witness (@impl.626.%Eat.decl) [concrete]
// CHECK:STDOUT:   %Eat.type.1b3: type = fn_type @Eat.2 [concrete]
// CHECK:STDOUT:   %Eat.73e: %Eat.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %Goat, (%impl_witness.9a5) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%impl_witness.1bc) [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %.aa8: type = fn_type_with_self_type %Eat.type.e5d, %Eats.facet [concrete]
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
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
// CHECK:STDOUT:   %Eat.decl: %Eat.type.e5d = fn_decl @Eat.1 [concrete = constants.%Eat.7cd] {} {}
// CHECK:STDOUT:   %assoc0: %Eats.assoc_type = assoc_entity element0, %Eat.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Eat = %assoc0
// CHECK:STDOUT:   witness = (%Eat.decl)
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
// CHECK:STDOUT: impl @impl.32f: %Self.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @Goat.%impl_witness.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.626: %Self.ref as %Eats.ref {
// CHECK:STDOUT:   %Eat.decl: %Eat.type.1b3 = fn_decl @Eat.2 [concrete = constants.%Eat.73e] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Eat = %Eat.decl
// CHECK:STDOUT:   witness = @Goat.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %Bleet.decl: %Bleet.type = fn_decl @Bleet [concrete = constants.%Bleet] {} {}
// CHECK:STDOUT:   impl_decl @impl.32f [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.626 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Goat [concrete = constants.%Goat]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.626.%Eat.decl) [concrete = constants.%impl_witness.9a5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT:   .Bleet = %Bleet.decl
// CHECK:STDOUT:   .Animal = <poisoned>
// CHECK:STDOUT:   .Eats = <poisoned>
// CHECK:STDOUT:   .Eat = <poisoned>
// CHECK:STDOUT:   extend @impl.626.%Eats.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Eat.1(@Eats.%Self: %Eats.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bleet() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Eat.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %Goat = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc22_33: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc22_28.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc22_28.3: init %Goat = class_init (), %.loc22_28.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc22_28.4: ref %Goat = temporary %.loc22_28.2, %.loc22_28.3
// CHECK:STDOUT:   %.loc22_30: ref %Goat = converted %.loc22_28.1, %.loc22_28.4
// CHECK:STDOUT:   %.loc22_15.1: type = splice_block %.loc22_15.3 [concrete = constants.%Goat] {
// CHECK:STDOUT:     %Goat.ref.loc22_10: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref.loc22: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %Animal.facet.loc22: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness.1bc) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %.loc22_15.2: %Animal.type = converted %Goat.ref.loc22_10, %Animal.facet.loc22 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:     %as_type.loc22: type = facet_access_type %.loc22_15.2 [concrete = constants.%Goat]
// CHECK:STDOUT:     %.loc22_15.3: type = converted %.loc22_15.2, %as_type.loc22 [concrete = constants.%Goat]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %Goat = bind_name x, %.loc22_30
// CHECK:STDOUT:   %x.ref.loc23: ref %Goat = name_ref x, %x
// CHECK:STDOUT:   %Bleet.ref.loc23: %Bleet.type = name_ref Bleet, @Goat.%Bleet.decl [concrete = constants.%Bleet]
// CHECK:STDOUT:   %Bleet.call.loc23: init %empty_tuple.type = call %Bleet.ref.loc23()
// CHECK:STDOUT:   %x.ref.loc24: ref %Goat = name_ref x, %x
// CHECK:STDOUT:   %Eat.ref.loc24: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc24: %.aa8 = impl_witness_access constants.%impl_witness.9a5, element0 [concrete = constants.%Eat.73e]
// CHECK:STDOUT:   %Eat.call.loc24: init %empty_tuple.type = call %impl.elem0.loc24()
// CHECK:STDOUT:   %.loc26_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc26_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc26_6.3: init %Goat = class_init (), %.loc26_6.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc26_6.4: ref %Goat = temporary %.loc26_6.2, %.loc26_6.3
// CHECK:STDOUT:   %.loc26_8: ref %Goat = converted %.loc26_6.1, %.loc26_6.4
// CHECK:STDOUT:   %Goat.ref.loc26_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc26: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc26: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness.1bc) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc26_26: %Animal.type = converted %Goat.ref.loc26_21, %Animal.facet.loc26 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc26: type = facet_access_type %.loc26_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc26_35: type = converted %.loc26_26, %as_type.loc26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Bleet.ref.loc26: %Bleet.type = name_ref Bleet, @Goat.%Bleet.decl [concrete = constants.%Bleet]
// CHECK:STDOUT:   %Bleet.call.loc26: init %empty_tuple.type = call %Bleet.ref.loc26()
// CHECK:STDOUT:   %.loc27_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref.loc27_11: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc27_6.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc27_6.3: init %Goat = class_init (), %.loc27_6.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc27_6.4: ref %Goat = temporary %.loc27_6.2, %.loc27_6.3
// CHECK:STDOUT:   %.loc27_8: ref %Goat = converted %.loc27_6.1, %.loc27_6.4
// CHECK:STDOUT:   %Goat.ref.loc27_21: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %Animal.ref.loc27: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   %Animal.facet.loc27: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness.1bc) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc27_26: %Animal.type = converted %Goat.ref.loc27_21, %Animal.facet.loc27 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %as_type.loc27: type = facet_access_type %.loc27_26 [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc27_35: type = converted %.loc27_26, %as_type.loc27 [concrete = constants.%Goat]
// CHECK:STDOUT:   %Eat.ref.loc27: %Eats.assoc_type = name_ref Eat, @Eats.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc27: %.aa8 = impl_witness_access constants.%impl_witness.9a5, element0 [concrete = constants.%Eat.73e]
// CHECK:STDOUT:   %Eat.call.loc27: init %empty_tuple.type = call %impl.elem0.loc27()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eat.1(constants.%Self.1b5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eat.1(constants.%Eats.facet) {}
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

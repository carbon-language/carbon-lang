// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_facet_value_value_to_itself.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_facet_value_value_to_itself.carbon

// --- core.carbon

package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- a.carbon

library "[[@TEST_NAME]]";

import Core;

interface Animal {}

fn FeedAnimal[T:! Animal](a: T) {}

fn HandleAnimal[T:! Animal](a: T) { FeedAnimal(a); }

class Goat {}
impl Goat as Animal {}

fn F() {
  HandleAnimal({} as Goat);
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc4_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc4_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc4_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc4_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc4_22.1: type) {
// CHECK:STDOUT:   %Dest.loc4_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc4_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc4_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc4_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc4_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest.loc4_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc4_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_20.1: type = splice_block %.loc5_20.3 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_20.2: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc5_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_20.3: type = converted %Self.ref, %Self.as_type.loc5_20.2 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc5_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%Dest.loc4_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc4_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc4_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc4_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Animal.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %FeedAnimal.type: type = fn_type @FeedAnimal [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FeedAnimal: %FeedAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.ec8: <specific function> = specific_function %FeedAnimal, @FeedAnimal(%T) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%impl_witness) [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet) [concrete]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.5d0: <specific function> = specific_function %FeedAnimal, @FeedAnimal(%Animal.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .FeedAnimal = %FeedAnimal.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %FeedAnimal.decl: %FeedAnimal.type = fn_decl @FeedAnimal [concrete = constants.%FeedAnimal] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @FeedAnimal.%T.as_type.loc8_30.2 (%T.as_type) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @FeedAnimal.%T.as_type.loc8_30.2 (%T.as_type) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc8_15.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @FeedAnimal.%T.as_type.loc8_30.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_30.1: type = splice_block %.loc8_30.2 [symbolic = %T.as_type.loc8_30.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_30.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_30.2: type = converted %T.ref, %T.as_type.loc8_30.1 [symbolic = %T.as_type.loc8_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @FeedAnimal.%T.as_type.loc8_30.2 (%T.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt.loc10_17.1: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc10_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_17.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc10_32.1: type = splice_block %.loc10_32.2 [symbolic = %T.as_type.loc10_32.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc10_17.1 [symbolic = %T.loc10_17.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc10_32.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_32.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_32.2: type = converted %T.ref, %T.as_type.loc10_32.1 [symbolic = %T.as_type.loc10_32.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
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
// CHECK:STDOUT: generic fn @FeedAnimal(%T.loc8_15.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc8_15.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_15.2: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_30.2: type = facet_access_type %T.loc8_15.2 [symbolic = %T.as_type.loc8_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @FeedAnimal.%T.as_type.loc8_30.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_15.1: %Animal.type](%a.param_patt: @FeedAnimal.%T.as_type.loc8_30.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc10_17.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc10_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_17.2: %Animal.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc10_32.2: type = facet_access_type %T.loc10_17.2 [symbolic = %T.as_type.loc10_32.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.loc10_37.2: <specific function> = specific_function constants.%FeedAnimal, @FeedAnimal(%T.loc10_17.2) [symbolic = %FeedAnimal.specific_fn.loc10_37.2 (constants.%FeedAnimal.specific_fn.ec8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc10_17.1: %Animal.type](%a.param_patt: @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedAnimal.ref: %FeedAnimal.type = name_ref FeedAnimal, file.%FeedAnimal.decl [concrete = constants.%FeedAnimal]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc10_32.2 (%T.as_type) = name_ref a, %a
// CHECK:STDOUT:     %.loc10_49.1: %Animal.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc10_17.2 (constants.%T)]
// CHECK:STDOUT:     %.loc10_49.2: %Animal.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc10_17.2 (constants.%T)]
// CHECK:STDOUT:     %FeedAnimal.specific_fn.loc10_37.1: <specific function> = specific_function %FeedAnimal.ref, @FeedAnimal(constants.%T) [symbolic = %FeedAnimal.specific_fn.loc10_37.2 (constants.%FeedAnimal.specific_fn.ec8)]
// CHECK:STDOUT:     %FeedAnimal.call: init %empty_tuple.type = call %FeedAnimal.specific_fn.loc10_37.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc16_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc16_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc16_17.3: init %Goat = class_init (), %.loc16_17.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc16_17.4: ref %Goat = temporary %.loc16_17.2, %.loc16_17.3
// CHECK:STDOUT:   %.loc16_19.1: ref %Goat = converted %.loc16_17.1, %.loc16_17.4
// CHECK:STDOUT:   %Animal.facet.loc16_26.1: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc16_26.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc16_26.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc16_26.2: %Animal.type = facet_value constants.%Goat, (constants.%impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc16_26.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc16_26.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc16_19.2: %Goat = bind_value %.loc16_19.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc16_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_30.2 => constants.%T.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T) {
// CHECK:STDOUT:   %T.loc10_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_17.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc10_32.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(@HandleAnimal.%T.loc10_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.loc10_17.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.patt.loc10_17.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc10_32.2 => constants.%Goat
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %FeedAnimal.specific_fn.loc10_37.2 => constants.%FeedAnimal.specific_fn.5d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc8_30.2 => constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:

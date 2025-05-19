// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_value_to_itself.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_value_to_itself.carbon

interface Animal {}

fn FeedAnimal[T:! Animal](a: T) {}

fn HandleAnimal[T:! Animal](a: T) { FeedAnimal(a); }

class Goat {}
impl Goat as Animal {}

fn F() {
  HandleAnimal({} as Goat);
}

// CHECK:STDOUT: --- convert_facet_value_value_to_itself.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b0: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.36a: type = pattern_type %T.as_type [symbolic]
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
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness file.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.ab7: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet) [concrete]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.82e: <specific function> = specific_function %FeedAnimal, @FeedAnimal(%Animal.facet) [concrete]
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
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .FeedAnimal = %FeedAnimal.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %FeedAnimal.decl: %FeedAnimal.type = fn_decl @FeedAnimal [concrete = constants.%FeedAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @FeedAnimal.%pattern_type (%pattern_type.36a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @FeedAnimal.%pattern_type (%pattern_type.36a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc15_15.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @FeedAnimal.%T.as_type.loc15_30.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_30.1: type = splice_block %.loc15_30.2 [symbolic = %T.as_type.loc15_30.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc15_15.1 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc15_30.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc15_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_30.2: type = converted %T.ref, %T.as_type.loc15_30.1 [symbolic = %T.as_type.loc15_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @FeedAnimal.%T.as_type.loc15_30.2 (%T.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.36a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.36a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc17_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_17.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc17_32.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_32.1: type = splice_block %.loc17_32.2 [symbolic = %T.as_type.loc17_32.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc17_17.1 [symbolic = %T.loc17_17.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc17_32.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc17_32.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc17_32.2: type = converted %T.ref, %T.as_type.loc17_32.1 [symbolic = %T.as_type.loc17_32.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc17_32.2 (%T.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
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
// CHECK:STDOUT:   witness = file.%Animal.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedAnimal(%T.loc15_15.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc15_15.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc15_30.2: type = facet_access_type %T.loc15_15.2 [symbolic = %T.as_type.loc15_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc15_30.2 [symbolic = %pattern_type (constants.%pattern_type.36a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc15_30.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @FeedAnimal.%T.as_type.loc15_30.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc17_17.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc17_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc17_32.2: type = facet_access_type %T.loc17_17.2 [symbolic = %T.as_type.loc17_32.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc17_32.2 [symbolic = %pattern_type (constants.%pattern_type.36a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc17_32.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.loc17_37.2: <specific function> = specific_function constants.%FeedAnimal, @FeedAnimal(%T.loc17_17.2) [symbolic = %FeedAnimal.specific_fn.loc17_37.2 (constants.%FeedAnimal.specific_fn.ec8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc17_32.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedAnimal.ref: %FeedAnimal.type = name_ref FeedAnimal, file.%FeedAnimal.decl [concrete = constants.%FeedAnimal]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc17_32.2 (%T.as_type) = name_ref a, %a
// CHECK:STDOUT:     %.loc17_49.1: %Animal.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc17_17.2 (constants.%T)]
// CHECK:STDOUT:     %.loc17_49.2: %Animal.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc17_17.2 (constants.%T)]
// CHECK:STDOUT:     %FeedAnimal.specific_fn.loc17_37.1: <specific function> = specific_function %FeedAnimal.ref, @FeedAnimal(constants.%T) [symbolic = %FeedAnimal.specific_fn.loc17_37.2 (constants.%FeedAnimal.specific_fn.ec8)]
// CHECK:STDOUT:     %FeedAnimal.call: init %empty_tuple.type = call %FeedAnimal.specific_fn.loc17_37.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc23_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc23_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc23_17.3: init %Goat = class_init (), %.loc23_17.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc23_17.4: ref %Goat = temporary %.loc23_17.2, %.loc23_17.3
// CHECK:STDOUT:   %.loc23_19.1: ref %Goat = converted %.loc23_17.1, %.loc23_17.4
// CHECK:STDOUT:   %Animal.facet.loc23_26.1: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc23_26.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc23_26.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc23_26.2: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc23_26.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc23_26.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc23_19.2: %Goat = bind_value %.loc23_19.1
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc23_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(constants.%T) {
// CHECK:STDOUT:   %T.loc15_15.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc15_30.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T) {
// CHECK:STDOUT:   %T.loc17_17.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc17_32.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.loc17_17.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc17_32.2 => constants.%Goat
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ab7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %FeedAnimal.specific_fn.loc17_37.2 => constants.%FeedAnimal.specific_fn.82e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.loc15_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc15_30.2 => constants.%Goat
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ab7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

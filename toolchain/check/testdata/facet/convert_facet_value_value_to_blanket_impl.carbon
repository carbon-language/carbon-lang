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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_facet_value_value_to_blanket_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_facet_value_value_to_blanket_impl.carbon

interface Eats {}
interface Animal {}

impl forall [A:! Animal] A as Eats {}

fn Feed[T:! Eats](e: T) {}

fn HandleAnimal[T:! Animal](a: T) { Feed(a); }

// CHECK:STDOUT: --- convert_facet_value_value_to_blanket_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b0: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.8abeaf.1: <witness> = impl_witness file.%Eats.impl_witness_table, @A.as_type.as.Eats.impl(%A) [symbolic]
// CHECK:STDOUT:   %T.1b5: %Eats.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a05: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.as_type.27d: type = facet_access_type %T.1b5 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b4: type = pattern_type %T.as_type.27d [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c75: <witness> = require_complete_type %T.as_type.27d [symbolic]
// CHECK:STDOUT:   %T.fd4: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.2ad: type = facet_access_type %T.fd4 [symbolic]
// CHECK:STDOUT:   %pattern_type.36a: type = pattern_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.234: <witness> = require_complete_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.8abeaf.2: <witness> = impl_witness file.%Eats.impl_witness_table, @A.as_type.as.Eats.impl(%T.fd4) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.fd4, @Eats [symbolic]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %T.as_type.2ad, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @A.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %A.patt: %pattern_type.3b0 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %Animal.type = name_ref A, %A.loc18_14.1 [symbolic = %A.loc18_14.2 (constants.%A)]
// CHECK:STDOUT:     %A.as_type.loc18_26.1: type = facet_access_type %A.ref [symbolic = %A.as_type.loc18_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %.loc18: type = converted %A.ref, %A.as_type.loc18_26.1 [symbolic = %A.as_type.loc18_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %A.loc18_14.1: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc18_14.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (), @A.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table, @A.as_type.as.Eats.impl(constants.%A) [symbolic = @A.as_type.as.Eats.impl.%Eats.impl_witness (constants.%Eats.impl_witness.8abeaf.1)]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a05 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.2b4) = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.2b4) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %T.loc20_9.2: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc20_9.1 (constants.%T.1b5)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc20_22.1 (%T.as_type.27d) = value_param call_param0
// CHECK:STDOUT:     %.loc20_22.1: type = splice_block %.loc20_22.2 [symbolic = %T.as_type.loc20_22.1 (constants.%T.as_type.27d)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc20_9.2 [symbolic = %T.loc20_9.1 (constants.%T.1b5)]
// CHECK:STDOUT:       %T.as_type.loc20_22.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc20_22.1 (constants.%T.as_type.27d)]
// CHECK:STDOUT:       %.loc20_22.2: type = converted %T.ref, %T.as_type.loc20_22.2 [symbolic = %T.as_type.loc20_22.1 (constants.%T.as_type.27d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc20_22.1 (%T.as_type.27d) = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.36a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.36a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc22_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.2ad) = value_param call_param0
// CHECK:STDOUT:     %.loc22_32.1: type = splice_block %.loc22_32.2 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc22_17.2 [symbolic = %T.loc22_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:       %T.as_type.loc22_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:       %.loc22_32.2: type = converted %T.ref, %T.as_type.loc22_32.2 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.2ad) = bind_name a, %a.param
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
// CHECK:STDOUT: generic impl @A.as_type.as.Eats.impl(%A.loc18_14.1: %Animal.type) {
// CHECK:STDOUT:   %A.loc18_14.2: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc18_14.2 (constants.%A)]
// CHECK:STDOUT:   %A.as_type.loc18_26.2: type = facet_access_type %A.loc18_14.2 [symbolic = %A.as_type.loc18_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table, @A.as_type.as.Eats.impl(%A.loc18_14.2) [symbolic = %Eats.impl_witness (constants.%Eats.impl_witness.8abeaf.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc18 as %Eats.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Eats.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc20_9.2: %Eats.type) {
// CHECK:STDOUT:   %T.loc20_9.1: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc20_9.1 (constants.%T.1b5)]
// CHECK:STDOUT:   %T.as_type.loc20_22.1: type = facet_access_type %T.loc20_9.1 [symbolic = %T.as_type.loc20_22.1 (constants.%T.as_type.27d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc20_22.1 [symbolic = %pattern_type (constants.%pattern_type.2b4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc20_22.1 [symbolic = %require_complete (constants.%require_complete.c75)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc20_22.1 (%T.as_type.27d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc22_17.2: %Animal.type) {
// CHECK:STDOUT:   %T.loc22_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.as_type.loc22_32.1: type = facet_access_type %T.loc22_17.1 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc22_32.1 [symbolic = %pattern_type (constants.%pattern_type.36a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc22_32.1 [symbolic = %require_complete (constants.%require_complete.234)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_17.1, @Eats [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc22_43.3: %Eats.type = facet_value %T.as_type.loc22_32.1, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:   %Feed.specific_fn.loc22_37.2: <specific function> = specific_function constants.%Feed, @Feed(%Eats.facet.loc22_43.3) [symbolic = %Feed.specific_fn.loc22_37.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.2ad)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.2ad) = name_ref a, %a
// CHECK:STDOUT:     %T.as_type.loc22_43.1: type = facet_access_type constants.%T.fd4 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc22_43.1: type = converted constants.%T.fd4, %T.as_type.loc22_43.1 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc22_43.2: %Animal.type = converted %.loc22_43.1, constants.%T.fd4 [symbolic = %T.loc22_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:     %Eats.facet.loc22_43.1: %Eats.type = facet_value constants.%T.as_type.2ad, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc22_43.3: %Eats.type = converted constants.%T.as_type.2ad, %Eats.facet.loc22_43.1 [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %T.as_type.loc22_43.2: type = facet_access_type constants.%T.fd4 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc22_43.4: type = converted constants.%T.fd4, %T.as_type.loc22_43.2 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     %.loc22_43.5: %Animal.type = converted %.loc22_43.4, constants.%T.fd4 [symbolic = %T.loc22_17.1 (constants.%T.fd4)]
// CHECK:STDOUT:     %Eats.facet.loc22_43.2: %Eats.type = facet_value constants.%T.as_type.2ad, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc22_43.6: %Eats.type = converted constants.%T.as_type.2ad, %Eats.facet.loc22_43.2 [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %Feed.specific_fn.loc22_37.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [symbolic = %Feed.specific_fn.loc22_37.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc22_37.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as_type.as.Eats.impl(constants.%A) {
// CHECK:STDOUT:   %A.loc18_14.2 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc18_26.2 => constants.%A.as_type
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.8abeaf.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.1b5) {
// CHECK:STDOUT:   %T.loc20_9.1 => constants.%T.1b5
// CHECK:STDOUT:   %T.as_type.loc20_22.1 => constants.%T.as_type.27d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.fd4) {
// CHECK:STDOUT:   %T.loc22_17.1 => constants.%T.fd4
// CHECK:STDOUT:   %T.as_type.loc22_32.1 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as_type.as.Eats.impl(constants.%T.fd4) {
// CHECK:STDOUT:   %A.loc18_14.2 => constants.%T.fd4
// CHECK:STDOUT:   %A.as_type.loc18_26.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.8abeaf.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %T.loc20_9.1 => constants.%Eats.facet
// CHECK:STDOUT:   %T.as_type.loc20_22.1 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.234
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %Self.267: %Eats.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.84c: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.62c94e.1: <witness> = impl_witness file.%Eats.impl_witness_table, @A.binding.as_type.as.Eats.impl(%A) [symbolic]
// CHECK:STDOUT:   %T.267: %Eats.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.384: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.e28: type = symbolic_binding_type T, 0, %T.267 [symbolic]
// CHECK:STDOUT:   %pattern_type.683: type = pattern_type %T.binding.as_type.e28 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.639: <witness> = require_complete_type %T.binding.as_type.e28 [symbolic]
// CHECK:STDOUT:   %T.84c: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.5cd: type = symbolic_binding_type T, 0, %T.84c [symbolic]
// CHECK:STDOUT:   %pattern_type.01b: type = pattern_type %T.binding.as_type.5cd [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c0f: <witness> = require_complete_type %T.binding.as_type.5cd [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.62c94e.2: <witness> = impl_witness file.%Eats.impl_witness_table, @A.binding.as_type.as.Eats.impl(%T.84c) [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.84c, @Eats [symbolic]
// CHECK:STDOUT:   %Eats.facet: %Eats.type = facet_value %T.binding.as_type.5cd, (%Eats.lookup_impl_witness) [symbolic]
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
// CHECK:STDOUT:   impl_decl @A.binding.as_type.as.Eats.impl [concrete] {
// CHECK:STDOUT:     %A.patt: %pattern_type.3c3 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %Animal.type = name_ref A, %A.loc18_14.1 [symbolic = %A.loc18_14.2 (constants.%A)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     %.loc18_26: type = converted %A.ref, %A.as_type [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %.loc18_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc18_14.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc18_14.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eats.impl_witness_table = impl_witness_table (), @A.binding.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness %Eats.impl_witness_table, @A.binding.as_type.as.Eats.impl(constants.%A) [symbolic = @A.binding.as_type.as.Eats.impl.%Eats.impl_witness (constants.%Eats.impl_witness.62c94e.1)]
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt: %pattern_type.384 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.683) = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.683) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_13: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_9.2: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc20_9.1 (constants.%T.267)]
// CHECK:STDOUT:     %e.param: @Feed.%T.binding.as_type (%T.binding.as_type.e28) = value_param call_param0
// CHECK:STDOUT:     %.loc20_22.1: type = splice_block %.loc20_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc20_9.2 [symbolic = %T.loc20_9.1 (constants.%T.267)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)]
// CHECK:STDOUT:       %.loc20_22.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.binding.as_type (%T.binding.as_type.e28) = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3c3 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.01b) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.01b) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_17.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc22_17.1 (constants.%T.84c)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd) = value_param call_param0
// CHECK:STDOUT:     %.loc22_32.1: type = splice_block %.loc22_32.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc22_17.2 [symbolic = %T.loc22_17.1 (constants.%T.84c)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:       %.loc22_32.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = symbolic_binding Self, 0 [symbolic = constants.%Self.267]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.84c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.binding.as_type.as.Eats.impl(%A.loc18_14.1: %Animal.type) {
// CHECK:STDOUT:   %A.loc18_14.2: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc18_14.2 (constants.%A)]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A.loc18_14.2 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:   %Eats.impl_witness: <witness> = impl_witness file.%Eats.impl_witness_table, @A.binding.as_type.as.Eats.impl(%A.loc18_14.2) [symbolic = %Eats.impl_witness (constants.%Eats.impl_witness.62c94e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc18_26 as %Eats.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Eats.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc20_9.2: %Eats.type) {
// CHECK:STDOUT:   %T.loc20_9.1: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc20_9.1 (constants.%T.267)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc20_9.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.683)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.639)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.binding.as_type (%T.binding.as_type.e28)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc22_17.2: %Animal.type) {
// CHECK:STDOUT:   %T.loc22_17.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc22_17.1 (constants.%T.84c)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc22_17.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.01b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.c0f)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_17.1, @Eats [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc22_43.3: %Eats.type = facet_value %T.binding.as_type, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:   %Feed.specific_fn.loc22_37.2: <specific function> = specific_function constants.%Feed, @Feed(%Eats.facet.loc22_43.3) [symbolic = %Feed.specific_fn.loc22_37.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd) = name_ref a, %a
// CHECK:STDOUT:     %Eats.facet.loc22_43.1: %Eats.type = facet_value constants.%T.binding.as_type.5cd, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc22_43.1: %Eats.type = converted constants.%T.binding.as_type.5cd, %Eats.facet.loc22_43.1 [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %Eats.facet.loc22_43.2: %Eats.type = facet_value constants.%T.binding.as_type.5cd, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %.loc22_43.2: %Eats.type = converted constants.%T.binding.as_type.5cd, %Eats.facet.loc22_43.2 [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet)]
// CHECK:STDOUT:     %Feed.specific_fn.loc22_37.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet) [symbolic = %Feed.specific_fn.loc22_37.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc22_37.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.binding.as_type.as.Eats.impl(constants.%A) {
// CHECK:STDOUT:   %A.loc18_14.2 => constants.%A
// CHECK:STDOUT:   %A.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.62c94e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.267) {
// CHECK:STDOUT:   %T.loc20_9.1 => constants.%T.267
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.e28
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.84c) {
// CHECK:STDOUT:   %T.loc22_17.1 => constants.%T.84c
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.5cd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.01b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.binding.as_type.as.Eats.impl(constants.%T.84c) {
// CHECK:STDOUT:   %A.loc18_14.2 => constants.%T.84c
// CHECK:STDOUT:   %A.binding.as_type => constants.%T.binding.as_type.5cd
// CHECK:STDOUT:   %Eats.impl_witness => constants.%Eats.impl_witness.62c94e.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet) {
// CHECK:STDOUT:   %T.loc20_9.1 => constants.%Eats.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.5cd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.01b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c0f
// CHECK:STDOUT: }
// CHECK:STDOUT:

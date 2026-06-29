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

fn Feed[T:! Eats](unused e: T) {}

fn HandleAnimal[T:! Animal](a: T) { Feed(a); }

// CHECK:STDOUT: --- convert_facet_value_value_to_blanket_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.23e: %Eats.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.f00: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.333: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %A.patt: %pattern_type.333 = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.e0811c.1: <witness> = impl_witness @A.as_type.as.Eats.impl.%Eats.impl_witness_table, @A.as_type.as.Eats.impl(%A) [symbolic]
// CHECK:STDOUT:   %Eats.facet.f93: %Eats.type = facet_value %A.as_type, (%Eats.impl_witness.e0811c.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.880: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.patt.9e8: %pattern_type.880 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.672: %Eats.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.422: type = facet_access_type %T.672 [symbolic]
// CHECK:STDOUT:   %pattern_type.aed: type = pattern_type %T.as_type.422 [symbolic]
// CHECK:STDOUT:   %e.param_patt.0be: %pattern_type.aed = value_param_pattern [symbolic]
// CHECK:STDOUT:   %e.patt.543: %pattern_type.aed = at_binding_pattern e, %e.param_patt.0be [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c4d: <witness> = require_complete_type %T.as_type.422 [symbolic]
// CHECK:STDOUT:   %T.patt.e44: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.443: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.01d: type = facet_access_type %T.443 [symbolic]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %T.as_type.01d [symbolic]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.e90 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.e90 = at_binding_pattern a, %a.param_patt [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.db4: <witness> = require_complete_type %T.as_type.01d [symbolic]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.443, @Eats [symbolic]
// CHECK:STDOUT:   %Eats.impl_witness.e0811c.2: <witness> = impl_witness @A.as_type.as.Eats.impl.%Eats.impl_witness_table, @A.as_type.as.Eats.impl(%T.443) [symbolic]
// CHECK:STDOUT:   %.2fe: require_specific_def_type = require_specific_def @A.as_type.as.Eats.impl(%T.443) [symbolic]
// CHECK:STDOUT:   %Eats.facet.934: %Eats.type = facet_value %T.as_type.01d, (%Eats.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %e.param_patt.1c7: %pattern_type.e90 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %e.patt.637: %pattern_type.e90 = at_binding_pattern e, %e.param_patt.1c7 [symbolic]
// CHECK:STDOUT:   %Feed.specific_fn: <specific function> = specific_function %Feed, @Feed(%Eats.facet.934) [symbolic]
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
// CHECK:STDOUT:     %A.patt.loc18_15.1: %pattern_type.333 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc18_15.2 (constants.%A.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %Animal.type = name_ref A, %A.loc18_15.1 [symbolic = %A.loc18_15.2 (constants.%A)]
// CHECK:STDOUT:     %A.as_type.loc18_26.1: type = facet_access_type %A.ref [symbolic = %A.as_type.loc18_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %.loc18_26: type = converted %A.ref, %A.as_type.loc18_26.1 [symbolic = %A.as_type.loc18_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %.loc18_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc18_15.1: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc18_15.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt.loc20_10.1: %pattern_type.880 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_10.2 (constants.%T.patt.9e8)]
// CHECK:STDOUT:     %e.param_patt.loc20_27.1: @Feed.%pattern_type (%pattern_type.aed) = value_param_pattern [symbolic = %e.param_patt.loc20_27.2 (constants.%e.param_patt.0be)]
// CHECK:STDOUT:     %e.patt.loc20_27.1: @Feed.%pattern_type (%pattern_type.aed) = at_binding_pattern e, %e.param_patt.loc20_27.1 [symbolic = %e.patt.loc20_27.2 (constants.%e.patt.543)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_13: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_10.2: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc20_10.1 (constants.%T.672)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc20_29.1 (%T.as_type.422) = value_param call_param0
// CHECK:STDOUT:     %.loc20_29.1: type = splice_block %.loc20_29.2 [symbolic = %T.as_type.loc20_29.1 (constants.%T.as_type.422)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc20_10.2 [symbolic = %T.loc20_10.1 (constants.%T.672)]
// CHECK:STDOUT:       %T.as_type.loc20_29.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc20_29.1 (constants.%T.as_type.422)]
// CHECK:STDOUT:       %.loc20_29.2: type = converted %T.ref, %T.as_type.loc20_29.2 [symbolic = %T.as_type.loc20_29.1 (constants.%T.as_type.422)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc20_29.1 (%T.as_type.422) = wrapper_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt.loc22_18.1: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_18.2 (constants.%T.patt.e44)]
// CHECK:STDOUT:     %a.param_patt.loc22_30.1: @HandleAnimal.%pattern_type (%pattern_type.e90) = value_param_pattern [symbolic = %a.param_patt.loc22_30.2 (constants.%a.param_patt)]
// CHECK:STDOUT:     %a.patt.loc22_30.1: @HandleAnimal.%pattern_type (%pattern_type.e90) = at_binding_pattern a, %a.param_patt.loc22_30.1 [symbolic = %a.patt.loc22_30.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_18.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc22_18.1 (constants.%T.443)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.01d) = value_param call_param0
// CHECK:STDOUT:     %.loc22_32.1: type = splice_block %.loc22_32.2 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.01d)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc22_18.2 [symbolic = %T.loc22_18.1 (constants.%T.443)]
// CHECK:STDOUT:       %T.as_type.loc22_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:       %.loc22_32.2: type = converted %T.ref, %T.as_type.loc22_32.2 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.01d) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = symbolic_binding Self, 0 [symbolic = constants.%Self.23e]
// CHECK:STDOUT:   %Eats.WithSelf.decl = interface_with_self_decl @Eats [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f00]
// CHECK:STDOUT:   %Animal.WithSelf.decl = interface_with_self_decl @Animal [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as_type.as.Eats.impl(%A.loc18_15.1: %Animal.type) {
// CHECK:STDOUT:   %A.patt.loc18_15.2: %pattern_type.333 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc18_15.2 (constants.%A.patt)]
// CHECK:STDOUT:   %A.loc18_15.2: %Animal.type = symbolic_binding A, 0 [symbolic = %A.loc18_15.2 (constants.%A)]
// CHECK:STDOUT:   %A.as_type.loc18_26.2: type = facet_access_type %A.loc18_15.2 [symbolic = %A.as_type.loc18_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:   %Eats.impl_witness.loc18_36.2: <witness> = impl_witness %Eats.impl_witness_table, @A.as_type.as.Eats.impl(%A.loc18_15.2) [symbolic = %Eats.impl_witness.loc18_36.2 (constants.%Eats.impl_witness.e0811c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc18_26 as %Eats.ref {
// CHECK:STDOUT:     %Eats.impl_witness_table = impl_witness_table (), @A.as_type.as.Eats.impl [concrete]
// CHECK:STDOUT:     %Eats.impl_witness.loc18_36.1: <witness> = impl_witness %Eats.impl_witness_table, @A.as_type.as.Eats.impl(constants.%A) [symbolic = %Eats.impl_witness.loc18_36.2 (constants.%Eats.impl_witness.e0811c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Eats.ref
// CHECK:STDOUT:     witness = %Eats.impl_witness.loc18_36.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc20_10.2: %Eats.type) {
// CHECK:STDOUT:   %T.patt.loc20_10.2: %pattern_type.880 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_10.2 (constants.%T.patt.9e8)]
// CHECK:STDOUT:   %T.loc20_10.1: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc20_10.1 (constants.%T.672)]
// CHECK:STDOUT:   %T.as_type.loc20_29.1: type = facet_access_type %T.loc20_10.1 [symbolic = %T.as_type.loc20_29.1 (constants.%T.as_type.422)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc20_29.1 [symbolic = %pattern_type (constants.%pattern_type.aed)]
// CHECK:STDOUT:   %e.param_patt.loc20_27.2: @Feed.%pattern_type (%pattern_type.aed) = value_param_pattern [symbolic = %e.param_patt.loc20_27.2 (constants.%e.param_patt.0be)]
// CHECK:STDOUT:   %e.patt.loc20_27.2: @Feed.%pattern_type (%pattern_type.aed) = at_binding_pattern e, %e.param_patt.loc20_27.2 [symbolic = %e.patt.loc20_27.2 (constants.%e.patt.543)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc20_29.1 [symbolic = %require_complete (constants.%require_complete.c4d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc20_29.1 (%T.as_type.422)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc22_18.2: %Animal.type) {
// CHECK:STDOUT:   %T.patt.loc22_18.2: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_18.2 (constants.%T.patt.e44)]
// CHECK:STDOUT:   %T.loc22_18.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc22_18.1 (constants.%T.443)]
// CHECK:STDOUT:   %T.as_type.loc22_32.1: type = facet_access_type %T.loc22_18.1 [symbolic = %T.as_type.loc22_32.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc22_32.1 [symbolic = %pattern_type (constants.%pattern_type.e90)]
// CHECK:STDOUT:   %a.param_patt.loc22_30.2: @HandleAnimal.%pattern_type (%pattern_type.e90) = value_param_pattern [symbolic = %a.param_patt.loc22_30.2 (constants.%a.param_patt)]
// CHECK:STDOUT:   %a.patt.loc22_30.2: @HandleAnimal.%pattern_type (%pattern_type.e90) = at_binding_pattern a, %a.param_patt.loc22_30.2 [symbolic = %a.patt.loc22_30.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc22_32.1 [symbolic = %require_complete (constants.%require_complete.db4)]
// CHECK:STDOUT:   %.loc22_43.3: require_specific_def_type = require_specific_def @A.as_type.as.Eats.impl(%T.loc22_18.1) [symbolic = %.loc22_43.3 (constants.%.2fe)]
// CHECK:STDOUT:   %Eats.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_18.1, @Eats [symbolic = %Eats.lookup_impl_witness (constants.%Eats.lookup_impl_witness)]
// CHECK:STDOUT:   %Eats.facet.loc22_43.3: %Eats.type = facet_value %T.as_type.loc22_32.1, (%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet.934)]
// CHECK:STDOUT:   %Feed.specific_fn.loc22_37.2: <specific function> = specific_function constants.%Feed, @Feed(%Eats.facet.loc22_43.3) [symbolic = %Feed.specific_fn.loc22_37.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.01d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc22_32.1 (%T.as_type.01d) = name_ref a, %a
// CHECK:STDOUT:     %Eats.facet.loc22_43.1: %Eats.type = facet_value constants.%T.as_type.01d, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet.934)]
// CHECK:STDOUT:     %.loc22_43.1: %Eats.type = converted constants.%T.as_type.01d, %Eats.facet.loc22_43.1 [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet.934)]
// CHECK:STDOUT:     %Eats.facet.loc22_43.2: %Eats.type = facet_value constants.%T.as_type.01d, (constants.%Eats.lookup_impl_witness) [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet.934)]
// CHECK:STDOUT:     %.loc22_43.2: %Eats.type = converted constants.%T.as_type.01d, %Eats.facet.loc22_43.2 [symbolic = %Eats.facet.loc22_43.3 (constants.%Eats.facet.934)]
// CHECK:STDOUT:     %Feed.specific_fn.loc22_37.1: <specific function> = specific_function %Feed.ref, @Feed(constants.%Eats.facet.934) [symbolic = %Feed.specific_fn.loc22_37.2 (constants.%Feed.specific_fn)]
// CHECK:STDOUT:     %Feed.call: init %empty_tuple.type = call %Feed.specific_fn.loc22_37.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Self.23e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Self.f00) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as_type.as.Eats.impl(constants.%A) {
// CHECK:STDOUT:   %A.patt.loc18_15.2 => constants.%A.patt
// CHECK:STDOUT:   %A.loc18_15.2 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc18_26.2 => constants.%A.as_type
// CHECK:STDOUT:   %Eats.impl_witness.loc18_36.2 => constants.%Eats.impl_witness.e0811c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Eats.facet.f93) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.672) {
// CHECK:STDOUT:   %T.patt.loc20_10.2 => constants.%T.patt.9e8
// CHECK:STDOUT:   %T.loc20_10.1 => constants.%T.672
// CHECK:STDOUT:   %T.as_type.loc20_29.1 => constants.%T.as_type.422
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.aed
// CHECK:STDOUT:   %e.param_patt.loc20_27.2 => constants.%e.param_patt.0be
// CHECK:STDOUT:   %e.patt.loc20_27.2 => constants.%e.patt.543
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.443) {
// CHECK:STDOUT:   %T.patt.loc22_18.2 => constants.%T.patt.e44
// CHECK:STDOUT:   %T.loc22_18.1 => constants.%T.443
// CHECK:STDOUT:   %T.as_type.loc22_32.1 => constants.%T.as_type.01d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e90
// CHECK:STDOUT:   %a.param_patt.loc22_30.2 => constants.%a.param_patt
// CHECK:STDOUT:   %a.patt.loc22_30.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as_type.as.Eats.impl(constants.%T.443) {
// CHECK:STDOUT:   %A.patt.loc18_15.2 => constants.%A.patt
// CHECK:STDOUT:   %A.loc18_15.2 => constants.%T.443
// CHECK:STDOUT:   %A.as_type.loc18_26.2 => constants.%T.as_type.01d
// CHECK:STDOUT:   %Eats.impl_witness.loc18_36.2 => constants.%Eats.impl_witness.e0811c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%Eats.facet.934) {
// CHECK:STDOUT:   %T.patt.loc20_10.2 => constants.%T.patt.9e8
// CHECK:STDOUT:   %T.loc20_10.1 => constants.%Eats.facet.934
// CHECK:STDOUT:   %T.as_type.loc20_29.1 => constants.%T.as_type.01d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e90
// CHECK:STDOUT:   %e.param_patt.loc20_27.2 => constants.%e.param_patt.1c7
// CHECK:STDOUT:   %e.patt.loc20_27.2 => constants.%e.patt.637
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.db4
// CHECK:STDOUT: }
// CHECK:STDOUT:

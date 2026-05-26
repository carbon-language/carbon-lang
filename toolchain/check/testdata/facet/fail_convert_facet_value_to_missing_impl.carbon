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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_convert_facet_value_to_missing_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_convert_facet_value_to_missing_impl.carbon

interface Eats {}
interface Animal {}

fn Feed[T:! Eats](unused e: T) {}

// CHECK:STDERR: fail_convert_facet_value_to_missing_impl.carbon:[[@LINE+7]]:37: error: cannot convert type `T` that implements `Animal` into type implementing `Eats` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn HandleAnimal[T:! Animal](a: T) { Feed(a); }
// CHECK:STDERR:                                     ^~~~~~~
// CHECK:STDERR: fail_convert_facet_value_to_missing_impl.carbon:[[@LINE-5]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn Feed[T:! Eats](unused e: T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn HandleAnimal[T:! Animal](a: T) { Feed(a); }

// CHECK:STDOUT: --- fail_convert_facet_value_to_missing_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.24a: %Eats.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.d73: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.42a: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.ad0: %Eats.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.f11: type = facet_access_type %T.ad0 [symbolic]
// CHECK:STDOUT:   %pattern_type.434: type = pattern_type %T.as_type.f11 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5df: <witness> = require_complete_type %T.as_type.f11 [symbolic]
// CHECK:STDOUT:   %pattern_type.69f: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.123: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.6eb: type = facet_access_type %T.123 [symbolic]
// CHECK:STDOUT:   %pattern_type.106: type = pattern_type %T.as_type.6eb [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.284: <witness> = require_complete_type %T.as_type.6eb [symbolic]
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
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt: %pattern_type.42a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.434) = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.434) = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_13: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_10.2: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc18_10.1 (constants.%T.ad0)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc18_29.1 (%T.as_type.f11) = value_param call_param0
// CHECK:STDOUT:     %.loc18_29.1: type = splice_block %.loc18_29.2 [symbolic = %T.as_type.loc18_29.1 (constants.%T.as_type.f11)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc18_10.2 [symbolic = %T.loc18_10.1 (constants.%T.ad0)]
// CHECK:STDOUT:       %T.as_type.loc18_29.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_29.1 (constants.%T.as_type.f11)]
// CHECK:STDOUT:       %.loc18_29.2: type = converted %T.ref, %T.as_type.loc18_29.2 [symbolic = %T.as_type.loc18_29.1 (constants.%T.as_type.f11)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc18_29.1 (%T.as_type.f11) = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.69f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.106) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.106) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_18.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc27_18.1 (constants.%T.123)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.6eb) = value_param call_param0
// CHECK:STDOUT:     %.loc27_32.1: type = splice_block %.loc27_32.2 [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.6eb)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc27_18.2 [symbolic = %T.loc27_18.1 (constants.%T.123)]
// CHECK:STDOUT:       %T.as_type.loc27_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.6eb)]
// CHECK:STDOUT:       %.loc27_32.2: type = converted %T.ref, %T.as_type.loc27_32.2 [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.6eb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.6eb) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = symbolic_binding Self, 0 [symbolic = constants.%Self.24a]
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
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d73]
// CHECK:STDOUT:   %Animal.WithSelf.decl = interface_with_self_decl @Animal [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc18_10.2: %Eats.type) {
// CHECK:STDOUT:   %T.loc18_10.1: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc18_10.1 (constants.%T.ad0)]
// CHECK:STDOUT:   %T.as_type.loc18_29.1: type = facet_access_type %T.loc18_10.1 [symbolic = %T.as_type.loc18_29.1 (constants.%T.as_type.f11)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc18_29.1 [symbolic = %pattern_type (constants.%pattern_type.434)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc18_29.1 [symbolic = %require_complete (constants.%require_complete.5df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc18_29.1 (%T.as_type.f11)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc27_18.2: %Animal.type) {
// CHECK:STDOUT:   %T.loc27_18.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc27_18.1 (constants.%T.123)]
// CHECK:STDOUT:   %T.as_type.loc27_32.1: type = facet_access_type %T.loc27_18.1 [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.6eb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc27_32.1 [symbolic = %pattern_type (constants.%pattern_type.106)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc27_32.1 [symbolic = %require_complete (constants.%require_complete.284)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.6eb)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.6eb) = name_ref a, %a
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Eats.WithSelf(constants.%Self.24a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Self.d73) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.ad0) {
// CHECK:STDOUT:   %T.loc18_10.1 => constants.%T.ad0
// CHECK:STDOUT:   %T.as_type.loc18_29.1 => constants.%T.as_type.f11
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.434
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.123) {
// CHECK:STDOUT:   %T.loc27_18.1 => constants.%T.123
// CHECK:STDOUT:   %T.as_type.loc27_32.1 => constants.%T.as_type.6eb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.106
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

fn Feed[T:! Eats](e: T) {}

// CHECK:STDERR: fail_convert_facet_value_to_missing_impl.carbon:[[@LINE+7]]:37: error: cannot convert type `T` that implements `Animal` into type implementing `Eats` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn HandleAnimal[T:! Animal](a: T) { Feed(a); }
// CHECK:STDERR:                                     ^~~~~~~
// CHECK:STDERR: fail_convert_facet_value_to_missing_impl.carbon:[[@LINE-5]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn Feed[T:! Eats](e: T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn HandleAnimal[T:! Animal](a: T) { Feed(a); }

// CHECK:STDOUT: --- fail_convert_facet_value_to_missing_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.267: %Eats.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.84c: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.267: %Eats.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.384: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.e28: type = symbolic_binding_type T, 0, %T.267 [symbolic]
// CHECK:STDOUT:   %pattern_type.683: type = pattern_type %T.binding.as_type.e28 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.639: <witness> = require_complete_type %T.binding.as_type.e28 [symbolic]
// CHECK:STDOUT:   %T.84c: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.5cd: type = symbolic_binding_type T, 0, %T.84c [symbolic]
// CHECK:STDOUT:   %pattern_type.01b: type = pattern_type %T.binding.as_type.5cd [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c0f: <witness> = require_complete_type %T.binding.as_type.5cd [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.384 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.683) = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.683) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_13: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_9.2: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc18_9.1 (constants.%T.267)]
// CHECK:STDOUT:     %e.param: @Feed.%T.binding.as_type (%T.binding.as_type.e28) = value_param call_param0
// CHECK:STDOUT:     %.loc18_22.1: type = splice_block %.loc18_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc18_9.2 [symbolic = %T.loc18_9.1 (constants.%T.267)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)]
// CHECK:STDOUT:       %.loc18_22.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.binding.as_type (%T.binding.as_type.e28) = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3c3 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.01b) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.01b) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_17.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.84c)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd) = value_param call_param0
// CHECK:STDOUT:     %.loc27_32.1: type = splice_block %.loc27_32.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc27_17.2 [symbolic = %T.loc27_17.1 (constants.%T.84c)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:       %.loc27_32.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
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
// CHECK:STDOUT: generic fn @Feed(%T.loc18_9.2: %Eats.type) {
// CHECK:STDOUT:   %T.loc18_9.1: %Eats.type = symbolic_binding T, 0 [symbolic = %T.loc18_9.1 (constants.%T.267)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc18_9.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.e28)]
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
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc27_17.2: %Animal.type) {
// CHECK:STDOUT:   %T.loc27_17.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.84c)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc27_17.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.5cd)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.01b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.c0f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.binding.as_type (%T.binding.as_type.5cd) = name_ref a, %a
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.267) {
// CHECK:STDOUT:   %T.loc18_9.1 => constants.%T.267
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.e28
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.84c) {
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%T.84c
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.5cd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.01b
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %Self.3c7: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.a8c: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.3c7: %Eats.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.08e: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.as_type.844: type = facet_access_type %T.3c7 [symbolic]
// CHECK:STDOUT:   %pattern_type.de8: type = pattern_type %T.as_type.844 [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a45: <witness> = require_complete_type %T.as_type.844 [symbolic]
// CHECK:STDOUT:   %T.a8c: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd6: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.as_type.b5e: type = facet_access_type %T.a8c [symbolic]
// CHECK:STDOUT:   %pattern_type.28a: type = pattern_type %T.as_type.b5e [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.29e: <witness> = require_complete_type %T.as_type.b5e [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.08e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.de8) = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.de8) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_13: type = splice_block %Eats.ref [concrete = constants.%Eats.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_9.2: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc18_9.1 (constants.%T.3c7)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc18_22.1 (%T.as_type.844) = value_param call_param0
// CHECK:STDOUT:     %.loc18_22.1: type = splice_block %.loc18_22.2 [symbolic = %T.as_type.loc18_22.1 (constants.%T.as_type.844)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc18_9.2 [symbolic = %T.loc18_9.1 (constants.%T.3c7)]
// CHECK:STDOUT:       %T.as_type.loc18_22.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_22.1 (constants.%T.as_type.844)]
// CHECK:STDOUT:       %.loc18_22.2: type = converted %T.ref, %T.as_type.loc18_22.2 [symbolic = %T.as_type.loc18_22.1 (constants.%T.as_type.844)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc18_22.1 (%T.as_type.844) = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.bd6 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.28a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.28a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_21: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc27_17.1 (constants.%T.a8c)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.b5e) = value_param call_param0
// CHECK:STDOUT:     %.loc27_32.1: type = splice_block %.loc27_32.2 [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.b5e)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc27_17.2 [symbolic = %T.loc27_17.1 (constants.%T.a8c)]
// CHECK:STDOUT:       %T.as_type.loc27_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.b5e)]
// CHECK:STDOUT:       %.loc27_32.2: type = converted %T.ref, %T.as_type.loc27_32.2 [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.b5e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.b5e) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3c7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.a8c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc18_9.2: %Eats.type) {
// CHECK:STDOUT:   %T.loc18_9.1: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc18_9.1 (constants.%T.3c7)]
// CHECK:STDOUT:   %T.as_type.loc18_22.1: type = facet_access_type %T.loc18_9.1 [symbolic = %T.as_type.loc18_22.1 (constants.%T.as_type.844)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc18_22.1 [symbolic = %pattern_type (constants.%pattern_type.de8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc18_22.1 [symbolic = %require_complete (constants.%require_complete.a45)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc18_22.1 (%T.as_type.844)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc27_17.2: %Animal.type) {
// CHECK:STDOUT:   %T.loc27_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc27_17.1 (constants.%T.a8c)]
// CHECK:STDOUT:   %T.as_type.loc27_32.1: type = facet_access_type %T.loc27_17.1 [symbolic = %T.as_type.loc27_32.1 (constants.%T.as_type.b5e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc27_32.1 [symbolic = %pattern_type (constants.%pattern_type.28a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc27_32.1 [symbolic = %require_complete (constants.%require_complete.29e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.b5e)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc27_32.1 (%T.as_type.b5e) = name_ref a, %a
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.3c7) {
// CHECK:STDOUT:   %T.loc18_9.1 => constants.%T.3c7
// CHECK:STDOUT:   %T.as_type.loc18_22.1 => constants.%T.as_type.844
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.de8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.a8c) {
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%T.a8c
// CHECK:STDOUT:   %T.as_type.loc27_32.1 => constants.%T.as_type.b5e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.28a
// CHECK:STDOUT: }
// CHECK:STDOUT:

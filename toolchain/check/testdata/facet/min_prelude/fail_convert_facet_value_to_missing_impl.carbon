// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/fail_convert_facet_value_to_missing_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/fail_convert_facet_value_to_missing_impl.carbon

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
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.1b5: %Eats.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a05: type = pattern_type %Eats.type [concrete]
// CHECK:STDOUT:   %T.as_type.27d: type = facet_access_type %T.1b5 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b4: type = pattern_type %T.as_type.27d [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c75: <witness> = require_complete_type %T.as_type.27d [symbolic]
// CHECK:STDOUT:   %T.fd4: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b0: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.as_type.2ad: type = facet_access_type %T.fd4 [symbolic]
// CHECK:STDOUT:   %pattern_type.36a: type = pattern_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.234: <witness> = require_complete_type %T.as_type.2ad [symbolic]
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
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a05 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %e.patt: @Feed.%pattern_type (%pattern_type.2b4) = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: @Feed.%pattern_type (%pattern_type.2b4) = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %T.loc16_9.1: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc16_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc16_22.2 (%T.as_type.27d) = value_param call_param0
// CHECK:STDOUT:     %.loc16_22.1: type = splice_block %.loc16_22.2 [symbolic = %T.as_type.loc16_22.2 (constants.%T.as_type.27d)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc16_9.1 [symbolic = %T.loc16_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:       %T.as_type.loc16_22.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc16_22.2 (constants.%T.as_type.27d)]
// CHECK:STDOUT:       %.loc16_22.2: type = converted %T.ref, %T.as_type.loc16_22.1 [symbolic = %T.as_type.loc16_22.2 (constants.%T.as_type.27d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc16_22.2 (%T.as_type.27d) = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @HandleAnimal.%pattern_type (%pattern_type.36a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @HandleAnimal.%pattern_type (%pattern_type.36a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc25_17.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc25_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc25_32.2 (%T.as_type.2ad) = value_param call_param0
// CHECK:STDOUT:     %.loc25_32.1: type = splice_block %.loc25_32.2 [symbolic = %T.as_type.loc25_32.2 (constants.%T.as_type.2ad)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc25_17.1 [symbolic = %T.loc25_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:       %T.as_type.loc25_32.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc25_32.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:       %.loc25_32.2: type = converted %T.ref, %T.as_type.loc25_32.1 [symbolic = %T.as_type.loc25_32.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc25_32.2 (%T.as_type.2ad) = bind_name a, %a.param
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
// CHECK:STDOUT: generic fn @Feed(%T.loc16_9.1: %Eats.type) {
// CHECK:STDOUT:   %T.loc16_9.2: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc16_9.2 (constants.%T.1b5)]
// CHECK:STDOUT:   %T.as_type.loc16_22.2: type = facet_access_type %T.loc16_9.2 [symbolic = %T.as_type.loc16_22.2 (constants.%T.as_type.27d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc16_22.2 [symbolic = %pattern_type (constants.%pattern_type.2b4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc16_22.2 [symbolic = %require_complete (constants.%require_complete.c75)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%e.param: @Feed.%T.as_type.loc16_22.2 (%T.as_type.27d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc25_17.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc25_17.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc25_17.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.as_type.loc25_32.2: type = facet_access_type %T.loc25_17.2 [symbolic = %T.as_type.loc25_32.2 (constants.%T.as_type.2ad)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc25_32.2 [symbolic = %pattern_type (constants.%pattern_type.36a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc25_32.2 [symbolic = %require_complete (constants.%require_complete.234)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc25_32.2 (%T.as_type.2ad)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc25_32.2 (%T.as_type.2ad) = name_ref a, %a
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T.1b5) {
// CHECK:STDOUT:   %T.loc16_9.2 => constants.%T.1b5
// CHECK:STDOUT:   %T.as_type.loc16_22.2 => constants.%T.as_type.27d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.fd4) {
// CHECK:STDOUT:   %T.loc25_17.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.as_type.loc25_32.2 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36a
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/generic_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/generic_import.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

interface AddWith(T:! type) {
  fn F();
}

// --- b.carbon

library "[[@TEST_NAME]]";
import library "a";

class C {}

impl C as AddWith(C) {
  fn F() {}
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %AddWith.type.d53: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.d53 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.eb8: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %AddWith.type.eb8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.F.type: type = fn_type @AddWith.F, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.F: %AddWith.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %AddWith.assoc_type = assoc_entity element0, @AddWith.%AddWith.F.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.d53 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc4_19.1)> [symbolic = %AddWith.type (constants.%AddWith.type.eb8)]
// CHECK:STDOUT:   %Self.loc4_29.2: @AddWith.%AddWith.type (%AddWith.type.eb8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self)]
// CHECK:STDOUT:   %AddWith.F.type: type = fn_type @AddWith.F, @AddWith(%T.loc4_19.1) [symbolic = %AddWith.F.type (constants.%AddWith.F.type)]
// CHECK:STDOUT:   %AddWith.F: @AddWith.%AddWith.F.type (%AddWith.F.type) = struct_value () [symbolic = %AddWith.F (constants.%AddWith.F)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T.loc4_19.1) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type) = assoc_entity element0, %AddWith.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_29.1: @AddWith.%AddWith.type (%AddWith.type.eb8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_29.2 (constants.%Self)]
// CHECK:STDOUT:     %AddWith.F.decl: @AddWith.%AddWith.F.type (%AddWith.F.type) = fn_decl @AddWith.F [symbolic = @AddWith.%AddWith.F (constants.%AddWith.F)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type) = assoc_entity element0, %AddWith.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_29.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%AddWith.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.F(@AddWith.%T.loc4_19.2: type, @AddWith.%Self.loc4_29.1: @AddWith.%AddWith.type (%AddWith.type.eb8)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.F(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AddWith.type.d53: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.d53 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.eb8: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.690: %AddWith.type.eb8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.130: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.ad5: %AddWith.assoc_type.130 = assoc_entity element0, imports.%Main.import_ref.3ef [symbolic]
// CHECK:STDOUT:   %AddWith.F.type.b80: type = fn_type @AddWith.F, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.F.349: %AddWith.F.type.b80 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.type.8b5: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @C.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.bbf: %AddWith.type.8b5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.F.type.b42: type = fn_type @AddWith.F, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %AddWith.F.3d6: %AddWith.F.type.b42 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.217: type = assoc_entity_type @AddWith, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.c17: %AddWith.assoc_type.217 = assoc_entity element0, imports.%Main.import_ref.3ef [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.F.type: type = fn_type @C.as.AddWith.impl.F [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.F: %C.as.AddWith.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.8b5 = facet_value %C, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.AddWith: %AddWith.type.d53 = import_ref Main//a, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Main.import_ref.e4c = import_ref Main//a, loc4_29, unloaded
// CHECK:STDOUT:   %Main.import_ref.f21 = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT:   %Main.F: @AddWith.%AddWith.F.type (%AddWith.F.type.b80) = import_ref Main//a, F, loaded [symbolic = @AddWith.%AddWith.F (constants.%AddWith.F.349)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//a, loc4_19, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3ef = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//a, loc4_19, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.75c: @AddWith.%AddWith.type (%AddWith.type.eb8) = import_ref Main//a, loc4_29, loaded [symbolic = @AddWith.%Self (constants.%Self.690)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AddWith = imports.%Main.AddWith
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.AddWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc7_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.d53 = name_ref AddWith, imports.%Main.AddWith [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %C.ref.loc7_19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%C)> [concrete = constants.%AddWith.type.8b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Main.import_ref.b3bc94.1: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.eb8)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.eb8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.690)]
// CHECK:STDOUT:   %AddWith.F.type: type = fn_type @AddWith.F, @AddWith(%T) [symbolic = %AddWith.F.type (constants.%AddWith.F.type.b80)]
// CHECK:STDOUT:   %AddWith.F: @AddWith.%AddWith.F.type (%AddWith.F.type.b80) = struct_value () [symbolic = %AddWith.F (constants.%AddWith.F.349)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.130)]
// CHECK:STDOUT:   %assoc0: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.130) = assoc_entity element0, imports.%Main.import_ref.3ef [symbolic = %assoc0 (constants.%assoc0.ad5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.e4c
// CHECK:STDOUT:     .F = imports.%Main.import_ref.f21
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.AddWith.impl: %C.ref.loc7_6 as %AddWith.type {
// CHECK:STDOUT:   %C.as.AddWith.impl.F.decl: %C.as.AddWith.impl.F.type = fn_decl @C.as.AddWith.impl.F [concrete = constants.%C.as.AddWith.impl.F] {} {}
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%C.as.AddWith.impl.F.decl), @C.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness %AddWith.impl_witness_table [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.AddWith.impl.F.decl
// CHECK:STDOUT:   witness = %AddWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.75c: @AddWith.%AddWith.type (%AddWith.type.eb8)) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.AddWith.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.F(constants.%T, constants.%Self.690) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8b5
// CHECK:STDOUT:   %Self => constants.%Self.bbf
// CHECK:STDOUT:   %AddWith.F.type => constants.%AddWith.F.type.b42
// CHECK:STDOUT:   %AddWith.F => constants.%AddWith.F.3d6
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.217
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.F(constants.%C, constants.%AddWith.facet) {}
// CHECK:STDOUT:

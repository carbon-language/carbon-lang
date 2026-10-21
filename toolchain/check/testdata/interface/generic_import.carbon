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

interface AddWith(T: type) {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.097: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.097 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.3cd: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %AddWith.type.3cd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.F.type: type = fn_type @AddWith.WithSelf.F, @AddWith.WithSelf(%T, %Self) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.F: %AddWith.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %AddWith.assoc_type = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.F.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.097 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt.loc4_20.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_22.1: type = splice_block %.loc4_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc4_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_20.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc4_20.1)> [symbolic = %AddWith.type (constants.%AddWith.type.3cd)]
// CHECK:STDOUT:   %Self.loc4_28.2: @AddWith.%AddWith.type (%AddWith.type.3cd) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_28.1: @AddWith.%AddWith.type (%AddWith.type.3cd) = symbolic_binding Self, 1 [symbolic = %Self.loc4_28.2 (constants.%Self)]
// CHECK:STDOUT:     %AddWith.WithSelf.decl = interface_with_self_decl @AddWith [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %AddWith.WithSelf.F.decl: @AddWith.WithSelf.%AddWith.WithSelf.F.type (%AddWith.WithSelf.F.type) = fn_decl @AddWith.WithSelf.F [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.F (constants.%AddWith.WithSelf.F)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type) = assoc_entity element0, %AddWith.WithSelf.F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_28.1
// CHECK:STDOUT:     .F = @AddWith.WithSelf.%assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (@AddWith.WithSelf.%AddWith.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.WithSelf.F(@AddWith.%T.loc4_20.2: type, @AddWith.%Self.loc4_28.1: @AddWith.%AddWith.type (%AddWith.type.3cd)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.F(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AddWith.type.097: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.097 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.3cd: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.cca: %AddWith.type.3cd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.F.type.206: type = fn_type @AddWith.WithSelf.F, @AddWith.WithSelf(%T, %Self.cca) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.F.b6b: %AddWith.WithSelf.F.type.206 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.type.5e1: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %.9ca: <witness> = impl_self_witness %C, @AddWith, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @C.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.656: %AddWith.type.5e1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.F.type.906: type = fn_type @AddWith.WithSelf.F, @AddWith.WithSelf(%C, %Self.cca) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.F.411: %AddWith.WithSelf.F.type.906 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.df6: type = assoc_entity_type @AddWith, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.cb9: %AddWith.assoc_type.df6 = assoc_entity element0, imports.%Main.import_ref.26a [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.F.type: type = fn_type @C.as.AddWith.impl.F [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.F: %C.as.AddWith.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.5e1 = facet_value %C, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.F.type.302: type = fn_type @AddWith.WithSelf.F, @AddWith.WithSelf(%C, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.F.956: %AddWith.WithSelf.F.type.302 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.AddWith: %AddWith.type.097 = import_ref Main//a, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Main.import_ref.693 = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT:   %Main.F: @AddWith.WithSelf.%AddWith.WithSelf.F.type (%AddWith.WithSelf.F.type.206) = import_ref Main//a, F, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.F (constants.%AddWith.WithSelf.F.b6b)]
// CHECK:STDOUT:   %Main.import_ref.26a = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//a, loc4_20, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.d09fa4.2: @AddWith.%AddWith.type (%AddWith.type.3cd) = import_ref Main//a, loc4_28, loaded [symbolic = @AddWith.%Self (constants.%Self.cca)]
// CHECK:STDOUT:   %Main.import_ref.5e9 = import_ref Main//a, loc4_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//a, loc4_20, loaded [symbolic = @AddWith.%T (constants.%T)]
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
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.097 = name_ref AddWith, imports.%Main.AddWith [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %C.ref.loc7_19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%C)> [concrete = constants.%AddWith.type.5e1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.AddWith.impl.%C.ref.loc7_6, @AddWith, @AddWith(constants.%C) [concrete = constants.%.9ca]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Main.import_ref.b3bc94.3: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.3cd)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.3cd) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.cca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.5e9
// CHECK:STDOUT:     .F = imports.%Main.import_ref.693
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
// CHECK:STDOUT:   extend %AddWith.type
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
// CHECK:STDOUT: generic fn @AddWith.WithSelf.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.d09fa4.2: @AddWith.%AddWith.type (%AddWith.type.3cd)) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.AddWith.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.cca) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.F(constants.%T, constants.%Self.cca) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%C) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.5e1
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%C, constants.%Self.cca) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.5e1
// CHECK:STDOUT:   %Self => constants.%Self.cca
// CHECK:STDOUT:   %AddWith.WithSelf.F.type => constants.%AddWith.WithSelf.F.type.906
// CHECK:STDOUT:   %AddWith.WithSelf.F => constants.%AddWith.WithSelf.F.411
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.df6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%C, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.5e1
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.F.type => constants.%AddWith.WithSelf.F.type.302
// CHECK:STDOUT:   %AddWith.WithSelf.F => constants.%AddWith.WithSelf.F.956
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.df6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.F(constants.%C, constants.%AddWith.facet) {}
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_import.carbon

// --- a.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  require impls Z;
}

constraint X {
  extend require impls Z;
}

// --- a.impl.carbon
//@include-in-dumps
impl library "[[@TEST_NAME]]";

fn F(A:! X, B:! Y) {}

// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self.f45: %X.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.f60: type = symbolic_binding_type Self, 0, %Self.f45 [symbolic]
// CHECK:STDOUT:   %A: %X.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type %X.type [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.d4d: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.63e: type = symbolic_binding_type Self, 0, %Self.d4d [symbolic]
// CHECK:STDOUT:   %B: %Y.type = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %Y.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Z = import_ref Main//a, Z, unloaded
// CHECK:STDOUT:   %Main.Y: type = import_ref Main//a, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//a, X, loaded [concrete = constants.%X.type]
// CHECK:STDOUT:   %Main.import_ref.69e = import_ref Main//a, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.67f: type = import_ref Main//a, loc10_18, loaded [symbolic = @X.Self.binding.as_type.impls.Z.type.require.%Self.binding.as_type (constants.%Self.binding.as_type.f60)]
// CHECK:STDOUT:   %Main.import_ref.926bc7.1: type = import_ref Main//a, loc10_24, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Main.import_ref.e33: %X.type = import_ref Main//a, loc9_14, loaded [symbolic = constants.%Self.f45]
// CHECK:STDOUT:   %Main.import_ref.65b = import_ref Main//a, loc9_14, unloaded
// CHECK:STDOUT:   %Main.import_ref.1c5 = import_ref Main//a, loc10_25, unloaded
// CHECK:STDOUT:   %Main.import_ref.7bb: type = import_ref Main//a, loc6_11, loaded [symbolic = @Y.Self.binding.as_type.impls.Z.type.require.%Self.binding.as_type (constants.%Self.binding.as_type.63e)]
// CHECK:STDOUT:   %Main.import_ref.926bc7.2: type = import_ref Main//a, loc6_17, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Main.import_ref.8c7: %Y.type = import_ref Main//a, loc5_13, loaded [symbolic = constants.%Self.d4d]
// CHECK:STDOUT:   %Main.import_ref.3e2 = import_ref Main//a, loc5_13, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = imports.%Main.Z
// CHECK:STDOUT:     .Y = imports.%Main.Y
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_17.2 = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.9a5 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.95b = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_10: type = splice_block %X.ref [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc4_6.2: %X.type = symbolic_binding A, 0 [symbolic = %A.loc4_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc4_17: type = splice_block %Y.ref [concrete = constants.%Y.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, imports.%Main.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc4_13.2: %Y.type = symbolic_binding B, 1 [symbolic = %B.loc4_13.1 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.69e
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.3e2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Y.Self.binding.as_type.impls.Z.type.require {
// CHECK:STDOUT:     require imports.%Main.import_ref.7bb impls imports.%Main.import_ref.926bc7.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @X [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.65b
// CHECK:STDOUT:   extend imports.%Main.import_ref.1c5
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @X.Self.binding.as_type.impls.Z.type.require {
// CHECK:STDOUT:     require imports.%Main.import_ref.67f impls imports.%Main.import_ref.926bc7.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @X.Self.binding.as_type.impls.Z.type.require(imports.%Main.import_ref.e33: %X.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f45)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.f60)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Y.Self.binding.as_type.impls.Z.type.require(imports.%Main.import_ref.8c7: %Y.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d4d)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.63e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc4_6.2: %X.type, %B.loc4_13.2: %Y.type) {
// CHECK:STDOUT:   %A.loc4_6.1: %X.type = symbolic_binding A, 0 [symbolic = %A.loc4_6.1 (constants.%A)]
// CHECK:STDOUT:   %B.loc4_13.1: %Y.type = symbolic_binding B, 1 [symbolic = %B.loc4_13.1 (constants.%B)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.Self.binding.as_type.impls.Z.type.require(constants.%Self.f45) {
// CHECK:STDOUT:   %Self => constants.%Self.f45
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.f60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.Self.binding.as_type.impls.Z.type.require(constants.%Self.d4d) {
// CHECK:STDOUT:   %Self => constants.%Self.d4d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.63e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc4_6.1 => constants.%A
// CHECK:STDOUT:   %B.loc4_13.1 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:

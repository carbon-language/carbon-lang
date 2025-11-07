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
// CHECK:STDOUT:   %Self.861: %type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.1a1: type = symbolic_binding_type Self, 0, %Self.861 [symbolic]
// CHECK:STDOUT:   %A: %X.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e25: type = pattern_type %X.type [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.29b: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.384: type = symbolic_binding_type Self, 0, %Self.29b [symbolic]
// CHECK:STDOUT:   %B: %Y.type = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.6e5: type = pattern_type %Y.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Z = import_ref Main//a, Z, unloaded
// CHECK:STDOUT:   %Main.Y: type = import_ref Main//a, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//a, X, loaded [concrete = constants.%X.type]
// CHECK:STDOUT:   %Main.import_ref.462 = import_ref Main//a, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.b7c: type = import_ref Main//a, loc10_18, loaded [symbolic = constants.%Self.binding.as_type.1a1]
// CHECK:STDOUT:   %Main.import_ref.dec: %type = import_ref Main//a, loc9_14, loaded [symbolic = constants.%Self.861]
// CHECK:STDOUT:   %Main.import_ref.cae = import_ref Main//a, loc9_14, unloaded
// CHECK:STDOUT:   %Main.import_ref.3e6: type = import_ref Main//a, loc6_11, loaded [symbolic = constants.%Self.binding.as_type.384]
// CHECK:STDOUT:   %Main.import_ref.35b: %Y.type = import_ref Main//a, loc5_13, loaded [symbolic = constants.%Self.29b]
// CHECK:STDOUT:   %Main.import_ref.581 = import_ref Main//a, loc5_13, unloaded
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
// CHECK:STDOUT:     %A.patt: %pattern_type.e25 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.6e5 = symbolic_binding_pattern B, 1 [concrete]
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.462
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.581
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Y.require1 {
// CHECK:STDOUT:     require imports.%Main.import_ref.3e6 impls <@Z>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @X [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cae
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @X.require0 {
// CHECK:STDOUT:     require imports.%Main.import_ref.b7c impls <@Z>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @X.require0(imports.%Main.import_ref.dec: %type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.861)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1a1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Y.require1(imports.%Main.import_ref.35b: %Y.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.29b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.384)]
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
// CHECK:STDOUT: specific @X.require0(constants.%Self.861) {
// CHECK:STDOUT:   %Self => constants.%Self.861
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.require1(constants.%Self.29b) {
// CHECK:STDOUT:   %Self => constants.%Self.29b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.384
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc4_6.1 => constants.%A
// CHECK:STDOUT:   %B.loc4_13.1 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:

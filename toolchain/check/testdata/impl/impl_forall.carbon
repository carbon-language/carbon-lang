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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_forall.carbon

interface Simple {
  fn F();
}

impl forall [T:! type] T as Simple {
  fn F() {}
}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [concrete]
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.type.75b: type = fn_type @Simple.WithSelf.F, @Simple.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.58c: %Simple.WithSelf.F.type.75b = struct_value () [symbolic]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type @Simple [concrete]
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, @Simple.WithSelf.%Simple.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness @T.as.Simple.impl.%Simple.impl_witness_table, @T.as.Simple.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.Simple.impl.F.type: type = fn_type @T.as.Simple.impl.F, @T.as.Simple.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.Simple.impl.F: %T.as.Simple.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Simple.facet: %Simple.type = facet_value %T, (%Simple.impl_witness) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.type.b56: type = fn_type @Simple.WithSelf.F, @Simple.WithSelf(%Simple.facet) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.2ed: %Simple.WithSelf.F.type.b56 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [concrete = constants.%Simple.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Simple.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_15.2 [symbolic = %T.loc19_15.1 (constants.%T)]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:     %.loc19_18.1: type = splice_block %.loc19_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc19_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc19_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Simple.WithSelf.decl = interface_with_self_decl @Simple [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Simple.WithSelf.F.decl: @Simple.WithSelf.%Simple.WithSelf.F.type (%Simple.WithSelf.F.type.75b) = fn_decl @Simple.WithSelf.F [symbolic = @Simple.WithSelf.%Simple.WithSelf.F (constants.%Simple.WithSelf.F.58c)] {} {}
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %Simple.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Simple.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Simple.WithSelf.%Simple.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Simple.impl(%T.loc19_15.2: type) {
// CHECK:STDOUT:   %T.loc19_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc19_15.1 (constants.%T)]
// CHECK:STDOUT:   %Simple.impl_witness.loc19_36.2: <witness> = impl_witness %Simple.impl_witness_table, @T.as.Simple.impl(%T.loc19_15.1) [symbolic = %Simple.impl_witness.loc19_36.2 (constants.%Simple.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.Simple.impl.F.type: type = fn_type @T.as.Simple.impl.F, @T.as.Simple.impl(%T.loc19_15.1) [symbolic = %T.as.Simple.impl.F.type (constants.%T.as.Simple.impl.F.type)]
// CHECK:STDOUT:   %T.as.Simple.impl.F: @T.as.Simple.impl.%T.as.Simple.impl.F.type (%T.as.Simple.impl.F.type) = struct_value () [symbolic = %T.as.Simple.impl.F (constants.%T.as.Simple.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Simple.ref {
// CHECK:STDOUT:     %T.as.Simple.impl.F.decl: @T.as.Simple.impl.%T.as.Simple.impl.F.type (%T.as.Simple.impl.F.type) = fn_decl @T.as.Simple.impl.F [symbolic = @T.as.Simple.impl.%T.as.Simple.impl.F (constants.%T.as.Simple.impl.F)] {} {}
// CHECK:STDOUT:     %Simple.impl_witness_table = impl_witness_table (%T.as.Simple.impl.F.decl), @T.as.Simple.impl [concrete]
// CHECK:STDOUT:     %Simple.impl_witness.loc19_36.1: <witness> = impl_witness %Simple.impl_witness_table, @T.as.Simple.impl(constants.%T) [symbolic = %Simple.impl_witness.loc19_36.2 (constants.%Simple.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.Simple.impl.F.decl
// CHECK:STDOUT:     extend %Simple.ref
// CHECK:STDOUT:     witness = %Simple.impl_witness.loc19_36.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple.WithSelf.F(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.Simple.impl.F(@T.as.Simple.impl.%T.loc19_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Simple.WithSelf.F.type => constants.%Simple.WithSelf.F.type.75b
// CHECK:STDOUT:   %Simple.WithSelf.F => constants.%Simple.WithSelf.F.58c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Simple.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc19_15.1 => constants.%T
// CHECK:STDOUT:   %Simple.impl_witness.loc19_36.2 => constants.%Simple.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.Simple.impl.F.type => constants.%T.as.Simple.impl.F.type
// CHECK:STDOUT:   %T.as.Simple.impl.F => constants.%T.as.Simple.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Simple.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf(constants.%Simple.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Simple.facet
// CHECK:STDOUT:   %Simple.WithSelf.F.type => constants.%Simple.WithSelf.F.type.b56
// CHECK:STDOUT:   %Simple.WithSelf.F => constants.%Simple.WithSelf.F.2ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.F(constants.%Simple.facet) {}
// CHECK:STDOUT:

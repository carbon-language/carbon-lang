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

// --- impl_forall.carbon
library "[[@TEST_NAME]]";

interface Simple {
  fn F();
}

impl forall [T: type] T as Simple {
  fn F() {}
}

// --- self_error.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}

// The self type `T` is an `ErrorInst`, and when identifying the facet type
// would get substituted into the `.Self`.
// CHECK:STDERR: self_error.carbon:[[@LINE+2]]:17-23: error: name `Invalid` not found [NameNotFound]
// CHECK:STDERR:
impl forall [T: Invalid] T as Z where .Z1 impls Y {}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [concrete]
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.type.a46: type = fn_type @Simple.WithSelf.F, @Simple.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.0d0: %Simple.WithSelf.F.type.a46 = struct_value () [symbolic]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type @Simple [concrete]
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, @Simple.WithSelf.%Simple.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %.2a3: <witness> = impl_self_witness %T, @Simple [symbolic]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness @T.as.Simple.impl.%Simple.impl_witness_table, @T.as.Simple.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.Simple.impl.F.type: type = fn_type @T.as.Simple.impl.F, @T.as.Simple.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.Simple.impl.F: %T.as.Simple.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Simple.facet: %Simple.type = facet_value %T, (%Simple.impl_witness) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.type.9de: type = fn_type @Simple.WithSelf.F, @Simple.WithSelf(%Simple.facet) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.25d: %Simple.WithSelf.F.type.9de = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [concrete = constants.%Simple.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Simple.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.2 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:     %.loc7_17.1: type = splice_block %.loc7_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @T.as.Simple.impl.%T.ref, @Simple [symbolic = @T.as.Simple.impl.%.loc7_35 (constants.%.2a3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Simple.WithSelf.decl = interface_with_self_decl @Simple [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Simple.WithSelf.F.decl: @Simple.WithSelf.%Simple.WithSelf.F.type (%Simple.WithSelf.F.type.a46) = fn_decl @Simple.WithSelf.F [symbolic = @Simple.WithSelf.%Simple.WithSelf.F (constants.%Simple.WithSelf.F.0d0)] {} {}
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
// CHECK:STDOUT: generic impl @T.as.Simple.impl(%T.loc7_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:   %.loc7_35: <witness> = impl_self_witness %T.loc7_15.1, @Simple [symbolic = %.loc7_35 (constants.%.2a3)]
// CHECK:STDOUT:   %Simple.impl_witness.loc7_35.2: <witness> = impl_witness %Simple.impl_witness_table, @T.as.Simple.impl(%T.loc7_15.1) [symbolic = %Simple.impl_witness.loc7_35.2 (constants.%Simple.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.Simple.impl.F.type: type = fn_type @T.as.Simple.impl.F, @T.as.Simple.impl(%T.loc7_15.1) [symbolic = %T.as.Simple.impl.F.type (constants.%T.as.Simple.impl.F.type)]
// CHECK:STDOUT:   %T.as.Simple.impl.F: @T.as.Simple.impl.%T.as.Simple.impl.F.type (%T.as.Simple.impl.F.type) = struct_value () [symbolic = %T.as.Simple.impl.F (constants.%T.as.Simple.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Simple.ref {
// CHECK:STDOUT:     %T.as.Simple.impl.F.decl: @T.as.Simple.impl.%T.as.Simple.impl.F.type (%T.as.Simple.impl.F.type) = fn_decl @T.as.Simple.impl.F [symbolic = @T.as.Simple.impl.%T.as.Simple.impl.F (constants.%T.as.Simple.impl.F)] {} {}
// CHECK:STDOUT:     %Simple.impl_witness_table = impl_witness_table (%T.as.Simple.impl.F.decl), @T.as.Simple.impl [concrete]
// CHECK:STDOUT:     %Simple.impl_witness.loc7_35.1: <witness> = impl_witness %Simple.impl_witness_table, @T.as.Simple.impl(constants.%T) [symbolic = %Simple.impl_witness.loc7_35.2 (constants.%Simple.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.Simple.impl.F.decl
// CHECK:STDOUT:     extend %Simple.ref
// CHECK:STDOUT:     witness = %Simple.impl_witness.loc7_35.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple.WithSelf.F(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.Simple.impl.F(@T.as.Simple.impl.%T.loc7_15.2: type) {
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
// CHECK:STDOUT:   %Simple.WithSelf.F.type => constants.%Simple.WithSelf.F.type.a46
// CHECK:STDOUT:   %Simple.WithSelf.F => constants.%Simple.WithSelf.F.0d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Simple.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_15.1 => constants.%T
// CHECK:STDOUT:   %.loc7_35 => constants.%.2a3
// CHECK:STDOUT:   %Simple.impl_witness.loc7_35.2 => constants.%Simple.impl_witness
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
// CHECK:STDOUT:   %Simple.WithSelf.F.type => constants.%Simple.WithSelf.F.type.9de
// CHECK:STDOUT:   %Simple.WithSelf.F => constants.%Simple.WithSelf.F.25d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.F(constants.%Simple.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.cb6: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z1 [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.765: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.7b7: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.7b7 [symbolic_self]
// CHECK:STDOUT:   %.db1: <witness> = impl_self_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e1c: type = impl_witness_access %.db1, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.513: <witness> = impl_self_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ff0: type = impl_witness_access %.513, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z where %impl.elem0.ff0 impls @Y> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .Invalid = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   impl_decl @<error>.as.<error>.impl [concrete] {
// CHECK:STDOUT:     %T.patt: <error> = symbolic_binding_pattern T, 0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, %T [concrete = <error>]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.frozen.loc10_33: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self.frozen.loc10_33 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc10_39: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_39.1: type = impl_witness_access constants.%.db1, element0 [symbolic_self = constants.%impl.elem0.e1c]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %impls.loc10_43.1 = requirement_impls %impl.elem0.loc10_39.1, %Y.ref [concrete]
// CHECK:STDOUT:     %impl.elem0.loc10_39.2: type = impl_witness_access constants.%.513, element0 [symbolic_self = constants.%impl.elem0.ff0]
// CHECK:STDOUT:     %impls.loc10_43.2 = requirement_impls %impl.elem0.loc10_39.2, %Y.ref [concrete]
// CHECK:STDOUT:     %.loc10_33: type = where_expr [concrete = constants.%Z_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %impls.loc10_43.2 = requirement_impls %impl.elem0.loc10_39.2, %Y.ref [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %.Self.frozen.loc10_15: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Invalid.ref: <error> = name_ref Invalid, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = symbolic_binding T, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cb6]
// CHECK:STDOUT:   %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z1: type = assoc_const_decl @Z1 [concrete] {
// CHECK:STDOUT:     %assoc0: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z1 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z1 = @Z1.%assoc0
// CHECK:STDOUT:   witness = (@Z.WithSelf.%Z1)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.765]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @<error>.as.<error>.impl(%T: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: <error> as <error> {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend <error>
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.765) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%.Self.frozen.7b7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.<error>.impl(<error>) {}
// CHECK:STDOUT:

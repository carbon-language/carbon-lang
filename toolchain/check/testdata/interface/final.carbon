// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/final.carbon

// --- require_complete_impl.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface I { fn F(); }
interface J {}

final impl forall [T:! J] T as I { fn F() {} }

// This is creating an indirection where the `final impl` requires an indirect
// instantiation by the caller.
fn UseI(T:! I) { T.F(); }
fn UseJ(T:! J) { UseI(T); }

class C {}
impl C as J {}
fn Use() { UseJ(C); }

// CHECK:STDOUT: --- require_complete_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.179: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.d9f: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.179) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.592: %I.WithSelf.F.type.d9f = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.fd8: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.1b4: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.1f3: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.771: type = facet_access_type %T.1f3 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.f6a: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.1f3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.1ee: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.1f3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.da0: %T.as_type.as.I.impl.F.type.1ee = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.e7f: %I.type = facet_value %T.as_type.771, (%I.impl_witness.f6a) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.9d4: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.e7f) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.4d1: %I.WithSelf.F.type.9d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.998: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.2e1: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %UseI.type: type = fn_type @UseI [concrete]
// CHECK:STDOUT:   %UseI: %UseI.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.a52: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.2e1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.08d: %I.WithSelf.F.type.a52 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.a70: type = facet_access_type %T.2e1 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.2e1, @I [symbolic]
// CHECK:STDOUT:   %.b7f: type = fn_type_with_self_type %I.WithSelf.F.type.a52, %T.2e1 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b7f = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%T.2e1) [symbolic]
// CHECK:STDOUT:   %UseJ.type: type = fn_type @UseJ [concrete]
// CHECK:STDOUT:   %UseJ: %UseJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %.af5: require_specific_def_type = require_specific_def @T.as_type.as.I.impl(%T.1f3) [symbolic]
// CHECK:STDOUT:   %UseI.specific_fn.722: <specific function> = specific_function %UseI, @UseI(%I.facet.e7f) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ, @UseJ(%J.facet) [concrete]
// CHECK:STDOUT:   %.9f6: type = fn_type_with_self_type %I.WithSelf.F.type.9d4, %I.facet.e7f [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.04c: <specific function> = specific_function %T.as_type.as.I.impl.F.da0, @T.as_type.as.I.impl.F(%T.1f3) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e9c: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.2ec: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.5b3: %T.as_type.as.I.impl.F.type.2ec = struct_value () [concrete]
// CHECK:STDOUT:   %.86c: require_specific_def_type = require_specific_def @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %I.facet.9c3: %I.type = facet_value %C, (%I.impl_witness.e9c) [concrete]
// CHECK:STDOUT:   %UseI.specific_fn.6ad: <specific function> = specific_function %UseI, @UseI(%I.facet.9c3) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.425: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.9c3) [concrete]
// CHECK:STDOUT:   %.59d: type = fn_type_with_self_type %I.WithSelf.F.type.425, %I.facet.9c3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.675: <specific function> = specific_function %T.as_type.as.I.impl.F.5b3, @T.as_type.as.I.impl.F(%J.facet) [concrete]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .UseI = %UseI.decl
// CHECK:STDOUT:     .UseJ = %UseJ.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.1b4 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc7_21.2 [symbolic = %T.loc7_21.1 (constants.%T.1f3)]
// CHECK:STDOUT:     %T.as_type.loc7_27.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_27.1 (constants.%T.as_type.771)]
// CHECK:STDOUT:     %.loc7_27: type = converted %T.ref, %T.as_type.loc7_27.2 [symbolic = %T.as_type.loc7_27.1 (constants.%T.as_type.771)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_24: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_21.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.1f3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseI.decl: %UseI.type = fn_decl @UseI [concrete = constants.%UseI] {
// CHECK:STDOUT:     %T.patt: %pattern_type.998 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_13: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T.2e1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseJ.decl: %UseJ.type = fn_decl @UseJ [concrete = constants.%UseJ] {
// CHECK:STDOUT:     %T.patt: %pattern_type.1b4 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_13: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_10.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_10.1 (constants.%T.1f3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.179]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.d9f) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.592)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.fd8]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.I.impl(%T.loc7_21.2: %J.type) {
// CHECK:STDOUT:   %T.loc7_21.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.1f3)]
// CHECK:STDOUT:   %T.as_type.loc7_27.1: type = facet_access_type %T.loc7_21.1 [symbolic = %T.as_type.loc7_27.1 (constants.%T.as_type.771)]
// CHECK:STDOUT:   %I.impl_witness.loc7_34.2: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %I.impl_witness.loc7_34.2 (constants.%I.impl_witness.f6a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %T.as_type.as.I.impl.F.type (constants.%T.as_type.as.I.impl.F.type.1ee)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.1ee) = struct_value () [symbolic = %T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.da0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7_27 as %I.ref {
// CHECK:STDOUT:     %T.as_type.as.I.impl.F.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.1ee) = fn_decl @T.as_type.as.I.impl.F [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.da0)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as_type.as.I.impl.F.decl), @T.as_type.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_34.1: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(constants.%T.1f3) [symbolic = %I.impl_witness.loc7_34.2 (constants.%I.impl_witness.f6a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as_type.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.F(@T.as_type.as.I.impl.%T.loc7_21.2: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseI(%T.loc11_10.2: %I.type) {
// CHECK:STDOUT:   %T.loc11_10.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T.2e1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_19.2: type = facet_access_type %T.loc11_10.1 [symbolic = %T.as_type.loc11_19.2 (constants.%T.as_type.a70)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc11_10.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.a52)]
// CHECK:STDOUT:   %.loc11_19.2: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc11_10.1 [symbolic = %.loc11_19.2 (constants.%.b7f)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_10.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_19.2: @UseI.%.loc11_19.2 (%.b7f) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_19.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_19.2: <specific function> = specific_impl_function %impl.elem0.loc11_19.2, @I.WithSelf.F(%T.loc11_10.1) [symbolic = %specific_impl_fn.loc11_19.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc11_10.2 [symbolic = %T.loc11_10.1 (constants.%T.2e1)]
// CHECK:STDOUT:     %T.as_type.loc11_19.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_19.2 (constants.%T.as_type.a70)]
// CHECK:STDOUT:     %.loc11_19.1: type = converted %T.ref, %T.as_type.loc11_19.1 [symbolic = %T.as_type.loc11_19.2 (constants.%T.as_type.a70)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11_19.1: @UseI.%.loc11_19.2 (%.b7f) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_19.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_19.1: <specific function> = specific_impl_function %impl.elem0.loc11_19.1, @I.WithSelf.F(constants.%T.2e1) [symbolic = %specific_impl_fn.loc11_19.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc11_19.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseJ(%T.loc12_10.2: %J.type) {
// CHECK:STDOUT:   %T.loc12_10.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_10.1 (constants.%T.1f3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_24.2: type = facet_access_type %T.loc12_10.1 [symbolic = %T.as_type.loc12_24.2 (constants.%T.as_type.771)]
// CHECK:STDOUT:   %.loc12_24.2: require_specific_def_type = require_specific_def @T.as_type.as.I.impl(%T.loc12_10.1) [symbolic = %.loc12_24.2 (constants.%.af5)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.loc12_10.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.f6a)]
// CHECK:STDOUT:   %I.facet.loc12_24.2: %I.type = facet_value %T.as_type.loc12_24.2, (%I.impl_witness) [symbolic = %I.facet.loc12_24.2 (constants.%I.facet.e7f)]
// CHECK:STDOUT:   %UseI.specific_fn.loc12_18.2: <specific function> = specific_function constants.%UseI, @UseI(%I.facet.loc12_24.2) [symbolic = %UseI.specific_fn.loc12_18.2 (constants.%UseI.specific_fn.722)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %UseI.ref: %UseI.type = name_ref UseI, file.%UseI.decl [concrete = constants.%UseI]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc12_10.2 [symbolic = %T.loc12_10.1 (constants.%T.1f3)]
// CHECK:STDOUT:     %T.as_type.loc12_24.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_24.2 (constants.%T.as_type.771)]
// CHECK:STDOUT:     %I.facet.loc12_24.1: %I.type = facet_value %T.as_type.loc12_24.1, (constants.%I.impl_witness.f6a) [symbolic = %I.facet.loc12_24.2 (constants.%I.facet.e7f)]
// CHECK:STDOUT:     %.loc12_24.1: %I.type = converted %T.ref, %I.facet.loc12_24.1 [symbolic = %I.facet.loc12_24.2 (constants.%I.facet.e7f)]
// CHECK:STDOUT:     %UseI.specific_fn.loc12_18.1: <specific function> = specific_function %UseI.ref, @UseI(constants.%I.facet.e7f) [symbolic = %UseI.specific_fn.loc12_18.2 (constants.%UseI.specific_fn.722)]
// CHECK:STDOUT:     %UseI.call: init %empty_tuple.type = call %UseI.specific_fn.loc12_18.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %UseJ.ref: %UseJ.type = name_ref UseJ, file.%UseJ.decl [concrete = constants.%UseJ]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc16: %J.type = converted %C.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ.ref, @UseJ(constants.%J.facet) [concrete = constants.%UseJ.specific_fn]
// CHECK:STDOUT:   %UseJ.call: init %empty_tuple.type = call %UseJ.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.179) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.179
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.d9f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.592
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.179) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.fd8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%T.1f3) {
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%T.1f3
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%T.as_type.771
// CHECK:STDOUT:   %I.impl_witness.loc7_34.2 => constants.%I.impl_witness.f6a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.1ee
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.da0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%T.1f3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.e7f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.e7f
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.9d4
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.4d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.e7f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%T.2e1) {
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%T.2e1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.2e1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T.2e1
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a52
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.08d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T.2e1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%T.1f3) {
// CHECK:STDOUT:   %T.loc12_10.1 => constants.%T.1f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.e7f) {
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%I.facet.e7f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_19.2 => constants.%T.as_type.771
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.9d4
// CHECK:STDOUT:   %.loc11_19.2 => constants.%.9f6
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.f6a
// CHECK:STDOUT:   %impl.elem0.loc11_19.2 => constants.%T.as_type.as.I.impl.F.da0
// CHECK:STDOUT:   %specific_impl_fn.loc11_19.2 => constants.%T.as_type.as.I.impl.F.specific_fn.04c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%J.facet) {
// CHECK:STDOUT:   %T.loc12_10.1 => constants.%J.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_24.2 => constants.%C
// CHECK:STDOUT:   %.loc12_24.2 => constants.%.86c
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.e9c
// CHECK:STDOUT:   %I.facet.loc12_24.2 => constants.%I.facet.9c3
// CHECK:STDOUT:   %UseI.specific_fn.loc12_18.2 => constants.%UseI.specific_fn.6ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet) {
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%J.facet
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%C
// CHECK:STDOUT:   %I.impl_witness.loc7_34.2 => constants.%I.impl_witness.e9c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.2ec
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.5b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.9c3) {
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%I.facet.9c3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_19.2 => constants.%C
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.425
// CHECK:STDOUT:   %.loc11_19.2 => constants.%.59d
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.e9c
// CHECK:STDOUT:   %impl.elem0.loc11_19.2 => constants.%T.as_type.as.I.impl.F.5b3
// CHECK:STDOUT:   %specific_impl_fn.loc11_19.2 => constants.%T.as_type.as.I.impl.F.specific_fn.675
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.9c3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.9c3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

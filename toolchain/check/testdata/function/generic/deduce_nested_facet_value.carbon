// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/deduce_nested_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/deduce_nested_facet_value.carbon

// --- deduce_nested_facet_value.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface W {}

// DD implements both Y and W.
class DD {}
impl DD as Y {}
impl DD as W {}

// CC requires D to implement Y.
class CC(D:! Y) {}

interface Z {}

// The `D` interface provides `Y` but not `W`, so we need to see that the
// parameter to `CC` is `DD` which provides `Y & W`, not just a FacetValue
// abstractly providing `Y.
impl forall [E:! Y & W] CC(E) as Z {}

fn F() {
  (CC(DD)) as Z;
}

// CHECK:STDOUT: --- deduce_nested_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.765: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %W.type: type = facet_type <@W> [concrete]
// CHECK:STDOUT:   %Self.dff: %W.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DD: type = class_type @DD [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @DD.as.Y.impl.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet.2a4: %Y.type = facet_value %DD, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %W.impl_witness: <witness> = impl_witness @DD.as.W.impl.%W.impl_witness_table [concrete]
// CHECK:STDOUT:   %W.facet: %W.type = facet_value %DD, (%W.impl_witness) [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.cfb: type = pattern_type %Y.type [concrete]
// CHECK:STDOUT:   %D: %Y.type = symbolic_binding D, 0 [symbolic]
// CHECK:STDOUT:   %CC.type: type = generic_class_type @CC [concrete]
// CHECK:STDOUT:   %CC.generic: %CC.type = struct_value () [concrete]
// CHECK:STDOUT:   %CC.b97: type = class_type @CC, @CC(%D) [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.cb6: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Y.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Y & @W> [concrete]
// CHECK:STDOUT:   %pattern_type.bd4: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %E: %facet_type = symbolic_binding E, 0 [symbolic]
// CHECK:STDOUT:   %E.as_type: type = facet_access_type %E [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %E, @Y [symbolic]
// CHECK:STDOUT:   %Y.facet.18f: %Y.type = facet_value %E.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %CC.db5: type = class_type @CC, @CC(%Y.facet.18f) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.6a9: <witness> = impl_witness @CC.as.Z.impl.%Z.impl_witness_table, @CC.as.Z.impl(%E) [symbolic]
// CHECK:STDOUT:   %Z.facet.674: %Z.type = facet_value %CC.db5, (%Z.impl_witness.6a9) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CC.d62: type = class_type @CC, @CC(%Y.facet.2a4) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %DD, (%Y.impl_witness, %W.impl_witness) [concrete]
// CHECK:STDOUT:   %Z.impl_witness.b3a: <witness> = impl_witness @CC.as.Z.impl.%Z.impl_witness_table, @CC.as.Z.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %Z.facet.737: %Z.type = facet_value %CC.d62, (%Z.impl_witness.b3a) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .W = %W.decl
// CHECK:STDOUT:     .DD = %DD.decl
// CHECK:STDOUT:     .CC = %CC.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %W.decl: type = interface_decl @W [concrete = constants.%W.type] {} {}
// CHECK:STDOUT:   %DD.decl: type = class_decl @DD [concrete = constants.%DD] {} {}
// CHECK:STDOUT:   impl_decl @DD.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, file.%DD.decl [concrete = constants.%DD]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @DD.as.W.impl [concrete] {} {
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, file.%DD.decl [concrete = constants.%DD]
// CHECK:STDOUT:     %W.ref: type = name_ref W, file.%W.decl [concrete = constants.%W.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CC.decl: %CC.type = class_decl @CC [concrete = constants.%CC.generic] {
// CHECK:STDOUT:     %D.patt: %pattern_type.cfb = symbolic_binding_pattern D, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: type = splice_block %Y.ref [concrete = constants.%Y.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.loc12_11.2: %Y.type = symbolic_binding D, 0 [symbolic = %D.loc12_11.1 (constants.%D)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @CC.as.Z.impl [concrete] {
// CHECK:STDOUT:     %E.patt: %pattern_type.bd4 = symbolic_binding_pattern E, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CC.ref: %CC.type = name_ref CC, file.%CC.decl [concrete = constants.%CC.generic]
// CHECK:STDOUT:     %E.ref: %facet_type = name_ref E, %E.loc19_15.1 [symbolic = %E.loc19_15.2 (constants.%E)]
// CHECK:STDOUT:     %E.as_type.loc19_29.1: type = facet_access_type %E.ref [symbolic = %E.as_type.loc19_29.2 (constants.%E.as_type)]
// CHECK:STDOUT:     %Y.facet.loc19_29.1: %Y.type = facet_value %E.as_type.loc19_29.1, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc19_29.2 (constants.%Y.facet.18f)]
// CHECK:STDOUT:     %.loc19_29: %Y.type = converted %E.ref, %Y.facet.loc19_29.1 [symbolic = %Y.facet.loc19_29.2 (constants.%Y.facet.18f)]
// CHECK:STDOUT:     %CC.loc19_29.1: type = class_type @CC, @CC(constants.%Y.facet.18f) [symbolic = %CC.loc19_29.2 (constants.%CC.db5)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.loc19_20.1: type = splice_block %.loc19_20.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:       %W.ref: type = name_ref W, file.%W.decl [concrete = constants.%W.type]
// CHECK:STDOUT:       %impl.elem0: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %Y.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%Y.ref, %W.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc19_20.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc19_20.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc19_20.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %E.loc19_15.1: %facet_type = symbolic_binding E, 0 [symbolic = %E.loc19_15.2 (constants.%E)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
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
// CHECK:STDOUT: interface @W {
// CHECK:STDOUT:   %Self: %W.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dff]
// CHECK:STDOUT:   %W.WithSelf.decl = interface_with_self_decl @W [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cb6]
// CHECK:STDOUT:   %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @DD.as.Y.impl: %DD.ref as %Y.ref {
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (), @DD.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Y.ref
// CHECK:STDOUT:   witness = %Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @DD.as.W.impl: %DD.ref as %W.ref {
// CHECK:STDOUT:   %W.impl_witness_table = impl_witness_table (), @DD.as.W.impl [concrete]
// CHECK:STDOUT:   %W.impl_witness: <witness> = impl_witness %W.impl_witness_table [concrete = constants.%W.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %W.ref
// CHECK:STDOUT:   witness = %W.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CC.as.Z.impl(%E.loc19_15.1: %facet_type) {
// CHECK:STDOUT:   %E.loc19_15.2: %facet_type = symbolic_binding E, 0 [symbolic = %E.loc19_15.2 (constants.%E)]
// CHECK:STDOUT:   %E.as_type.loc19_29.2: type = facet_access_type %E.loc19_15.2 [symbolic = %E.as_type.loc19_29.2 (constants.%E.as_type)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %E.loc19_15.2, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %Y.facet.loc19_29.2: %Y.type = facet_value %E.as_type.loc19_29.2, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc19_29.2 (constants.%Y.facet.18f)]
// CHECK:STDOUT:   %CC.loc19_29.2: type = class_type @CC, @CC(%Y.facet.loc19_29.2) [symbolic = %CC.loc19_29.2 (constants.%CC.db5)]
// CHECK:STDOUT:   %Z.impl_witness.loc19_36.2: <witness> = impl_witness %Z.impl_witness_table, @CC.as.Z.impl(%E.loc19_15.2) [symbolic = %Z.impl_witness.loc19_36.2 (constants.%Z.impl_witness.6a9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %CC.loc19_29.1 as %Z.ref {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (), @CC.as.Z.impl [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc19_36.1: <witness> = impl_witness %Z.impl_witness_table, @CC.as.Z.impl(constants.%E) [symbolic = %Z.impl_witness.loc19_36.2 (constants.%Z.impl_witness.6a9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.ref
// CHECK:STDOUT:     witness = %Z.impl_witness.loc19_36.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DD {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CC(%D.loc12_11.2: %Y.type) {
// CHECK:STDOUT:   %D.loc12_11.1: %Y.type = symbolic_binding D, 0 [symbolic = %D.loc12_11.1 (constants.%D)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CC.b97
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CC.ref: %CC.type = name_ref CC, file.%CC.decl [concrete = constants.%CC.generic]
// CHECK:STDOUT:   %DD.ref: type = name_ref DD, file.%DD.decl [concrete = constants.%DD]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %DD.ref, (constants.%Y.impl_witness) [concrete = constants.%Y.facet.2a4]
// CHECK:STDOUT:   %.loc22_9: %Y.type = converted %DD.ref, %Y.facet [concrete = constants.%Y.facet.2a4]
// CHECK:STDOUT:   %CC: type = class_type @CC, @CC(constants.%Y.facet.2a4) [concrete = constants.%CC.d62]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %CC, (constants.%Z.impl_witness.b3a) [concrete = constants.%Z.facet.737]
// CHECK:STDOUT:   %.loc22_12: %Z.type = converted %CC, %Z.facet [concrete = constants.%Z.facet.737]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.765) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W.WithSelf(constants.%Self.dff) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Y.facet.2a4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W.WithSelf(constants.%W.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC(constants.%D) {
// CHECK:STDOUT:   %D.loc12_11.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC(constants.%Y.facet.18f) {
// CHECK:STDOUT:   %D.loc12_11.1 => constants.%Y.facet.18f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC.as.Z.impl(constants.%E) {
// CHECK:STDOUT:   %E.loc19_15.2 => constants.%E
// CHECK:STDOUT:   %E.as_type.loc19_29.2 => constants.%E.as_type
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness
// CHECK:STDOUT:   %Y.facet.loc19_29.2 => constants.%Y.facet.18f
// CHECK:STDOUT:   %CC.loc19_29.2 => constants.%CC.db5
// CHECK:STDOUT:   %Z.impl_witness.loc19_36.2 => constants.%Z.impl_witness.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Z.facet.674) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC(constants.%Y.facet.2a4) {
// CHECK:STDOUT:   %D.loc12_11.1 => constants.%Y.facet.2a4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC.as.Z.impl(constants.%facet_value) {
// CHECK:STDOUT:   %E.loc19_15.2 => constants.%facet_value
// CHECK:STDOUT:   %E.as_type.loc19_29.2 => constants.%DD
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.impl_witness
// CHECK:STDOUT:   %Y.facet.loc19_29.2 => constants.%Y.facet.2a4
// CHECK:STDOUT:   %CC.loc19_29.2 => constants.%CC.d62
// CHECK:STDOUT:   %Z.impl_witness.loc19_36.2 => constants.%Z.impl_witness.b3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
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
// CHECK:STDOUT:   %Self.33b: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %W.type: type = facet_type <@W> [concrete]
// CHECK:STDOUT:   %Self.9bf: %W.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %DD: type = class_type @DD [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0ec: <witness> = impl_witness @DD.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.3d1: type = ptr_type %DD [concrete]
// CHECK:STDOUT:   %pattern_type.3fc: type = pattern_type %ptr.3d1 [concrete]
// CHECK:STDOUT:   %DD.as.Destroy.impl.Op.type: type = fn_type @DD.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %DD.as.Destroy.impl.Op: %DD.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %W.impl_witness: <witness> = impl_witness file.%W.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %D: %Y.type = bind_symbolic_name D, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f7e: type = pattern_type %Y.type [concrete]
// CHECK:STDOUT:   %CC.type: type = generic_class_type @CC [concrete]
// CHECK:STDOUT:   %CC.generic: %CC.type = struct_value () [concrete]
// CHECK:STDOUT:   %CC.6c6: type = class_type @CC, @CC(%D) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.195: <witness> = impl_witness @CC.%Destroy.impl_witness_table, @CC.as.Destroy.impl(%D) [symbolic]
// CHECK:STDOUT:   %ptr.f53: type = ptr_type %CC.6c6 [symbolic]
// CHECK:STDOUT:   %pattern_type.6d4: type = pattern_type %ptr.f53 [symbolic]
// CHECK:STDOUT:   %CC.as.Destroy.impl.Op.type: type = fn_type @CC.as.Destroy.impl.Op, @CC.as.Destroy.impl(%D) [symbolic]
// CHECK:STDOUT:   %CC.as.Destroy.impl.Op: %CC.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.9ba: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.b22: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Y.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Y & @W> [concrete]
// CHECK:STDOUT:   %E: %facet_type = bind_symbolic_name E, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.29c: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %E, @Y [symbolic]
// CHECK:STDOUT:   %E.as_type: type = facet_access_type %E [symbolic]
// CHECK:STDOUT:   %Y.facet.ecf: %Y.type = facet_value %E.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %CC.6eb: type = class_type @CC, @CC(%Y.facet.ecf) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.481: <witness> = impl_witness file.%Z.impl_witness_table, @CC.as.Z.impl(%E) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet.b41: %Y.type = facet_value %DD, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %CC.017: type = class_type @CC, @CC(%Y.facet.b41) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %DD, (%Y.impl_witness, %W.impl_witness) [concrete]
// CHECK:STDOUT:   %Z.impl_witness.9cd: <witness> = impl_witness file.%Z.impl_witness_table, @CC.as.Z.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %CC.017, (%Z.impl_witness.9cd) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.a4f: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude, loc14_42, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.a4f), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (), @DD.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   impl_decl @DD.as.W.impl [concrete] {} {
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, file.%DD.decl [concrete = constants.%DD]
// CHECK:STDOUT:     %W.ref: type = name_ref W, file.%W.decl [concrete = constants.%W.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %W.impl_witness_table = impl_witness_table (), @DD.as.W.impl [concrete]
// CHECK:STDOUT:   %W.impl_witness: <witness> = impl_witness %W.impl_witness_table [concrete = constants.%W.impl_witness]
// CHECK:STDOUT:   %CC.decl: %CC.type = class_decl @CC [concrete = constants.%CC.generic] {
// CHECK:STDOUT:     %D.patt: %pattern_type.f7e = symbolic_binding_pattern D, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: type = splice_block %Y.ref [concrete = constants.%Y.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.loc12_10.2: %Y.type = bind_symbolic_name D, 0 [symbolic = %D.loc12_10.1 (constants.%D)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @CC.as.Z.impl [concrete] {
// CHECK:STDOUT:     %E.patt: %pattern_type.29c = symbolic_binding_pattern E, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CC.ref: %CC.type = name_ref CC, file.%CC.decl [concrete = constants.%CC.generic]
// CHECK:STDOUT:     %E.ref: %facet_type = name_ref E, %E.loc19_14.1 [symbolic = %E.loc19_14.2 (constants.%E)]
// CHECK:STDOUT:     %E.as_type.loc19_29.1: type = facet_access_type constants.%E [symbolic = %E.as_type.loc19_29.2 (constants.%E.as_type)]
// CHECK:STDOUT:     %Y.facet.loc19_29.1: %Y.type = facet_value %E.as_type.loc19_29.1, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc19_29.2 (constants.%Y.facet.ecf)]
// CHECK:STDOUT:     %.loc19_29: %Y.type = converted %E.ref, %Y.facet.loc19_29.1 [symbolic = %Y.facet.loc19_29.2 (constants.%Y.facet.ecf)]
// CHECK:STDOUT:     %CC.loc19_29.1: type = class_type @CC, @CC(constants.%Y.facet.ecf) [symbolic = %CC.loc19_29.2 (constants.%CC.6eb)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.loc19_20.1: type = splice_block %.loc19_20.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:       %W.ref: type = name_ref W, file.%W.decl [concrete = constants.%W.type]
// CHECK:STDOUT:       %impl.elem0: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %Y.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%Y.ref, %W.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc19_20.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc19_20.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc19_20.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %E.loc19_14.1: %facet_type = bind_symbolic_name E, 0 [symbolic = %E.loc19_14.2 (constants.%E)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (), @CC.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table, @CC.as.Z.impl(constants.%E) [symbolic = @CC.as.Z.impl.%Z.impl_witness (constants.%Z.impl_witness.481)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.33b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @W {
// CHECK:STDOUT:   %Self: %W.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9bf]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9ba]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @DD.as.Destroy.impl: @DD.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %DD.as.Destroy.impl.Op.decl: %DD.as.Destroy.impl.Op.type = fn_decl @DD.as.Destroy.impl.Op [concrete = constants.%DD.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.3fc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3fc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc7: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.3d1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%DD [concrete = constants.%DD]
// CHECK:STDOUT:     %self: %ptr.3d1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %DD.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @DD.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @DD.as.Y.impl: %DD.ref as %Y.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @DD.as.W.impl: %DD.ref as %W.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%W.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CC.as.Destroy.impl(@CC.%D.loc12_10.2: %Y.type) {
// CHECK:STDOUT:   %D: %Y.type = bind_symbolic_name D, 0 [symbolic = %D (constants.%D)]
// CHECK:STDOUT:   %CC: type = class_type @CC, @CC(%D) [symbolic = %CC (constants.%CC.6c6)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @CC.%Destroy.impl_witness_table, @CC.as.Destroy.impl(%D) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.195)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CC.as.Destroy.impl.Op.type: type = fn_type @CC.as.Destroy.impl.Op, @CC.as.Destroy.impl(%D) [symbolic = %CC.as.Destroy.impl.Op.type (constants.%CC.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %CC.as.Destroy.impl.Op: @CC.as.Destroy.impl.%CC.as.Destroy.impl.Op.type (%CC.as.Destroy.impl.Op.type) = struct_value () [symbolic = %CC.as.Destroy.impl.Op (constants.%CC.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @CC.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %CC.as.Destroy.impl.Op.decl: @CC.as.Destroy.impl.%CC.as.Destroy.impl.Op.type (%CC.as.Destroy.impl.Op.type) = fn_decl @CC.as.Destroy.impl.Op [symbolic = @CC.as.Destroy.impl.%CC.as.Destroy.impl.Op (constants.%CC.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @CC.as.Destroy.impl.Op.%pattern_type (%pattern_type.6d4) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @CC.as.Destroy.impl.Op.%pattern_type (%pattern_type.6d4) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc12_17.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @CC.as.Destroy.impl.Op.%ptr (%ptr.f53) = value_param call_param0
// CHECK:STDOUT:       %.loc12_17.2: type = splice_block %Self.ref [symbolic = %CC (constants.%CC.6c6)] {
// CHECK:STDOUT:         %.loc12_17.3: type = specific_constant constants.%CC.6c6, @CC(constants.%D) [symbolic = %CC (constants.%CC.6c6)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_17.3 [symbolic = %CC (constants.%CC.6c6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @CC.as.Destroy.impl.Op.%ptr (%ptr.f53) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %CC.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @CC.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CC.as.Z.impl(%E.loc19_14.1: %facet_type) {
// CHECK:STDOUT:   %E.loc19_14.2: %facet_type = bind_symbolic_name E, 0 [symbolic = %E.loc19_14.2 (constants.%E)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %E.loc19_14.2, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %E.as_type.loc19_29.2: type = facet_access_type %E.loc19_14.2 [symbolic = %E.as_type.loc19_29.2 (constants.%E.as_type)]
// CHECK:STDOUT:   %Y.facet.loc19_29.2: %Y.type = facet_value %E.as_type.loc19_29.2, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc19_29.2 (constants.%Y.facet.ecf)]
// CHECK:STDOUT:   %CC.loc19_29.2: type = class_type @CC, @CC(%Y.facet.loc19_29.2) [symbolic = %CC.loc19_29.2 (constants.%CC.6eb)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table, @CC.as.Z.impl(%E.loc19_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.481)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %CC.loc19_29.1 as %Z.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DD {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%DD [concrete = constants.%DD]
// CHECK:STDOUT:   impl_decl @DD.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@DD.as.Destroy.impl.%DD.as.Destroy.impl.Op.decl), @DD.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.0ec]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CC(%D.loc12_10.2: %Y.type) {
// CHECK:STDOUT:   %D.loc12_10.1: %Y.type = bind_symbolic_name D, 0 [symbolic = %D.loc12_10.1 (constants.%D)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%CC.6c6 [symbolic = @CC.as.Destroy.impl.%CC (constants.%CC.6c6)]
// CHECK:STDOUT:     impl_decl @CC.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@CC.as.Destroy.impl.%CC.as.Destroy.impl.Op.decl), @CC.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @CC.as.Destroy.impl(constants.%D) [symbolic = @CC.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.195)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CC.6c6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DD.as.Destroy.impl.Op(%self.param: %ptr.3d1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CC.as.Destroy.impl.Op(@CC.%D.loc12_10.2: %Y.type) {
// CHECK:STDOUT:   %D: %Y.type = bind_symbolic_name D, 0 [symbolic = %D (constants.%D)]
// CHECK:STDOUT:   %CC: type = class_type @CC, @CC(%D) [symbolic = %CC (constants.%CC.6c6)]
// CHECK:STDOUT:   %ptr: type = ptr_type %CC [symbolic = %ptr (constants.%ptr.f53)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.6d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @CC.as.Destroy.impl.Op.%ptr (%ptr.f53)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CC.ref: %CC.type = name_ref CC, file.%CC.decl [concrete = constants.%CC.generic]
// CHECK:STDOUT:   %DD.ref: type = name_ref DD, file.%DD.decl [concrete = constants.%DD]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value constants.%DD, (constants.%Y.impl_witness) [concrete = constants.%Y.facet.b41]
// CHECK:STDOUT:   %.loc22_9: %Y.type = converted %DD.ref, %Y.facet [concrete = constants.%Y.facet.b41]
// CHECK:STDOUT:   %CC: type = class_type @CC, @CC(constants.%Y.facet.b41) [concrete = constants.%CC.017]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value constants.%DD, (constants.%Y.impl_witness, constants.%W.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc22_12.1: %facet_type = converted constants.%DD, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value constants.%CC.017, (constants.%Z.impl_witness.9cd) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc22_12.2: %Z.type = converted %CC, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC(constants.%D) {
// CHECK:STDOUT:   %D.loc12_10.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC.as.Destroy.impl(constants.%D) {
// CHECK:STDOUT:   %D => constants.%D
// CHECK:STDOUT:   %CC => constants.%CC.6c6
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC.as.Destroy.impl.Op(constants.%D) {
// CHECK:STDOUT:   %D => constants.%D
// CHECK:STDOUT:   %CC => constants.%CC.6c6
// CHECK:STDOUT:   %ptr => constants.%ptr.f53
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC(constants.%Y.facet.ecf) {
// CHECK:STDOUT:   %D.loc12_10.1 => constants.%Y.facet.ecf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC.as.Z.impl(constants.%E) {
// CHECK:STDOUT:   %E.loc19_14.2 => constants.%E
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness
// CHECK:STDOUT:   %E.as_type.loc19_29.2 => constants.%E.as_type
// CHECK:STDOUT:   %Y.facet.loc19_29.2 => constants.%Y.facet.ecf
// CHECK:STDOUT:   %CC.loc19_29.2 => constants.%CC.6eb
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.481
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC(constants.%Y.facet.b41) {
// CHECK:STDOUT:   %D.loc12_10.1 => constants.%Y.facet.b41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CC.as.Z.impl(constants.%facet_value) {
// CHECK:STDOUT:   %E.loc19_14.2 => constants.%facet_value
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.impl_witness
// CHECK:STDOUT:   %E.as_type.loc19_29.2 => constants.%DD
// CHECK:STDOUT:   %Y.facet.loc19_29.2 => constants.%Y.facet.b41
// CHECK:STDOUT:   %CC.loc19_29.2 => constants.%CC.017
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.9cd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

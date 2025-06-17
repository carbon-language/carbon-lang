// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/canonical_query_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/canonical_query_self.carbon

interface I {
  fn II[self: Self]();
}
interface J {
  fn JJ[self: Self]();
}

fn F(T:! I & J, t: T) {
  // Nesting FacetValue and FacetAccessTypes, the `lookup_impl_witness`
  // instructions should remain canonical, pointing symbolically to `T`, and not
  // include these conversions.
  (t as (((((((T as I) as type) as J) as type) as I) as type) as J)).JJ();
}

fn G() {
  class C {
    extend impl as I {
      fn II[self: Self]() {}
    }
    extend impl as J {
      fn JJ[self: Self]() {}
    }
  }
  // Nesting FacetValue and FacetAccessTypes, the impl lookup for `JJ` resolves
  // to a canonical `impl_witness` for the `impl C as J`.
  (({} as C) as (((((((C as I) as type) as J) as type) as I) as type) as J)).JJ();

  // Produces a specific for `F` with a `FacetValue` to replace the
  // `BindSymbolicName` parameter `T` when evaluated. The `lookup_impl_witness`
  // instructions should be replaced in the specific with concrete
  // `impl_witness` instructions for `impl C as I` and `impl C as J`.
  F(C, {} as C);
}

// CHECK:STDOUT: --- canonical_query_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %II.type.465: type = fn_type @II.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %II.a07: %II.type.465 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%II.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca: type = pattern_type %Self.as_type.3df [symbolic]
// CHECK:STDOUT:   %JJ.type.622: type = fn_type @JJ.1 [concrete]
// CHECK:STDOUT:   %JJ.5f5: %JJ.type.622 = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.78c: %J.assoc_type = assoc_entity element0, @J.%JJ.decl [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.e8c: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %Op.type.9a3: type = fn_type @Op.1, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.e8c = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.2af: type = fn_type_with_self_type %Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %Op.type.1cc: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.ff4: %Op.type.1cc = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.ff4 [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %T: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8f: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a4f: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet.e75: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet.fa4: %J.type = facet_value %T.as_type, (%J.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.935: type = fn_type_with_self_type %JJ.type.622, %J.facet.fa4 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.935 = impl_witness_access %J.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @JJ.1(%J.facet.fa4) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.893: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %II.type.f11: type = fn_type @II.2 [concrete]
// CHECK:STDOUT:   %II.b71: %II.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.98f: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %JJ.type.811: type = fn_type @JJ.2 [concrete]
// CHECK:STDOUT:   %JJ.9c9: %JJ.type.811 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.5df: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.638: type = fn_type_with_self_type %JJ.type.622, %J.facet.5df [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %C, (%I.impl_witness, %J.impl_witness) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.012: %Op.type.1cc = import_ref Core//prelude, loc13_42, loaded [concrete = constants.%Op.ff4]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.012), @impl.865 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f8f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.a4f) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.a4f) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_12.1: type = splice_block %.loc22_12.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %I.ref.loc22: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %J.ref.loc22: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %impl.elem0.loc22: %.2af = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%Op.ff4]
// CHECK:STDOUT:       %bound_method.loc22: <bound method> = bound_method %I.ref.loc22, %impl.elem0.loc22 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %type.and: init type = call %bound_method.loc22(%I.ref.loc22, %J.ref.loc22) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc22_12.2: type = value_of_initializer %type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc22_12.3: type = converted %type.and, %.loc22_12.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_6.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc22_20.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc22_20.1: type = splice_block %.loc22_20.2 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc22: %facet_type = name_ref T, %T.loc22_6.1 [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22_20.1: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_20.2: type = converted %T.ref.loc22, %T.as_type.loc22_20.1 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc22_20.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %II.decl: %II.type.465 = fn_decl @II.1 [concrete = constants.%II.a07] {
// CHECK:STDOUT:     %self.patt: @II.1.%pattern_type (%pattern_type.6de) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @II.1.%pattern_type (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @II.1.%Self.as_type.loc16_15.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc16_15.1: type = splice_block %.loc16_15.2 [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc16_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc16_15.2: type = converted %Self.ref, %Self.as_type.loc16_15.2 [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @II.1.%Self.as_type.loc16_15.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %II.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .II = %assoc0
// CHECK:STDOUT:   .JJ = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = (%II.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %JJ.decl: %JJ.type.622 = fn_decl @JJ.1 [concrete = constants.%JJ.5f5] {
// CHECK:STDOUT:     %self.patt: @JJ.1.%pattern_type (%pattern_type.4ca) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @JJ.1.%pattern_type (%pattern_type.4ca) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @JJ.1.%Self.as_type.loc19_15.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc19_15.1: type = splice_block %.loc19_15.2 [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc19_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc19_15.2: type = converted %Self.ref, %Self.as_type.loc19_15.2 [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @JJ.1.%Self.as_type.loc19_15.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %JJ.decl [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .JJ = %assoc0
// CHECK:STDOUT:   witness = (%JJ.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.093: %Self.ref as %I.ref {
// CHECK:STDOUT:   %II.decl: %II.type.f11 = fn_decl @II.2 [concrete = constants.%II.b71] {
// CHECK:STDOUT:     %self.patt: %pattern_type.893 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.893 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .II = %II.decl
// CHECK:STDOUT:   witness = @C.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.023: %Self.ref as %J.ref {
// CHECK:STDOUT:   %JJ.decl: %JJ.type.811 = fn_decl @JJ.2 [concrete = constants.%JJ.9c9] {
// CHECK:STDOUT:     %self.patt: %pattern_type.893 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.893 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .JJ = %JJ.decl
// CHECK:STDOUT:   witness = @C.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.093 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.093.%II.decl), @impl.093 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   impl_decl @impl.023 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.023.%JJ.decl), @impl.023 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .JJ = <poisoned>
// CHECK:STDOUT:   extend @impl.093.%I.ref
// CHECK:STDOUT:   extend @impl.023.%J.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @II.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc16_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_15.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @II.1.%Self.as_type.loc16_15.1 (%Self.as_type.b70));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @JJ.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc19_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc19_15.1 [symbolic = %pattern_type (constants.%pattern_type.4ca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @JJ.1.%Self.as_type.loc19_15.1 (%Self.as_type.3df));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc22_6.1: %facet_type) {
// CHECK:STDOUT:   %T.loc22_6.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc22_20.2: type = facet_access_type %T.loc22_6.2 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc22_20.2 [symbolic = %pattern_type (constants.%pattern_type.a4f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc22_20.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_6.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet.loc26_18.2: %I.type = facet_value %T.as_type.loc22_20.2, (%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_6.2, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness)]
// CHECK:STDOUT:   %J.facet.loc26_33.2: %J.type = facet_value %T.as_type.loc22_20.2, (%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:   %.loc26_69.2: type = fn_type_with_self_type constants.%JJ.type.622, %J.facet.loc26_33.2 [symbolic = %.loc26_69.2 (constants.%.935)]
// CHECK:STDOUT:   %impl.elem0.loc26_69.2: @F.%.loc26_69.2 (%.935) = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_69.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_69.2: <specific function> = specific_impl_function %impl.elem0.loc26_69.2, @JJ.1(%J.facet.loc26_33.2) [symbolic = %specific_impl_fn.loc26_69.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc22_20.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc22_20.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %T.ref.loc26: %facet_type = name_ref T, %T.loc22_6.1 [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:     %I.ref.loc26_21: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.as_type.loc26_18: type = facet_access_type constants.%T [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.facet.loc26_18.1: %I.type = facet_value %T.as_type.loc26_18, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %.loc26_18: %I.type = converted %T.ref.loc26, %I.facet.loc26_18.1 [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %as_type.loc26_24: type = facet_access_type %.loc26_18 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_24: type = converted %.loc26_18, %as_type.loc26_24 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc26_36: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc26_33.1: %J.type = facet_value constants.%T.as_type, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %.loc26_33: %J.type = converted %.loc26_24, %J.facet.loc26_33.1 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %as_type.loc26_39: type = facet_access_type %.loc26_33 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_39: type = converted %.loc26_33, %as_type.loc26_39 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref.loc26_51: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.facet.loc26_48: %I.type = facet_value constants.%T.as_type, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %.loc26_48: %I.type = converted %.loc26_39, %I.facet.loc26_48 [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %as_type.loc26_54: type = facet_access_type %.loc26_48 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_54: type = converted %.loc26_48, %as_type.loc26_54 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc26_66: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc26_63: %J.type = facet_value constants.%T.as_type, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %.loc26_63: %J.type = converted %.loc26_54, %J.facet.loc26_63 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %as_type.loc26_67: type = facet_access_type %.loc26_63 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_67: type = converted %.loc26_63, %as_type.loc26_67 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %JJ.ref: %J.assoc_type = name_ref JJ, @J.%assoc0 [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:     %T.as_type.loc26_69: type = facet_access_type constants.%T [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_69.1: type = converted constants.%T, %T.as_type.loc26_69 [symbolic = %T.as_type.loc22_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc26_69.1: @F.%.loc26_69.2 (%.935) = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_69.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc26_69: <bound method> = bound_method %t.ref, %impl.elem0.loc26_69.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_69.1: <specific function> = specific_impl_function %impl.elem0.loc26_69.1, @JJ.1(constants.%J.facet.fa4) [symbolic = %specific_impl_fn.loc26_69.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc26_73: <bound method> = bound_method %t.ref, %specific_impl_fn.loc26_69.1
// CHECK:STDOUT:     %.loc26_73: init %empty_tuple.type = call %bound_method.loc26_73(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.loc40_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc40_11: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc40_6.3: init %C = class_init (), %.loc40_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc40_6.4: ref %C = temporary %.loc40_6.2, %.loc40_6.3
// CHECK:STDOUT:   %.loc40_8.1: ref %C = converted %.loc40_6.1, %.loc40_6.4
// CHECK:STDOUT:   %C.ref.loc40_24: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc40_29: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc40_26: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %.loc40_26: %I.type = converted %C.ref.loc40_24, %I.facet.loc40_26 [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %as_type.loc40_32: type = facet_access_type %.loc40_26 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_32: type = converted %.loc40_26, %as_type.loc40_32 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc40_44: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc40_41: %J.type = facet_value constants.%C, (constants.%J.impl_witness) [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %.loc40_41: %J.type = converted %.loc40_32, %J.facet.loc40_41 [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %as_type.loc40_47: type = facet_access_type %.loc40_41 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_47: type = converted %.loc40_41, %as_type.loc40_47 [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc40_59: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc40_56: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %.loc40_56: %I.type = converted %.loc40_47, %I.facet.loc40_56 [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %as_type.loc40_62: type = facet_access_type %.loc40_56 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_62: type = converted %.loc40_56, %as_type.loc40_62 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc40_74: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc40_71: %J.type = facet_value constants.%C, (constants.%J.impl_witness) [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %.loc40_71: %J.type = converted %.loc40_62, %J.facet.loc40_71 [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %as_type.loc40_75: type = facet_access_type %.loc40_71 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_75: type = converted %.loc40_71, %as_type.loc40_75 [concrete = constants.%C]
// CHECK:STDOUT:   %JJ.ref: %J.assoc_type = name_ref JJ, @J.%assoc0 [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:   %impl.elem0: %.638 = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%JJ.9c9]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc40_8.1, %impl.elem0
// CHECK:STDOUT:   %.loc40_8.2: %C = bind_value %.loc40_8.1
// CHECK:STDOUT:   %JJ.call: init %empty_tuple.type = call %bound_method(%.loc40_8.2)
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref.loc46_5: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc46_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc46_14: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc46_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc46_9.3: init %C = class_init (), %.loc46_9.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc46_9.4: ref %C = temporary %.loc46_9.2, %.loc46_9.3
// CHECK:STDOUT:   %.loc46_11.1: ref %C = converted %.loc46_9.1, %.loc46_9.4
// CHECK:STDOUT:   %facet_value.loc46_15.1: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc46_15.1: %facet_type = converted %C.ref.loc46_5, %facet_value.loc46_15.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc46_15.2: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc46_15.2: %facet_type = converted constants.%C, %facet_value.loc46_15.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc46_15.3: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc46_15.3: %facet_type = converted constants.%C, %facet_value.loc46_15.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%facet_value) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc46_11.2: %C = bind_value %.loc46_11.1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc46_11.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @II.2(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JJ.2(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @II.1(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc16_15.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc19_15.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc22_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc22_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(constants.%J.facet.fa4) {
// CHECK:STDOUT:   %Self => constants.%J.facet.fa4
// CHECK:STDOUT:   %Self.as_type.loc19_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @II.1(constants.%I.facet.98f) {
// CHECK:STDOUT:   %Self => constants.%I.facet.98f
// CHECK:STDOUT:   %Self.as_type.loc16_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(constants.%J.facet.5df) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5df
// CHECK:STDOUT:   %Self.as_type.loc19_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc22_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc22_20.2 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %I.facet.loc26_18.2 => constants.%I.facet.98f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %J.facet.loc26_33.2 => constants.%J.facet.5df
// CHECK:STDOUT:   %.loc26_69.2 => constants.%.638
// CHECK:STDOUT:   %impl.elem0.loc26_69.2 => constants.%JJ.9c9
// CHECK:STDOUT:   %specific_impl_fn.loc26_69.2 => constants.%JJ.9c9
// CHECK:STDOUT: }
// CHECK:STDOUT:

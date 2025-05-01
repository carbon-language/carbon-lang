// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/canonical_query_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/canonical_query_self.carbon

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
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.d45: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.ab9: %Op.type.27a = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.19f: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %pattern_type.67d: type = pattern_type %Self.as_type.19f [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness.b7b: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness.0e5: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness.0e5) [concrete]
// CHECK:STDOUT:   %.518: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %I.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %T.527: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8f: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.527 [symbolic]
// CHECK:STDOUT:   %pattern_type.a4f: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df2: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.527, @I [symbolic]
// CHECK:STDOUT:   %I.facet.e75: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.527, @J [symbolic]
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
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.638: type = fn_type_with_self_type %JJ.type.622, %J.facet.5df [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %C, (%I.impl_witness, %J.impl_witness) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst107 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.a46: %BitAnd.assoc_type = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%assoc0.d45]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.a93: %Op.type.27a = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%Op.ab9]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//prelude, inst107 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.140: <witness> = import_ref Core//prelude, loc21_36, loaded [symbolic = @impl.f92.%BitAnd.impl_witness (constants.%BitAnd.impl_witness.b7b)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.1e6), @impl.f92 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
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
// CHECK:STDOUT:     %.loc21_12.1: type = splice_block %.loc21_12.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %I.ref.loc21: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %J.ref.loc21: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %impl.elem0.loc21: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc21_12.1: <bound method> = bound_method %I.ref.loc21, %impl.elem0.loc21 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc21_12.2: <bound method> = bound_method %I.ref.loc21, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %type.and: init type = call %bound_method.loc21_12.2(%I.ref.loc21, %J.ref.loc21) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc21_12.2: type = value_of_initializer %type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc21_12.3: type = converted %type.and, %.loc21_12.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_6.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc21_6.2 (constants.%T.527)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc21_20.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc21_20.1: type = splice_block %.loc21_20.2 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc21: %facet_type = name_ref T, %T.loc21_6.1 [symbolic = %T.loc21_6.2 (constants.%T.527)]
// CHECK:STDOUT:       %T.as_type.loc21_20.1: type = facet_access_type %T.ref.loc21 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc21_20.2: type = converted %T.ref.loc21, %T.as_type.loc21_20.1 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc21_20.2 (%T.as_type) = bind_name t, %t.param
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
// CHECK:STDOUT:     %self.param: @II.1.%Self.as_type.loc15_15.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc15_15.1: type = splice_block %.loc15_15.2 [symbolic = %Self.as_type.loc15_15.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc15_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc15_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc15_15.2: type = converted %Self.ref, %Self.as_type.loc15_15.2 [symbolic = %Self.as_type.loc15_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @II.1.%Self.as_type.loc15_15.1 (%Self.as_type.b70) = bind_name self, %self.param
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
// CHECK:STDOUT:     %self.param: @JJ.1.%Self.as_type.loc18_15.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @JJ.1.%Self.as_type.loc18_15.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %JJ.decl [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .JJ = %assoc0
// CHECK:STDOUT:   witness = (%JJ.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.a46
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(%T) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness.b7b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.140
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %Self.as_type.loc15_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc15_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc15_15.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @II.1.%Self.as_type.loc15_15.1 (%Self.as_type.b70));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @JJ.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc18_15.1 [symbolic = %pattern_type (constants.%pattern_type.4ca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @JJ.1.%Self.as_type.loc18_15.1 (%Self.as_type.3df));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.19f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc21_6.1: %facet_type) {
// CHECK:STDOUT:   %T.loc21_6.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc21_6.2 (constants.%T.527)]
// CHECK:STDOUT:   %T.as_type.loc21_20.2: type = facet_access_type %T.loc21_6.2 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc21_20.2 [symbolic = %pattern_type (constants.%pattern_type.a4f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc21_20.2 [symbolic = %require_complete (constants.%require_complete.df2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc21_6.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet.loc25_18.2: %I.type = facet_value %T.as_type.loc21_20.2, (%I.lookup_impl_witness) [symbolic = %I.facet.loc25_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc21_6.2, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness)]
// CHECK:STDOUT:   %J.facet.loc25_33.2: %J.type = facet_value %T.as_type.loc21_20.2, (%J.lookup_impl_witness) [symbolic = %J.facet.loc25_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:   %.loc25_69.2: type = fn_type_with_self_type constants.%JJ.type.622, %J.facet.loc25_33.2 [symbolic = %.loc25_69.2 (constants.%.935)]
// CHECK:STDOUT:   %impl.elem0.loc25_69.2: @F.%.loc25_69.2 (%.935) = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_69.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_69.2: <specific function> = specific_impl_function %impl.elem0.loc25_69.2, @JJ.1(%J.facet.loc25_33.2) [symbolic = %specific_impl_fn.loc25_69.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc21_20.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc21_20.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %T.ref.loc25: %facet_type = name_ref T, %T.loc21_6.1 [symbolic = %T.loc21_6.2 (constants.%T.527)]
// CHECK:STDOUT:     %I.ref.loc25_21: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.as_type.loc25_18: type = facet_access_type constants.%T.527 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.facet.loc25_18.1: %I.type = facet_value %T.as_type.loc25_18, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc25_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %.loc25_18: %I.type = converted %T.ref.loc25, %I.facet.loc25_18.1 [symbolic = %I.facet.loc25_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %as_type.loc25_24: type = facet_access_type %.loc25_18 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_24: type = converted %.loc25_18, %as_type.loc25_24 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc25_36: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc25_33.1: %J.type = facet_value constants.%T.as_type, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc25_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %.loc25_33: %J.type = converted %.loc25_24, %J.facet.loc25_33.1 [symbolic = %J.facet.loc25_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %as_type.loc25_39: type = facet_access_type %.loc25_33 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_39: type = converted %.loc25_33, %as_type.loc25_39 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref.loc25_51: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.facet.loc25_48: %I.type = facet_value constants.%T.as_type, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc25_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %.loc25_48: %I.type = converted %.loc25_39, %I.facet.loc25_48 [symbolic = %I.facet.loc25_18.2 (constants.%I.facet.e75)]
// CHECK:STDOUT:     %as_type.loc25_54: type = facet_access_type %.loc25_48 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_54: type = converted %.loc25_48, %as_type.loc25_54 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc25_66: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc25_63: %J.type = facet_value constants.%T.as_type, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc25_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %.loc25_63: %J.type = converted %.loc25_54, %J.facet.loc25_63 [symbolic = %J.facet.loc25_33.2 (constants.%J.facet.fa4)]
// CHECK:STDOUT:     %as_type.loc25_67: type = facet_access_type %.loc25_63 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_67: type = converted %.loc25_63, %as_type.loc25_67 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %JJ.ref: %J.assoc_type = name_ref JJ, @J.%assoc0 [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:     %T.as_type.loc25_69: type = facet_access_type constants.%T.527 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_69.1: type = converted constants.%T.527, %T.as_type.loc25_69 [symbolic = %T.as_type.loc21_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc25_69.1: @F.%.loc25_69.2 (%.935) = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_69.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc25_69: <bound method> = bound_method %t.ref, %impl.elem0.loc25_69.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_69.1: <specific function> = specific_impl_function %impl.elem0.loc25_69.1, @JJ.1(constants.%J.facet.fa4) [symbolic = %specific_impl_fn.loc25_69.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc25_73: <bound method> = bound_method %t.ref, %specific_impl_fn.loc25_69.1
// CHECK:STDOUT:     %.loc25_73: init %empty_tuple.type = call %bound_method.loc25_73(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.loc39_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc39_11: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc39_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc39_6.3: init %C = class_init (), %.loc39_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc39_6.4: ref %C = temporary %.loc39_6.2, %.loc39_6.3
// CHECK:STDOUT:   %.loc39_8.1: ref %C = converted %.loc39_6.1, %.loc39_6.4
// CHECK:STDOUT:   %C.ref.loc39_24: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc39_29: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc39_26: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %.loc39_26: %I.type = converted %C.ref.loc39_24, %I.facet.loc39_26 [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %as_type.loc39_32: type = facet_access_type %.loc39_26 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc39_32: type = converted %.loc39_26, %as_type.loc39_32 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc39_44: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc39_41: %J.type = facet_value constants.%C, (constants.%J.impl_witness) [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %.loc39_41: %J.type = converted %.loc39_32, %J.facet.loc39_41 [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %as_type.loc39_47: type = facet_access_type %.loc39_41 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc39_47: type = converted %.loc39_41, %as_type.loc39_47 [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc39_59: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc39_56: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %.loc39_56: %I.type = converted %.loc39_47, %I.facet.loc39_56 [concrete = constants.%I.facet.98f]
// CHECK:STDOUT:   %as_type.loc39_62: type = facet_access_type %.loc39_56 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc39_62: type = converted %.loc39_56, %as_type.loc39_62 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc39_74: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc39_71: %J.type = facet_value constants.%C, (constants.%J.impl_witness) [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %.loc39_71: %J.type = converted %.loc39_62, %J.facet.loc39_71 [concrete = constants.%J.facet.5df]
// CHECK:STDOUT:   %as_type.loc39_75: type = facet_access_type %.loc39_71 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc39_75: type = converted %.loc39_71, %as_type.loc39_75 [concrete = constants.%C]
// CHECK:STDOUT:   %JJ.ref: %J.assoc_type = name_ref JJ, @J.%assoc0 [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:   %impl.elem0: %.638 = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%JJ.9c9]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc39_8.1, %impl.elem0
// CHECK:STDOUT:   %.loc39_8.2: %C = bind_value %.loc39_8.1
// CHECK:STDOUT:   %JJ.call: init %empty_tuple.type = call %bound_method(%.loc39_8.2)
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref.loc45_5: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc45_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc45_14: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc45_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc45_9.3: init %C = class_init (), %.loc45_9.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc45_9.4: ref %C = temporary %.loc45_9.2, %.loc45_9.3
// CHECK:STDOUT:   %.loc45_11.1: ref %C = converted %.loc45_9.1, %.loc45_9.4
// CHECK:STDOUT:   %facet_value.loc45_15.1: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_15.1: %facet_type = converted %C.ref.loc45_5, %facet_value.loc45_15.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc45_15.2: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_15.2: %facet_type = converted constants.%C, %facet_value.loc45_15.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc45_15.3: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_15.3: %facet_type = converted constants.%C, %facet_value.loc45_15.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%facet_value) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc45_11.2: %C = bind_value %.loc45_11.1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc45_11.2)
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
// CHECK:STDOUT:   %Self.as_type.loc15_15.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.19f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness.b7b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness.0e5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.527) {
// CHECK:STDOUT:   %T.loc21_6.2 => constants.%T.527
// CHECK:STDOUT:   %T.as_type.loc21_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(constants.%J.facet.fa4) {
// CHECK:STDOUT:   %Self => constants.%J.facet.fa4
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(@F.%J.facet.loc25_33.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @II.1(constants.%I.facet.98f) {
// CHECK:STDOUT:   %Self => constants.%I.facet.98f
// CHECK:STDOUT:   %Self.as_type.loc15_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @JJ.1(constants.%J.facet.5df) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5df
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc21_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc21_20.2 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %I.facet.loc25_18.2 => constants.%I.facet.98f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %J.facet.loc25_33.2 => constants.%J.facet.5df
// CHECK:STDOUT:   %.loc25_69.2 => constants.%.638
// CHECK:STDOUT:   %impl.elem0.loc25_69.2 => constants.%JJ.9c9
// CHECK:STDOUT:   %specific_impl_fn.loc25_69.2 => constants.%JJ.9c9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %Self.as_type.7f0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type.419 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %pattern_type.a80: type = pattern_type %Self.as_type.560 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.220: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%BitAnd.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness %BitAnd.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest.loc9_14.2) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Op.1.%pattern_type (%pattern_type.a80) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.220]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T.loc21_14.2) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc21_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %other.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern other [concrete]
// CHECK:STDOUT:       %other.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%BitAnd.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:   %pattern_type.loc10_14: type = pattern_type %Self.as_type.loc10_20.1 [symbolic = %pattern_type.loc10_14 (constants.%pattern_type.947)]
// CHECK:STDOUT:   %pattern_type.loc10_28: type = pattern_type %Dest [symbolic = %pattern_type.loc10_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)) -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:   %pattern_type.loc14_14: type = pattern_type %Self.as_type.loc14_20.1 [symbolic = %pattern_type.loc14_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc14_28: type = pattern_type %Dest [symbolic = %pattern_type.loc14_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)) -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc18_15.1 [symbolic = %pattern_type (constants.%pattern_type.a80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560), %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T), %other.param: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT:   %pattern_type.loc10_14 => constants.%pattern_type.947
// CHECK:STDOUT:   %pattern_type.loc10_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT:   %pattern_type.loc14_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc14_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

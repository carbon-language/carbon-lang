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
// CHECK:STDOUT:   %Self.7ee: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.a67: type = facet_access_type %Self.7ee [symbolic]
// CHECK:STDOUT:   %pattern_type.d22: type = pattern_type %Self.as_type.a67 [symbolic]
// CHECK:STDOUT:   %I.II.type: type = fn_type @I.II [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.II: %I.II.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%I.II.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.bf6: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.4e7: type = facet_access_type %Self.bf6 [symbolic]
// CHECK:STDOUT:   %pattern_type.8ba: type = pattern_type %Self.as_type.4e7 [symbolic]
// CHECK:STDOUT:   %J.JJ.type: type = fn_type @J.JJ [concrete]
// CHECK:STDOUT:   %J.JJ: %J.JJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.78c: %J.assoc_type = assoc_entity element0, @J.%J.JJ.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.8a6: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.8a6 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %I.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %T: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.add: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.94b: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.387: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet.6ef: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet.10a: %J.type = facet_value %T.as_type, (%J.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.4b9: type = fn_type_with_self_type %J.JJ.type, %J.facet.10a [symbolic]
// CHECK:STDOUT:   %impl.elem0.366: %.4b9 = impl_witness_access %J.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7b5: <specific function> = specific_impl_function %impl.elem0.366, @J.JJ(%J.facet.10a) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.893: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.I.impl.II.type: type = fn_type @C.as.I.impl.II [concrete]
// CHECK:STDOUT:   %C.as.I.impl.II: %C.as.I.impl.II.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.0a4: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.J.impl.JJ.type: type = fn_type @C.as.J.impl.JJ [concrete]
// CHECK:STDOUT:   %C.as.J.impl.JJ: %C.as.J.impl.JJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.5ad: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.be8: type = fn_type_with_self_type %J.JJ.type, %J.facet.5ad [concrete]
// CHECK:STDOUT:   %facet_value.d36: %facet_type = facet_value %C, (%I.impl_witness, %J.impl_witness) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%facet_value.d36) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.c6b: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.cb1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.c6b) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.f2c: %AggregateT.as_type.as.Destroy.impl.Op.type.cb1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8e6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.f2c, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.c6b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.636: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc25_42, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.636), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.add = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.94b) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.94b) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_12.1: type = splice_block %.loc22_12.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc22: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %J.ref.loc22: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %impl.elem0.loc22: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc22: <bound method> = bound_method %I.ref.loc22, %impl.elem0.loc22 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc22(%I.ref.loc22, %J.ref.loc22) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc22_12.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc22_12.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc22_12.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_6.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc22_20.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc22_20.1: type = splice_block %.loc22_20.2 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc22: %facet_type = name_ref T, %T.loc22_6.2 [symbolic = %T.loc22_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22_20.2: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_20.2: type = converted %T.ref.loc22, %T.as_type.loc22_20.2 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc22_20.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:   %I.II.decl: %I.II.type = fn_decl @I.II [concrete = constants.%I.II] {
// CHECK:STDOUT:     %self.patt: @I.II.%pattern_type (%pattern_type.d22) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.II.%pattern_type (%pattern_type.d22) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.II.%Self.as_type.loc16_15.1 (%Self.as_type.a67) = value_param call_param0
// CHECK:STDOUT:     %.loc16_15.1: type = splice_block %.loc16_15.2 [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.a67)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.7ee)]
// CHECK:STDOUT:       %Self.as_type.loc16_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.a67)]
// CHECK:STDOUT:       %.loc16_15.2: type = converted %Self.ref, %Self.as_type.loc16_15.2 [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.a67)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.II.%Self.as_type.loc16_15.1 (%Self.as_type.a67) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.II.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .II = %assoc0
// CHECK:STDOUT:   .JJ = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = (%I.II.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bf6]
// CHECK:STDOUT:   %J.JJ.decl: %J.JJ.type = fn_decl @J.JJ [concrete = constants.%J.JJ] {
// CHECK:STDOUT:     %self.patt: @J.JJ.%pattern_type (%pattern_type.8ba) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.JJ.%pattern_type (%pattern_type.8ba) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @J.JJ.%Self.as_type.loc19_15.1 (%Self.as_type.4e7) = value_param call_param0
// CHECK:STDOUT:     %.loc19_15.1: type = splice_block %.loc19_15.2 [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.4e7)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.bf6)]
// CHECK:STDOUT:       %Self.as_type.loc19_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.4e7)]
// CHECK:STDOUT:       %.loc19_15.2: type = converted %Self.ref, %Self.as_type.loc19_15.2 [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.4e7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.JJ.%Self.as_type.loc19_15.1 (%Self.as_type.4e7) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.JJ.decl [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .JJ = %assoc0
// CHECK:STDOUT:   witness = (%J.JJ.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.II.decl: %C.as.I.impl.II.type = fn_decl @C.as.I.impl.II [concrete = constants.%C.as.I.impl.II] {
// CHECK:STDOUT:     %self.patt: %pattern_type.893 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.893 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .II = %C.as.I.impl.II.decl
// CHECK:STDOUT:   witness = @C.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %C.as.J.impl.JJ.decl: %C.as.J.impl.JJ.type = fn_decl @C.as.J.impl.JJ [concrete = constants.%C.as.J.impl.JJ] {
// CHECK:STDOUT:     %self.patt: %pattern_type.893 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.893 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .JJ = %C.as.J.impl.JJ.decl
// CHECK:STDOUT:   witness = @C.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@C.as.I.impl.%C.as.I.impl.II.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@C.as.J.impl.%C.as.J.impl.JJ.decl), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .JJ = <poisoned>
// CHECK:STDOUT:   extend @C.as.I.impl.%I.ref
// CHECK:STDOUT:   extend @C.as.J.impl.%J.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.II(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.7ee)]
// CHECK:STDOUT:   %Self.as_type.loc16_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_15.1 (constants.%Self.as_type.a67)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_15.1 [symbolic = %pattern_type (constants.%pattern_type.d22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.II.%Self.as_type.loc16_15.1 (%Self.as_type.a67));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.JJ(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.bf6)]
// CHECK:STDOUT:   %Self.as_type.loc19_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_15.1 (constants.%Self.as_type.4e7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc19_15.1 [symbolic = %pattern_type (constants.%pattern_type.8ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.JJ.%Self.as_type.loc19_15.1 (%Self.as_type.4e7));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc22_6.2: %facet_type) {
// CHECK:STDOUT:   %T.loc22_6.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc22_20.1: type = facet_access_type %T.loc22_6.1 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc22_20.1 [symbolic = %pattern_type (constants.%pattern_type.94b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc22_20.1 [symbolic = %require_complete (constants.%require_complete.387)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_6.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet.loc26_18.2: %I.type = facet_value %T.as_type.loc22_20.1, (%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.6ef)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_6.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness)]
// CHECK:STDOUT:   %J.facet.loc26_33.2: %J.type = facet_value %T.as_type.loc22_20.1, (%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.10a)]
// CHECK:STDOUT:   %.loc26_69: type = fn_type_with_self_type constants.%J.JJ.type, %J.facet.loc26_33.2 [symbolic = %.loc26_69 (constants.%.4b9)]
// CHECK:STDOUT:   %impl.elem0.loc26_69.2: @F.%.loc26_69 (%.4b9) = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_69.2 (constants.%impl.elem0.366)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_69.2: <specific function> = specific_impl_function %impl.elem0.loc26_69.2, @J.JJ(%J.facet.loc26_33.2) [symbolic = %specific_impl_fn.loc26_69.2 (constants.%specific_impl_fn.7b5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc22_20.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc22_20.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %T.ref.loc26: %facet_type = name_ref T, %T.loc22_6.2 [symbolic = %T.loc22_6.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref.loc26_21: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type %T.ref.loc26 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.facet.loc26_18.1: %I.type = facet_value %T.as_type.loc26, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.6ef)]
// CHECK:STDOUT:     %.loc26_18: %I.type = converted %T.ref.loc26, %I.facet.loc26_18.1 [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.6ef)]
// CHECK:STDOUT:     %as_type.loc26_24: type = facet_access_type %.loc26_18 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_24: type = converted %.loc26_18, %as_type.loc26_24 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc26_36: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc26_33.1: %J.type = facet_value %.loc26_24, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.10a)]
// CHECK:STDOUT:     %.loc26_33: %J.type = converted %.loc26_24, %J.facet.loc26_33.1 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.10a)]
// CHECK:STDOUT:     %as_type.loc26_39: type = facet_access_type %.loc26_33 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_39: type = converted %.loc26_33, %as_type.loc26_39 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref.loc26_51: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.facet.loc26_48: %I.type = facet_value %.loc26_39, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.6ef)]
// CHECK:STDOUT:     %.loc26_48: %I.type = converted %.loc26_39, %I.facet.loc26_48 [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.6ef)]
// CHECK:STDOUT:     %as_type.loc26_54: type = facet_access_type %.loc26_48 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_54: type = converted %.loc26_48, %as_type.loc26_54 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc26_66: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc26_63: %J.type = facet_value %.loc26_54, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.10a)]
// CHECK:STDOUT:     %.loc26_63: %J.type = converted %.loc26_54, %J.facet.loc26_63 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.10a)]
// CHECK:STDOUT:     %as_type.loc26_67: type = facet_access_type %.loc26_63 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_67: type = converted %.loc26_63, %as_type.loc26_67 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %JJ.ref: %J.assoc_type = name_ref JJ, @J.%assoc0 [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:     %impl.elem0.loc26_69.1: @F.%.loc26_69 (%.4b9) = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_69.2 (constants.%impl.elem0.366)]
// CHECK:STDOUT:     %bound_method.loc26_69: <bound method> = bound_method %t.ref, %impl.elem0.loc26_69.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_69.1: <specific function> = specific_impl_function %impl.elem0.loc26_69.1, @J.JJ(constants.%J.facet.10a) [symbolic = %specific_impl_fn.loc26_69.2 (constants.%specific_impl_fn.7b5)]
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
// CHECK:STDOUT:   %I.facet.loc40_26: %I.type = facet_value %C.ref.loc40_24, (constants.%I.impl_witness) [concrete = constants.%I.facet.0a4]
// CHECK:STDOUT:   %.loc40_26: %I.type = converted %C.ref.loc40_24, %I.facet.loc40_26 [concrete = constants.%I.facet.0a4]
// CHECK:STDOUT:   %as_type.loc40_32: type = facet_access_type %.loc40_26 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_32: type = converted %.loc40_26, %as_type.loc40_32 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc40_44: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc40_41: %J.type = facet_value %.loc40_32, (constants.%J.impl_witness) [concrete = constants.%J.facet.5ad]
// CHECK:STDOUT:   %.loc40_41: %J.type = converted %.loc40_32, %J.facet.loc40_41 [concrete = constants.%J.facet.5ad]
// CHECK:STDOUT:   %as_type.loc40_47: type = facet_access_type %.loc40_41 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_47: type = converted %.loc40_41, %as_type.loc40_47 [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc40_59: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc40_56: %I.type = facet_value %.loc40_47, (constants.%I.impl_witness) [concrete = constants.%I.facet.0a4]
// CHECK:STDOUT:   %.loc40_56: %I.type = converted %.loc40_47, %I.facet.loc40_56 [concrete = constants.%I.facet.0a4]
// CHECK:STDOUT:   %as_type.loc40_62: type = facet_access_type %.loc40_56 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_62: type = converted %.loc40_56, %as_type.loc40_62 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc40_74: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc40_71: %J.type = facet_value %.loc40_62, (constants.%J.impl_witness) [concrete = constants.%J.facet.5ad]
// CHECK:STDOUT:   %.loc40_71: %J.type = converted %.loc40_62, %J.facet.loc40_71 [concrete = constants.%J.facet.5ad]
// CHECK:STDOUT:   %as_type.loc40_75: type = facet_access_type %.loc40_71 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_75: type = converted %.loc40_71, %as_type.loc40_75 [concrete = constants.%C]
// CHECK:STDOUT:   %JJ.ref: %J.assoc_type = name_ref JJ, @J.%assoc0 [concrete = constants.%assoc0.78c]
// CHECK:STDOUT:   %impl.elem0: %.be8 = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%C.as.J.impl.JJ]
// CHECK:STDOUT:   %bound_method.loc40_77: <bound method> = bound_method %.loc40_8.1, %impl.elem0
// CHECK:STDOUT:   %.loc40_8.2: %C = bind_value %.loc40_8.1
// CHECK:STDOUT:   %C.as.J.impl.JJ.call: init %empty_tuple.type = call %bound_method.loc40_77(%.loc40_8.2)
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref.loc46_5: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc46_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc46_14: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc46_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc46_9.3: init %C = class_init (), %.loc46_9.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc46_9.4: ref %C = temporary %.loc46_9.2, %.loc46_9.3
// CHECK:STDOUT:   %.loc46_11.1: ref %C = converted %.loc46_9.1, %.loc46_9.4
// CHECK:STDOUT:   %facet_value.loc46_15.1: %facet_type = facet_value %C.ref.loc46_5, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc46_15.1: %facet_type = converted %C.ref.loc46_5, %facet_value.loc46_15.1 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %facet_value.loc46_15.2: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc46_15.2: %facet_type = converted constants.%C, %facet_value.loc46_15.2 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %facet_value.loc46_15.3: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc46_15.3: %facet_type = converted constants.%C, %facet_value.loc46_15.3 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%facet_value.d36) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc46_11.2: %C = bind_value %.loc46_11.1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc46_11.2)
// CHECK:STDOUT:   %facet_value.loc46_9: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value.c6b]
// CHECK:STDOUT:   %.loc46_9.5: %type_where = converted constants.%C, %facet_value.loc46_9 [concrete = constants.%facet_value.c6b]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc46: <bound method> = bound_method %.loc46_9.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.f2c
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.f2c, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.c6b) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46: <bound method> = bound_method %.loc46_9.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc46: %ptr.8e6 = addr_of %.loc46_9.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc46: init %empty_tuple.type = call %bound_method.loc46(%addr.loc46)
// CHECK:STDOUT:   %facet_value.loc40: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value.c6b]
// CHECK:STDOUT:   %.loc40_6.5: %type_where = converted constants.%C, %facet_value.loc40 [concrete = constants.%facet_value.c6b]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc40: <bound method> = bound_method %.loc40_6.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.f2c
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.f2c, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.c6b) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40_6: <bound method> = bound_method %.loc40_6.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc40: %ptr.8e6 = addr_of %.loc40_6.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc40: init %empty_tuple.type = call %bound_method.loc40_6(%addr.loc40)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.II(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.J.impl.JJ(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.II(constants.%Self.7ee) {
// CHECK:STDOUT:   %Self => constants.%Self.7ee
// CHECK:STDOUT:   %Self.as_type.loc16_15.1 => constants.%Self.as_type.a67
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.JJ(constants.%Self.bf6) {
// CHECK:STDOUT:   %Self => constants.%Self.bf6
// CHECK:STDOUT:   %Self.as_type.loc19_15.1 => constants.%Self.as_type.4e7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc22_6.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc22_20.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.94b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.JJ(constants.%J.facet.10a) {
// CHECK:STDOUT:   %Self => constants.%J.facet.10a
// CHECK:STDOUT:   %Self.as_type.loc19_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.94b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.II(constants.%I.facet.0a4) {
// CHECK:STDOUT:   %Self => constants.%I.facet.0a4
// CHECK:STDOUT:   %Self.as_type.loc16_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.JJ(constants.%J.facet.5ad) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5ad
// CHECK:STDOUT:   %Self.as_type.loc19_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value.d36) {
// CHECK:STDOUT:   %T.loc22_6.1 => constants.%facet_value.d36
// CHECK:STDOUT:   %T.as_type.loc22_20.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.893
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %I.facet.loc26_18.2 => constants.%I.facet.0a4
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %J.facet.loc26_33.2 => constants.%J.facet.5ad
// CHECK:STDOUT:   %.loc26_69 => constants.%.be8
// CHECK:STDOUT:   %impl.elem0.loc26_69.2 => constants.%C.as.J.impl.JJ
// CHECK:STDOUT:   %specific_impl_fn.loc26_69.2 => constants.%C.as.J.impl.JJ
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
  fn II(self);
}
interface J {
  fn JJ(self);
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
      fn II(unused self) {}
    }
    extend impl as J {
      fn JJ(unused self) {}
    }
  }
  // Nesting FacetValue and FacetAccessTypes, the impl lookup for `JJ` resolves
  // to a canonical `impl_witness` for the `impl C as J`.
  (({} as C) as (((((((C as I) as type) as J) as type) as I) as type) as J)).JJ();

  // Produces a specific for `F` with a `FacetValue` to replace the
  // `SymbolicBinding` parameter `T` when evaluated. The `lookup_impl_witness`
  // instructions should be replaced in the specific with concrete
  // `impl_witness` instructions for `impl C as I` and `impl C as J`.
  F(C, {} as C);
}


// CHECK:STDOUT: --- canonical_query_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.II.type.8a2: type = fn_type @I.WithSelf.II, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.II.09c: %I.WithSelf.II.type.8a2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.153: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.II.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.976: type = facet_access_type %Self.653 [symbolic]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %Self.as_type.976 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.JJ.type.e02: type = fn_type @J.WithSelf.JJ, @J.WithSelf(%Self.653) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.JJ.1d0: %J.WithSelf.JJ.type.e02 = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.7d8: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.JJ.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %I.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.936: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet.d3d: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet.cbc: %J.type = facet_value %T.as_type, (%J.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.II.type.21c: type = fn_type @I.WithSelf.II, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.II.a59: %I.WithSelf.II.type.21c = struct_value () [symbolic]
// CHECK:STDOUT:   %J.WithSelf.JJ.type.f9e: type = fn_type @J.WithSelf.JJ, @J.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.JJ.e0c: %J.WithSelf.JJ.type.f9e = struct_value () [symbolic]
// CHECK:STDOUT:   %J.WithSelf.JJ.type.2d7: type = fn_type @J.WithSelf.JJ, @J.WithSelf(%J.facet.cbc) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.JJ.6d6: %J.WithSelf.JJ.type.2d7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.443: type = fn_type_with_self_type %J.WithSelf.JJ.type.2d7, %J.facet.cbc [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.443 = impl_witness_access %J.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @J.WithSelf.JJ(%J.facet.cbc) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f79: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.I.impl.II.type: type = fn_type @C.as.I.impl.II [concrete]
// CHECK:STDOUT:   %C.as.I.impl.II: %C.as.I.impl.II.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.947: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.II.type.9c8: type = fn_type @I.WithSelf.II, @I.WithSelf(%I.facet.947) [concrete]
// CHECK:STDOUT:   %I.WithSelf.II.622: %I.WithSelf.II.type.9c8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.J.impl.JJ.type: type = fn_type @C.as.J.impl.JJ [concrete]
// CHECK:STDOUT:   %C.as.J.impl.JJ: %C.as.J.impl.JJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.a74: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.JJ.type.95f: type = fn_type @J.WithSelf.JJ, @J.WithSelf(%J.facet.a74) [concrete]
// CHECK:STDOUT:   %J.WithSelf.JJ.d09: %J.WithSelf.JJ.type.95f = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.5d3: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %J.WithSelf.JJ.type.383: type = fn_type @J.WithSelf.JJ, @J.WithSelf(%C.type.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.JJ.87d: %J.WithSelf.JJ.type.383 = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.II.type.92c: type = fn_type @I.WithSelf.II, @I.WithSelf(%C.type.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.II.03d: %I.WithSelf.II.type.92c = struct_value () [concrete]
// CHECK:STDOUT:   %.dd2: type = fn_type_with_self_type %J.WithSelf.JJ.type.95f, %J.facet.a74 [concrete]
// CHECK:STDOUT:   %C.as.J.impl.JJ.bound: <bound method> = bound_method %.5d3, %C.as.J.impl.JJ [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %C, (%I.impl_witness, %J.impl_witness) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc46_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.5d3, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.fb7 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.936) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.936) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_12.1: type = splice_block %.loc22_12.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc22: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %J.ref.loc22: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %impl.elem0.loc22: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc22: <bound method> = bound_method %I.ref.loc22, %impl.elem0.loc22 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc22(%I.ref.loc22, %J.ref.loc22) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc22_12.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc22_12.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc22_12.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_7.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc22_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc22_20.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc22_20.1: type = splice_block %.loc22_20.2 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc22: %facet_type = name_ref T, %T.loc22_7.2 [symbolic = %T.loc22_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22_20.2: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_20.2: type = converted %T.ref.loc22, %T.as_type.loc22_20.2 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc22_20.1 (%T.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.II.decl: @I.WithSelf.%I.WithSelf.II.type (%I.WithSelf.II.type.8a2) = fn_decl @I.WithSelf.II [symbolic = @I.WithSelf.%I.WithSelf.II (constants.%I.WithSelf.II.09c)] {
// CHECK:STDOUT:     %self.param_patt: @I.WithSelf.II.%pattern_type (%pattern_type.c60) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @I.WithSelf.II.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.II.%Self.as_type.loc16_9.1 (%Self.as_type.c84) = value_param call_param0
// CHECK:STDOUT:     %.loc16_9.1: type = splice_block %.loc16_9.2 [symbolic = %Self.as_type.loc16_9.1 (constants.%Self.as_type.c84)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %Self.as_type.loc16_9.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_9.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:       %.loc16_9.2: type = converted %Self.ref, %Self.as_type.loc16_9.2 [symbolic = %Self.as_type.loc16_9.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.II.%Self.as_type.loc16_9.1 (%Self.as_type.c84) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.II.decl [concrete = constants.%assoc0.153]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .II = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .JJ = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.II.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.JJ.decl: @J.WithSelf.%J.WithSelf.JJ.type (%J.WithSelf.JJ.type.e02) = fn_decl @J.WithSelf.JJ [symbolic = @J.WithSelf.%J.WithSelf.JJ (constants.%J.WithSelf.JJ.1d0)] {
// CHECK:STDOUT:     %self.param_patt: @J.WithSelf.JJ.%pattern_type (%pattern_type.506) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @J.WithSelf.JJ.%pattern_type (%pattern_type.506) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @J.WithSelf.JJ.%Self.as_type.loc19_9.1 (%Self.as_type.976) = value_param call_param0
// CHECK:STDOUT:     %.loc19_9.1: type = splice_block %.loc19_9.2 [symbolic = %Self.as_type.loc19_9.1 (constants.%Self.as_type.976)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.653)]
// CHECK:STDOUT:       %Self.as_type.loc19_9.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc19_9.1 (constants.%Self.as_type.976)]
// CHECK:STDOUT:       %.loc19_9.2: type = converted %Self.ref, %Self.as_type.loc19_9.2 [symbolic = %Self.as_type.loc19_9.1 (constants.%Self.as_type.976)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.WithSelf.JJ.%Self.as_type.loc19_9.1 (%Self.as_type.976) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.JJ.decl [concrete = constants.%assoc0.7d8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .JJ = @J.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@J.WithSelf.%J.WithSelf.JJ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.II.decl: %C.as.I.impl.II.type = fn_decl @C.as.I.impl.II [concrete = constants.%C.as.I.impl.II] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f79 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f79 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%C.as.I.impl.II.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .II = %C.as.I.impl.II.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %C.as.J.impl.JJ.decl: %C.as.J.impl.JJ.type = fn_decl @C.as.J.impl.JJ [concrete = constants.%C.as.J.impl.JJ] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f79 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f79 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%C.as.J.impl.JJ.decl), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .JJ = %C.as.J.impl.JJ.decl
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @I.WithSelf.II(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:   %Self.as_type.loc16_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_9.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_9.1 [symbolic = %pattern_type (constants.%pattern_type.c60)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.II.%Self.as_type.loc16_9.1 (%Self.as_type.c84));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.JJ(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.653)]
// CHECK:STDOUT:   %Self.as_type.loc19_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_9.1 (constants.%Self.as_type.976)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc19_9.1 [symbolic = %pattern_type (constants.%pattern_type.506)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.WithSelf.JJ.%Self.as_type.loc19_9.1 (%Self.as_type.976));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc22_7.2: %facet_type) {
// CHECK:STDOUT:   %T.loc22_7.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc22_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc22_20.1: type = facet_access_type %T.loc22_7.1 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc22_20.1 [symbolic = %pattern_type (constants.%pattern_type.936)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc22_20.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet.loc26_18.2: %I.type = facet_value %T.as_type.loc22_20.1, (%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.d3d)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc22_7.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness)]
// CHECK:STDOUT:   %J.facet.loc26_33.2: %J.type = facet_value %T.as_type.loc22_20.1, (%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:   %J.WithSelf.JJ.type: type = fn_type @J.WithSelf.JJ, @J.WithSelf(%J.facet.loc26_33.2) [symbolic = %J.WithSelf.JJ.type (constants.%J.WithSelf.JJ.type.2d7)]
// CHECK:STDOUT:   %.loc26_69: type = fn_type_with_self_type %J.WithSelf.JJ.type, %J.facet.loc26_33.2 [symbolic = %.loc26_69 (constants.%.443)]
// CHECK:STDOUT:   %impl.elem0.loc26_69.2: @F.%.loc26_69 (%.443) = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_69.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_69.2: <specific function> = specific_impl_function %impl.elem0.loc26_69.2, @J.WithSelf.JJ(%J.facet.loc26_33.2) [symbolic = %specific_impl_fn.loc26_69.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc22_20.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc22_20.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %T.ref.loc26: %facet_type = name_ref T, %T.loc22_7.2 [symbolic = %T.loc22_7.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref.loc26_21: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type %T.ref.loc26 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.facet.loc26_18.1: %I.type = facet_value %T.as_type.loc26, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.d3d)]
// CHECK:STDOUT:     %.loc26_18: %I.type = converted %T.ref.loc26, %I.facet.loc26_18.1 [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.d3d)]
// CHECK:STDOUT:     %.loc26_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %as_type.loc26_24: type = facet_access_type %.loc26_18 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_24: type = converted %.loc26_18, %as_type.loc26_24 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc26_36: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc26_33.1: %J.type = facet_value %.loc26_24, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:     %.loc26_33: %J.type = converted %.loc26_24, %J.facet.loc26_33.1 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:     %.loc26_42: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %as_type.loc26_39: type = facet_access_type %.loc26_33 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_39: type = converted %.loc26_33, %as_type.loc26_39 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref.loc26_51: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.facet.loc26_48: %I.type = facet_value %.loc26_39, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.d3d)]
// CHECK:STDOUT:     %.loc26_48: %I.type = converted %.loc26_39, %I.facet.loc26_48 [symbolic = %I.facet.loc26_18.2 (constants.%I.facet.d3d)]
// CHECK:STDOUT:     %.loc26_57: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %as_type.loc26_54: type = facet_access_type %.loc26_48 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_54: type = converted %.loc26_48, %as_type.loc26_54 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc26_66: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %J.facet.loc26_63: %J.type = facet_value %.loc26_54, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:     %.loc26_63: %J.type = converted %.loc26_54, %J.facet.loc26_63 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:     %as_type.loc26_67: type = facet_access_type %.loc26_63 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_67: type = converted %.loc26_63, %as_type.loc26_67 [symbolic = %T.as_type.loc22_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %JJ.ref: %J.assoc_type = name_ref JJ, @J.WithSelf.%assoc0 [concrete = constants.%assoc0.7d8]
// CHECK:STDOUT:     %impl.elem0.loc26_69.1: @F.%.loc26_69 (%.443) = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_69.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc26_69: <bound method> = bound_method %t.ref, %impl.elem0.loc26_69.1
// CHECK:STDOUT:     %J.facet.loc26_73: %J.type = facet_value constants.%T.as_type, (constants.%J.lookup_impl_witness) [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:     %.loc26_73: %J.type = converted constants.%T.as_type, %J.facet.loc26_73 [symbolic = %J.facet.loc26_33.2 (constants.%J.facet.cbc)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_69.1: <specific function> = specific_impl_function %impl.elem0.loc26_69.1, @J.WithSelf.JJ(constants.%J.facet.cbc) [symbolic = %specific_impl_fn.loc26_69.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc26_73: <bound method> = bound_method %t.ref, %specific_impl_fn.loc26_69.1
// CHECK:STDOUT:     %J.WithSelf.JJ.call: init %empty_tuple.type = call %bound_method.loc26_73(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.loc40_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref.loc40_11: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc40_6.3: init %C to %.loc40_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc40_8.1: init %C = converted %.loc40_6.1, %.loc40_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref.loc40_24: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc40_29: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc40_26: %I.type = facet_value %C.ref.loc40_24, (constants.%I.impl_witness) [concrete = constants.%I.facet.947]
// CHECK:STDOUT:   %.loc40_26: %I.type = converted %C.ref.loc40_24, %I.facet.loc40_26 [concrete = constants.%I.facet.947]
// CHECK:STDOUT:   %.loc40_35: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %as_type.loc40_32: type = facet_access_type %.loc40_26 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_32: type = converted %.loc40_26, %as_type.loc40_32 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc40_44: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc40_41: %J.type = facet_value %.loc40_32, (constants.%J.impl_witness) [concrete = constants.%J.facet.a74]
// CHECK:STDOUT:   %.loc40_41: %J.type = converted %.loc40_32, %J.facet.loc40_41 [concrete = constants.%J.facet.a74]
// CHECK:STDOUT:   %.loc40_50: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %as_type.loc40_47: type = facet_access_type %.loc40_41 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_47: type = converted %.loc40_41, %as_type.loc40_47 [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc40_59: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet.loc40_56: %I.type = facet_value %.loc40_47, (constants.%I.impl_witness) [concrete = constants.%I.facet.947]
// CHECK:STDOUT:   %.loc40_56: %I.type = converted %.loc40_47, %I.facet.loc40_56 [concrete = constants.%I.facet.947]
// CHECK:STDOUT:   %.loc40_65: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %as_type.loc40_62: type = facet_access_type %.loc40_56 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_62: type = converted %.loc40_56, %as_type.loc40_62 [concrete = constants.%C]
// CHECK:STDOUT:   %J.ref.loc40_74: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %J.facet.loc40_71: %J.type = facet_value %.loc40_62, (constants.%J.impl_witness) [concrete = constants.%J.facet.a74]
// CHECK:STDOUT:   %.loc40_71: %J.type = converted %.loc40_62, %J.facet.loc40_71 [concrete = constants.%J.facet.a74]
// CHECK:STDOUT:   %as_type.loc40_75: type = facet_access_type %.loc40_71 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_75: type = converted %.loc40_71, %as_type.loc40_75 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40_8.2: ref %C = temporary %.loc40_6.2, %.loc40_8.1 [concrete = constants.%.5d3]
// CHECK:STDOUT:   %JJ.ref: %J.assoc_type = name_ref JJ, @J.WithSelf.%assoc0 [concrete = constants.%assoc0.7d8]
// CHECK:STDOUT:   %impl.elem0: %.dd2 = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%C.as.J.impl.JJ]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc40_8.2, %impl.elem0 [concrete = constants.%C.as.J.impl.JJ.bound]
// CHECK:STDOUT:   %.loc40_8.3: %C = acquire_value %.loc40_8.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.as.J.impl.JJ.call: init %empty_tuple.type = call %bound_method(%.loc40_8.3)
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref.loc46_5: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc46_9.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref.loc46_14: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc46_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc46_9.3: init %C to %.loc46_9.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc46_11.1: init %C = converted %.loc46_9.1, %.loc46_9.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %facet_value.loc46_15.1: %facet_type = facet_value %C.ref.loc46_5, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc46_15.1: %facet_type = converted %C.ref.loc46_5, %facet_value.loc46_15.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc46_15.2: %facet_type = facet_value constants.%C, (constants.%I.impl_witness, constants.%J.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc46_15.2: %facet_type = converted constants.%C, %facet_value.loc46_15.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%facet_value) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc46_11.2: ref %C = temporary %.loc46_9.2, %.loc46_11.1 [concrete = constants.%.5d3]
// CHECK:STDOUT:   %.loc46_11.3: %C = acquire_value %.loc46_11.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc46_11.3)
// CHECK:STDOUT:   %Destroy.Op.call.loc46: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.5d3)
// CHECK:STDOUT:   %Destroy.Op.call.loc40: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.5d3)
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
// CHECK:STDOUT: fn @Destroy.Op.loc46_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc46_11.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.II.type => constants.%I.WithSelf.II.type.8a2
// CHECK:STDOUT:   %I.WithSelf.II => constants.%I.WithSelf.II.09c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.II(constants.%Self.37e) {
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type.loc16_9.1 => constants.%Self.as_type.c84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.653
// CHECK:STDOUT:   %J.WithSelf.JJ.type => constants.%J.WithSelf.JJ.type.e02
// CHECK:STDOUT:   %J.WithSelf.JJ => constants.%J.WithSelf.JJ.1d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.JJ(constants.%Self.653) {
// CHECK:STDOUT:   %Self => constants.%Self.653
// CHECK:STDOUT:   %Self.as_type.loc19_9.1 => constants.%Self.as_type.976
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.506
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc22_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc22_20.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.936
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %I.WithSelf.II.type => constants.%I.WithSelf.II.type.21c
// CHECK:STDOUT:   %I.WithSelf.II => constants.%I.WithSelf.II.a59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %J.WithSelf.JJ.type => constants.%J.WithSelf.JJ.type.f9e
// CHECK:STDOUT:   %J.WithSelf.JJ => constants.%J.WithSelf.JJ.e0c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.cbc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.cbc
// CHECK:STDOUT:   %J.WithSelf.JJ.type => constants.%J.WithSelf.JJ.type.2d7
// CHECK:STDOUT:   %J.WithSelf.JJ => constants.%J.WithSelf.JJ.6d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.JJ(constants.%J.facet.cbc) {
// CHECK:STDOUT:   %Self => constants.%J.facet.cbc
// CHECK:STDOUT:   %Self.as_type.loc19_9.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.936
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.947) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.947
// CHECK:STDOUT:   %I.WithSelf.II.type => constants.%I.WithSelf.II.type.9c8
// CHECK:STDOUT:   %I.WithSelf.II => constants.%I.WithSelf.II.622
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.II(constants.%I.facet.947) {
// CHECK:STDOUT:   %Self => constants.%I.facet.947
// CHECK:STDOUT:   %Self.as_type.loc16_9.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.a74) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.a74
// CHECK:STDOUT:   %J.WithSelf.JJ.type => constants.%J.WithSelf.JJ.type.95f
// CHECK:STDOUT:   %J.WithSelf.JJ => constants.%J.WithSelf.JJ.d09
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.JJ(constants.%J.facet.a74) {
// CHECK:STDOUT:   %Self => constants.%J.facet.a74
// CHECK:STDOUT:   %Self.as_type.loc19_9.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %J.WithSelf.JJ.type => constants.%J.WithSelf.JJ.type.383
// CHECK:STDOUT:   %J.WithSelf.JJ => constants.%J.WithSelf.JJ.87d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %I.WithSelf.II.type => constants.%I.WithSelf.II.type.92c
// CHECK:STDOUT:   %I.WithSelf.II => constants.%I.WithSelf.II.03d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc22_7.1 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc22_20.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f79
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %I.facet.loc26_18.2 => constants.%I.facet.947
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %J.facet.loc26_33.2 => constants.%J.facet.a74
// CHECK:STDOUT:   %J.WithSelf.JJ.type => constants.%J.WithSelf.JJ.type.95f
// CHECK:STDOUT:   %.loc26_69 => constants.%.dd2
// CHECK:STDOUT:   %impl.elem0.loc26_69.2 => constants.%C.as.J.impl.JJ
// CHECK:STDOUT:   %specific_impl_fn.loc26_69.2 => constants.%C.as.J.impl.JJ
// CHECK:STDOUT: }
// CHECK:STDOUT:

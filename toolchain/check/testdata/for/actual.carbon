// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/for/actual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/for/actual.carbon

// --- lib.carbon

library "lib";

class IntRange(N: Core.IntLiteral) {
  fn Make(start: Core.Int(N), end: Core.Int(N)) -> Self {
    return {.start = start, .end = end};
  }

  impl as Core.Iterate where .CursorType = Core.Int(N) and .ElementType = Core.Int(N) {
    fn NewCursor(self) -> Core.Int(N) { return self.start; }
    fn Next(self, ref cursor: Core.Int(N)) -> Core.Optional(Core.Int(N)) {
      var value: Core.Int(N) = cursor;
      if (value < self.end) {
        ++cursor;
        return Core.Optional(Core.Int(N)).Some(value);
      } else {
        return Core.Optional(Core.Int(N)).None();
      }
    }
  }

  private var start: Core.Int(N);
  private var end: Core.Int(N);
}

fn Range(end: i32) -> IntRange(32) {
  return IntRange(32).Make(0, end);
}

// --- trivial.carbon

import library "lib";

fn Read(generic y: Core.IntLiteral) {
  var unused x: IntRange(32) = Range(y);
}


// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %IntRange.type: type = parameterized_class_name_type @IntRange [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.0bb: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.5515d5.1: type = pattern_type %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %start.param_patt.b10: %pattern_type.5515d5.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %start.patt.2ad: %pattern_type.5515d5.1 = wrapper_binding_pattern start, %start.param_patt.b10 [symbolic]
// CHECK:STDOUT:   %end.param_patt.dde: %pattern_type.5515d5.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %end.patt.962: %pattern_type.5515d5.1 = wrapper_binding_pattern end, %end.param_patt.dde [symbolic]
// CHECK:STDOUT:   %.147: Core.Form = init_form %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %pattern_type.711: type = pattern_type %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %return.param_patt.53d: %pattern_type.711 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.195: %pattern_type.711 = return_slot_pattern %return.param_patt.53d, %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.120: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.646: %IntRange.Make.type.120 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.63b: type = fn_type @Optional.Some, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Some.792: %Optional.Some.type.63b = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.None.type.626: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.3ad: %Optional.None.type.626 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.bdd: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.bdd [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.c37: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.451 [concrete]
// CHECK:STDOUT:   %.052: <witness> = impl_self_witness %.Self.frozen.bdd, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.a76: %facet_type.8a3 = impl_witness_access %.052, element1 [symbolic_self]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.f5e: <witness> = lookup_impl_witness %Int.d4bbe0.1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.79f: <witness> = lookup_impl_witness %Int.d4bbe0.1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cbf: %SubobjectDestroy.type = facet_value %Int.d4bbe0.1, (%SubobjectDestroy.lookup_impl_witness.f5e) [symbolic]
// CHECK:STDOUT:   %require_complete.d0a0eb.1: <witness> = require_complete_type %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %.811: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cbf) [symbolic]
// CHECK:STDOUT:   %facet_value.959: %facet_type.8a3 = facet_value %Int.d4bbe0.1, (%SubobjectDestroy.lookup_impl_witness.f5e, %DestroyImpl.lookup_impl_witness.79f) [symbolic]
// CHECK:STDOUT:   %assoc0.0d2: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.941 [concrete]
// CHECK:STDOUT:   %impl.elem0.0c4: %facet_type.d61 = impl_witness_access %.052, element0 [symbolic_self]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %return.param_patt.a51197.1: %pattern_type.5515d5.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %.c504e9.1: Core.Form = init_form %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.923: <witness> = lookup_impl_witness %Int.d4bbe0.1, @Copy [symbolic]
// CHECK:STDOUT:   %.cbd: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %facet_value.f1f: %facet_type.d61 = facet_value %Int.d4bbe0.1, (%SubobjectDestroy.lookup_impl_witness.f5e, %DestroyImpl.lookup_impl_witness.79f, %Copy.lookup_impl_witness.923) [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.327: <witness> = impl_self_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.15e: %facet_type.8a3 = impl_witness_access %.327, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: %facet_type.d61 = impl_witness_access %.327, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.48d: type = facet_type <@Iterate where %impl.elem1.15e = %facet_value.959 and %impl.elem0.53a = %facet_value.f1f> [symbolic]
// CHECK:STDOUT:   %.712: <witness> = impl_self_witness %IntRange.0bb, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.as.Iterate.impl.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %self.param_patt.b7f: %pattern_type.711 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.835f4: %pattern_type.711 = wrapper_binding_pattern self, %self.param_patt.b7f [symbolic]
// CHECK:STDOUT:   %return.patt.740: %pattern_type.5515d5.1 = return_slot_pattern %return.param_patt.a51197.1, %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %cursor.param_patt.a08: %pattern_type.5515d5.1 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %cursor.patt.019: %pattern_type.5515d5.1 = wrapper_binding_pattern cursor, %cursor.param_patt.a08 [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.f1f: <witness> = lookup_impl_witness %Int.d4bbe0.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %.19b: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value.f1f) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.b7c: %OptionalStorage.type = facet_value %Int.d4bbe0.1, (%OptionalStorage.lookup_impl_witness.f1f) [symbolic]
// CHECK:STDOUT:   %Optional.b5c: type = class_type @Optional, @Optional(%OptionalStorage.facet.b7c) [symbolic]
// CHECK:STDOUT:   %.940: Core.Form = init_form %Optional.b5c [symbolic]
// CHECK:STDOUT:   %pattern_type.47f: type = pattern_type %Optional.b5c [symbolic]
// CHECK:STDOUT:   %return.param_patt.972: %pattern_type.47f = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.733: %pattern_type.47f = return_slot_pattern %return.param_patt.972, %Optional.b5c [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IntRange.elem.945: type = unbound_element_type %IntRange.0bb, %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.fc0: type = struct_type {.start: %Int.d4bbe0.1, .end: %Int.d4bbe0.1} [symbolic]
// CHECK:STDOUT:   %complete_type.ae3: <witness> = complete_type_witness %struct_type.start.end.fc0 [symbolic]
// CHECK:STDOUT:   %require_complete.cca: <witness> = require_complete_type %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %Copy.facet.ad7: %Copy.type = facet_value %Int.d4bbe0.1, (%Copy.lookup_impl_witness.923) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a18: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ad7) [symbolic]
// CHECK:STDOUT:   %.ade: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a18, %Copy.facet.ad7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.6c8: %.ade = impl_witness_access %Copy.lookup_impl_witness.923, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d42: <specific function> = specific_impl_function %impl.elem0.6c8, @Copy.WithSelf.Op(%Copy.facet.ad7) [symbolic]
// CHECK:STDOUT:   %Optional.None.type.145: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.b7c) [symbolic]
// CHECK:STDOUT:   %Optional.None.125: %Optional.None.type.145 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.36c: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.b7c) [symbolic]
// CHECK:STDOUT:   %Optional.Some.a4c: %Optional.Some.type.36c = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.710: <witness> = require_complete_type %Optional.b5c [symbolic]
// CHECK:STDOUT:   %value.patt.f0c: %pattern_type.5515d5.1 = ref_binding_pattern value [symbolic]
// CHECK:STDOUT:   %value.var_patt: %pattern_type.5515d5.1 = var_pattern %value.patt.f0c [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ad8: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.ad8 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507: type = facet_type <@OrderedWith, @OrderedWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.635: %OrderedWith.type.507 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.635) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.51e: %OrderedWith.WithSelf.Less.type.104 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.a49: type = facet_type <@OrderedWith, @OrderedWith(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e39: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.d4bbe0.1) [symbolic]
// CHECK:STDOUT:   %assoc0.5ea: %OrderedWith.assoc_type.e39 = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic]
// CHECK:STDOUT:   %require_complete.9ab: <witness> = require_complete_type %OrderedWith.type.a49 [symbolic]
// CHECK:STDOUT:   %assoc0.d91: %OrderedWith.assoc_type.06a = assoc_entity element0, imports.%Core.import_ref.97f [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.a29: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.97c(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.245: %Int.as.OrderedWith.impl.Less.type.a29 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.2aa: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.ef8, @Int.as.OrderedWith.impl.97c(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.6a9: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.97c(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.546: %Int.as.OrderedWith.impl.Less.type.6a9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.ad9: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.97c(%N, %N) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet.e79: %OrderedWith.type.a49 = facet_value %Int.d4bbe0.1, (%OrderedWith.impl_witness.2aa) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.160: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.d4bbe0.1, %OrderedWith.facet.e79) [symbolic]
// CHECK:STDOUT:   %.31f: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.160, %OrderedWith.facet.e79 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.4b7: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.546, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.d4bbe0.1, @Inc [symbolic]
// CHECK:STDOUT:   %.ae8: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.d4bbe0.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.7c7: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %.bfa: type = fn_type_with_self_type %Inc.WithSelf.Op.type.7c7, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.fc5: %.bfa = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b3e: <specific function> = specific_impl_function %impl.elem0.fc5, @Inc.WithSelf.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.a4c, @Optional.Some(%OptionalStorage.facet.b7c) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6d7: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.959) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.125, @Optional.None(%OptionalStorage.facet.b7c) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %end.param_patt.4c3: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %end.patt.9b2: %pattern_type.d94 = wrapper_binding_pattern end, %end.param_patt.4c3 [concrete]
// CHECK:STDOUT:   %IntRange.773: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %.bef: Core.Form = init_form %IntRange.773 [concrete]
// CHECK:STDOUT:   %pattern_type.9c3: type = pattern_type %IntRange.773 [concrete]
// CHECK:STDOUT:   %return.param_patt.82a: %pattern_type.9c3 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.9e6: %pattern_type.9c3 = return_slot_pattern %return.param_patt.82a, %IntRange.773 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.3ab: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.582: %IntRange.Make.type.3ab = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.elem.85f: type = unbound_element_type %IntRange.773, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.7e8: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a5c: <witness> = complete_type_witness %struct_type.start.end.7e8 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %start.param_patt.2c9: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %start.patt.57a: %pattern_type.d94 = wrapper_binding_pattern start, %start.param_patt.2c9 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.582, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %.72f: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Copy = %Core.Copy.e5d
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Iterate: type = import_ref Core//prelude/iterate, Iterate, loaded [concrete = constants.%Iterate.type]
// CHECK:STDOUT:   %Core.import_ref.2fe: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.0d2]
// CHECK:STDOUT:   %Core.import_ref.a26: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1.c37]
// CHECK:STDOUT:   %Core.import_ref.987: @Optional.%Optional.None.type (%Optional.None.type.626) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.3ad)]
// CHECK:STDOUT:   %Core.import_ref.300: @Optional.%Optional.Some.type (%Optional.Some.type.63b) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.792)]
// CHECK:STDOUT:   %Core.import_ref.451: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: %facet_type.8a3 = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.941: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %facet_type.d61 = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Copy.e5d: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.ad8 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8cd: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.d91)]
// CHECK:STDOUT:   %Core.import_ref.b46: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.104) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.51e)]
// CHECK:STDOUT:   %Core.import_ref.97f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.03e: @Int.as.OrderedWith.impl.97c.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.a29) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.97c.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.245)]
// CHECK:STDOUT:   %Core.import_ref.27a = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3c0 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ae2 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.ef8 = impl_witness_table (%Core.import_ref.03e, %Core.import_ref.27a, %Core.import_ref.3c0, %Core.import_ref.ae2), @Int.as.OrderedWith.impl.97c [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .IntRange = %IntRange.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %IntRange.decl: %IntRange.type = class_decl @IntRange [concrete = constants.%IntRange.generic] {
// CHECK:STDOUT:     %N.patt.loc4_17.1: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_17.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_17.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: %Range.type = fn_decl @Range [concrete = constants.%Range] {
// CHECK:STDOUT:     %end.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%end.param_patt.4c3]
// CHECK:STDOUT:     %end.patt: %pattern_type.d94 = wrapper_binding_pattern end, %end.param_patt [concrete = constants.%end.patt.9b2]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9c3 = out_param_pattern [concrete = constants.%return.param_patt.82a]
// CHECK:STDOUT:     %return.patt: %pattern_type.9c3 = return_slot_pattern %return.param_patt, %IntRange.loc26 [concrete = constants.%return.patt.9e6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc26: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc26: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.773]
// CHECK:STDOUT:     %.loc26_34.2: Core.Form = init_form %IntRange.loc26 [concrete = constants.%.bef]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %end: %i32 = wrapper_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.773 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.773 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.f5e)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Int.loc9_54.1, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.cbf)]
// CHECK:STDOUT:   %.loc9_54.1: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc9_54.1 (constants.%.811)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.79f)]
// CHECK:STDOUT:   %facet_value.loc9_54.1: %facet_type.8a3 = facet_value %Int.loc9_54.1, (%SubobjectDestroy.lookup_impl_witness, %DestroyImpl.lookup_impl_witness) [symbolic = %facet_value.loc9_54.1 (constants.%facet_value.959)]
// CHECK:STDOUT:   %.loc9_85.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc9_85.1 (constants.%.cbd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.923)]
// CHECK:STDOUT:   %facet_value.loc9_85.1: %facet_type.d61 = facet_value %Int.loc9_54.1, (%SubobjectDestroy.lookup_impl_witness, %DestroyImpl.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value.f1f)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.15e = %facet_value.loc9_54.1 and constants.%impl.elem0.53a = %facet_value.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type.48d)]
// CHECK:STDOUT:   %.loc9_87: <witness> = impl_self_witness %IntRange, @Iterate [symbolic = %.loc9_87 (constants.%.712)]
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type (constants.%IntRange.as.Iterate.impl.NewCursor.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type (constants.%IntRange.as.Iterate.impl.Next.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Iterate.ref {
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = fn_decl @IntRange.as.Iterate.impl.NewCursor [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)] {
// CHECK:STDOUT:       %self.param_patt.loc10_18.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.711) = value_param_pattern [symbolic = %self.param_patt.loc10_18.2 (constants.%self.param_patt.b7f)]
// CHECK:STDOUT:       %self.patt.loc10_18.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.711) = wrapper_binding_pattern self, %self.param_patt.loc10_18.1 [symbolic = %self.patt.loc10_18.2 (constants.%self.patt.835f4)]
// CHECK:STDOUT:       %return.param_patt.loc10_37.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.5515d5.1) = out_param_pattern [symbolic = %return.param_patt.loc10_37.2 (constants.%return.param_patt.a51197.1)]
// CHECK:STDOUT:       %return.patt.loc10_24.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.5515d5.1) = return_slot_pattern %return.param_patt.loc10_37.1, %Int.loc10_37.2 [symbolic = %return.patt.loc10_24.2 (constants.%return.patt.740)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %.loc10_36: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %.loc10_36 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_37.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_37.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       %.loc10_37.2: Core.Form = init_form %Int.loc10_37.2 [symbolic = %.loc10_37.1 (constants.%.c504e9.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.0bb) = value_param call_param0
// CHECK:STDOUT:       %.loc10_18.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.0bb)] {
// CHECK:STDOUT:         %.loc10_18.2: type = specific_constant constants.%IntRange.0bb, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_18.2 [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.0bb) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.d4bbe0.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.d4bbe0.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.param_patt.loc11_13.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.711) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.b7f)]
// CHECK:STDOUT:       %self.patt.loc11_13.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.711) = wrapper_binding_pattern self, %self.param_patt.loc11_13.1 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.835f4)]
// CHECK:STDOUT:       %cursor.param_patt.loc11_29.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = ref_param_pattern [symbolic = %cursor.param_patt.loc11_29.2 (constants.%cursor.param_patt.a08)]
// CHECK:STDOUT:       %cursor.patt.loc11_29.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = wrapper_binding_pattern cursor, %cursor.param_patt.loc11_29.1 [symbolic = %cursor.patt.loc11_29.2 (constants.%cursor.patt.019)]
// CHECK:STDOUT:       %return.param_patt.loc11_72.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.47f) = out_param_pattern [symbolic = %return.param_patt.loc11_72.2 (constants.%return.param_patt.972)]
// CHECK:STDOUT:       %return.patt.loc11_44.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.47f) = return_slot_pattern %return.param_patt.loc11_72.1, %Optional.loc11_72.2 [symbolic = %return.patt.loc11_44.2 (constants.%return.patt.733)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_47: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_61: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_65: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %.loc11_70: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %N.ref.loc11_70: Core.IntLiteral = name_ref N, %.loc11_70 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_71: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_72.2: %OptionalStorage.type = facet_value %Int.loc11_71, (constants.%OptionalStorage.lookup_impl_witness.f1f) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:       %.loc11_72.5: %OptionalStorage.type = converted %Int.loc11_71, %OptionalStorage.facet.loc11_72.2 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:       %Optional.loc11_72.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.loc11_72.1 (constants.%Optional.b5c)]
// CHECK:STDOUT:       %.loc11_72.6: Core.Form = init_form %Optional.loc11_72.2 [symbolic = %.loc11_72.4 (constants.%.940)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.0bb) = value_param call_param0
// CHECK:STDOUT:       %.loc11_13.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.0bb)] {
// CHECK:STDOUT:         %.loc11_13.2: type = specific_constant constants.%IntRange.0bb, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_13.2 [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.0bb) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = ref_param call_param1
// CHECK:STDOUT:       %.loc11_41: type = splice_block %Int.loc11_41.2 [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)] {
// CHECK:STDOUT:         %Core.ref.loc11_31: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_35: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %.loc11_40: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %N.ref.loc11_40: Core.IntLiteral = name_ref N, %.loc11_40 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_41.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = wrapper_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.1, %impl_witness_assoc_constant.loc9_87.2, %IntRange.as.Iterate.impl.NewCursor.decl, %IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness.loc9_87.1: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: %facet_type.d61 = impl_witness_assoc_constant constants.%facet_value.f1f [symbolic = %facet_value.loc9_85.1 (constants.%facet_value.f1f)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %facet_type.8a3 = impl_witness_assoc_constant constants.%facet_value.959 [symbolic = %facet_value.loc9_54.1 (constants.%facet_value.959)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .NewCursor = %IntRange.as.Iterate.impl.NewCursor.decl
// CHECK:STDOUT:     .Next = %IntRange.as.Iterate.impl.Next.decl
// CHECK:STDOUT:     extend %Iterate.ref
// CHECK:STDOUT:     witness = %Iterate.impl_witness.loc9_87.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc4_17.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_17.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc4_17.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_17.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.120)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.120) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.646)]
// CHECK:STDOUT:   %Int.loc22_32.2: type = class_type @Int, @Int(%N.loc4_17.1) [symbolic = %Int.loc22_32.2 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc22_32.2 [symbolic = %require_complete (constants.%require_complete.d0a0eb.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_17.1) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc22_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.945)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int.loc22_32.2 (%Int.d4bbe0.1), .end: @IntRange.%Int.loc22_32.2 (%Int.d4bbe0.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fc0)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.ae3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.120) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.646)] {
// CHECK:STDOUT:       %start.param_patt.loc5_16.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = value_param_pattern [symbolic = %start.param_patt.loc5_16.2 (constants.%start.param_patt.b10)]
// CHECK:STDOUT:       %start.patt.loc5_16.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = wrapper_binding_pattern start, %start.param_patt.loc5_16.1 [symbolic = %start.patt.loc5_16.2 (constants.%start.patt.2ad)]
// CHECK:STDOUT:       %end.param_patt.loc5_34.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = value_param_pattern [symbolic = %end.param_patt.loc5_34.2 (constants.%end.param_patt.dde)]
// CHECK:STDOUT:       %end.patt.loc5_34.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = wrapper_binding_pattern end, %end.param_patt.loc5_34.1 [symbolic = %end.patt.loc5_34.2 (constants.%end.patt.962)]
// CHECK:STDOUT:       %return.param_patt.loc5_52.1: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.711) = out_param_pattern [symbolic = %return.param_patt.loc5_52.2 (constants.%return.param_patt.53d)]
// CHECK:STDOUT:       %return.patt.loc5_49.1: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.711) = return_slot_pattern %return.param_patt.loc5_52.1, %Self.ref [symbolic = %return.patt.loc5_49.2 (constants.%return.patt.195)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52.2: type = specific_constant constants.%IntRange.0bb, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52.2 [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:       %.loc5_52.3: Core.Form = init_form %Self.ref [symbolic = %.loc5_52.1 (constants.%.147)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.d4bbe0.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %.loc5_27: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, %.loc5_27 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = wrapper_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.d4bbe0.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %.loc5_45: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, %.loc5_45 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = wrapper_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.0bb) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.0bb) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.0bb [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self.frozen: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.bdd]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Iterate.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.bdd]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.a26 [concrete = constants.%assoc1.c37]
// CHECK:STDOUT:       %impl.elem1.loc9_30.1: %facet_type.8a3 = impl_witness_access constants.%.052, element1 [symbolic_self = constants.%impl.elem1.a76]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %.loc9_53: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, %.loc9_53 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       %facet_value.loc9_54.2: %facet_type.8a3 = facet_value %Int.loc9_54.2, (constants.%SubobjectDestroy.lookup_impl_witness.f5e, constants.%DestroyImpl.lookup_impl_witness.79f) [symbolic = %facet_value.loc9_54.1 (constants.%facet_value.959)]
// CHECK:STDOUT:       %.loc9_54.2: %facet_type.8a3 = converted %Int.loc9_54.2, %facet_value.loc9_54.2 [symbolic = %facet_value.loc9_54.1 (constants.%facet_value.959)]
// CHECK:STDOUT:       %rewrite.loc9_42.1 = requirement_rewrite %impl.elem1.loc9_30.1, %.loc9_54.2 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.bdd]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.2fe [concrete = constants.%assoc0.0d2]
// CHECK:STDOUT:       %impl.elem0.loc9_60.1: %facet_type.d61 = impl_witness_access constants.%.052, element0 [symbolic_self = constants.%impl.elem0.0c4]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %.loc9_84: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, %.loc9_84 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:       %facet_value.loc9_85.2: %facet_type.d61 = facet_value %Int.loc9_85, (constants.%SubobjectDestroy.lookup_impl_witness.f5e, constants.%DestroyImpl.lookup_impl_witness.79f, constants.%Copy.lookup_impl_witness.923) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value.f1f)]
// CHECK:STDOUT:       %.loc9_85.2: %facet_type.d61 = converted %Int.loc9_85, %facet_value.loc9_85.2 [symbolic = %facet_value.loc9_85.1 (constants.%facet_value.f1f)]
// CHECK:STDOUT:       %rewrite.loc9_73.1 = requirement_rewrite %impl.elem0.loc9_60.1, %.loc9_85.2 [concrete]
// CHECK:STDOUT:       %impl.elem1.loc9_30.2: %facet_type.8a3 = impl_witness_access constants.%.327, element1 [symbolic_self = constants.%impl.elem1.15e]
// CHECK:STDOUT:       %rewrite.loc9_42.2 = requirement_rewrite %impl.elem1.loc9_30.2, %.loc9_54.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc9_60.2: %facet_type.d61 = impl_witness_access constants.%.327, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:       %rewrite.loc9_73.2 = requirement_rewrite %impl.elem0.loc9_60.2, %.loc9_85.2 [concrete]
// CHECK:STDOUT:       %.loc9_24: type = where_expr [symbolic = %Iterate_where.type (constants.%Iterate_where.type.48d)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Iterate.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc9_42.2 = requirement_rewrite %impl.elem1.loc9_30.2, %.loc9_54.2 [concrete]
// CHECK:STDOUT:         %rewrite.loc9_73.2 = requirement_rewrite %impl.elem0.loc9_60.2, %.loc9_85.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: <witness> = impl_self_witness @IntRange.as.Iterate.impl.%Self.ref, @Iterate [symbolic = @IntRange.as.Iterate.impl.%.loc9_87 (constants.%.712)]
// CHECK:STDOUT:     %field_decl.loc22: @IntRange.%IntRange.elem (%IntRange.elem.945) = field_decl start, element0, %Int.loc22_32.1 in [concrete] {
// CHECK:STDOUT:       %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc22: Core.IntLiteral = name_ref N, %N.loc4_17.2 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc22_32.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %field_decl.loc23: @IntRange.%IntRange.elem (%IntRange.elem.945) = field_decl end, element1, %Int.loc23 in [concrete] {
// CHECK:STDOUT:       %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc23: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc23: Core.IntLiteral = name_ref N, %N.loc4_17.2 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc23: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.fc0 [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.ae3)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.0bb
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %field_decl.loc22
// CHECK:STDOUT:     .end [private] = %field_decl.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.5515d5.1)]
// CHECK:STDOUT:   %start.param_patt.loc5_16.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = value_param_pattern [symbolic = %start.param_patt.loc5_16.2 (constants.%start.param_patt.b10)]
// CHECK:STDOUT:   %start.patt.loc5_16.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = wrapper_binding_pattern start, %start.param_patt.loc5_16.2 [symbolic = %start.patt.loc5_16.2 (constants.%start.patt.2ad)]
// CHECK:STDOUT:   %end.param_patt.loc5_34.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = value_param_pattern [symbolic = %end.param_patt.loc5_34.2 (constants.%end.param_patt.dde)]
// CHECK:STDOUT:   %end.patt.loc5_34.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.5515d5.1) = wrapper_binding_pattern end, %end.param_patt.loc5_34.2 [symbolic = %end.patt.loc5_34.2 (constants.%end.patt.962)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %.loc5_52.1: Core.Form = init_form %IntRange [symbolic = %.loc5_52.1 (constants.%.147)]
// CHECK:STDOUT:   %pattern_type.loc5_52: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_52 (constants.%pattern_type.711)]
// CHECK:STDOUT:   %return.param_patt.loc5_52.2: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.711) = out_param_pattern [symbolic = %return.param_patt.loc5_52.2 (constants.%return.param_patt.53d)]
// CHECK:STDOUT:   %return.patt.loc5_49.2: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.711) = return_slot_pattern %return.param_patt.loc5_52.2, %IntRange [symbolic = %return.patt.loc5_49.2 (constants.%return.patt.195)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.d0a0eb.1)]
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.cca)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fc0)]
// CHECK:STDOUT:   %.loc6_22.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc6_22.3 (constants.%.cbd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.923)]
// CHECK:STDOUT:   %Copy.facet.loc6_22.3: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc6_22.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a18)]
// CHECK:STDOUT:   %.loc6_22.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc6_22.3 [symbolic = %.loc6_22.4 (constants.%.ade)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.4 (%.ade) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.WithSelf.Op(%Copy.facet.loc6_22.3) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1)) -> out %return.param: @IntRange.Make.%IntRange (%IntRange.0bb) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.fc0) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.4 (%.ade) = impl_witness_access constants.%Copy.lookup_impl_witness.923, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %Copy.facet.loc6_22.1: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc6_22.1: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc6_22.1 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %Copy.facet.loc6_22.2: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc6_22.2: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc6_22.2 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.WithSelf.Op(constants.%Copy.facet.ad7) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_22: init @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = class_element_access %return.param, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) to %.loc6_39.2 = in_place_init %Copy.WithSelf.Op.call.loc6_22
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.4 (%.ade) = impl_witness_access constants.%Copy.lookup_impl_witness.923, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %Copy.facet.loc6_36.1: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc6_36.1: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc6_36.1 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %Copy.facet.loc6_36.2: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc6_36.2: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc6_36.2 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.WithSelf.Op(constants.%Copy.facet.ad7) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) = class_element_access %return.param, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.d4bbe0.1) to %.loc6_39.4 = in_place_init %Copy.WithSelf.Op.call.loc6_36
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.0bb) to %return.param = class_init (%.loc6_39.3, %.loc6_39.5)
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.0bb) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.711)]
// CHECK:STDOUT:   %self.param_patt.loc10_18.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.711) = value_param_pattern [symbolic = %self.param_patt.loc10_18.2 (constants.%self.param_patt.b7f)]
// CHECK:STDOUT:   %self.patt.loc10_18.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.711) = wrapper_binding_pattern self, %self.param_patt.loc10_18.2 [symbolic = %self.patt.loc10_18.2 (constants.%self.patt.835f4)]
// CHECK:STDOUT:   %Int.loc10_37.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_37.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %.loc10_37.1: Core.Form = init_form %Int.loc10_37.1 [symbolic = %.loc10_37.1 (constants.%.c504e9.1)]
// CHECK:STDOUT:   %pattern_type.loc10_37: type = pattern_type %Int.loc10_37.1 [symbolic = %pattern_type.loc10_37 (constants.%pattern_type.5515d5.1)]
// CHECK:STDOUT:   %return.param_patt.loc10_37.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.5515d5.1) = out_param_pattern [symbolic = %return.param_patt.loc10_37.2 (constants.%return.param_patt.a51197.1)]
// CHECK:STDOUT:   %return.patt.loc10_24.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.5515d5.1) = return_slot_pattern %return.param_patt.loc10_37.2, %Int.loc10_37.1 [symbolic = %return.patt.loc10_24.2 (constants.%return.patt.740)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_18: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_18 (constants.%require_complete.cca)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_37.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.945)]
// CHECK:STDOUT:   %require_complete.loc10_52: <witness> = require_complete_type %Int.loc10_37.1 [symbolic = %require_complete.loc10_52 (constants.%require_complete.d0a0eb.1)]
// CHECK:STDOUT:   %.loc10_52.5: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc10_52.5 (constants.%.cbd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc10_37.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.923)]
// CHECK:STDOUT:   %Copy.facet.loc10_52.3: %Copy.type = facet_value %Int.loc10_37.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_52.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a18)]
// CHECK:STDOUT:   %.loc10_52.6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_52.3 [symbolic = %.loc10_52.6 (constants.%.ade)]
// CHECK:STDOUT:   %impl.elem0.loc10_52.2: @IntRange.as.Iterate.impl.NewCursor.%.loc10_52.6 (%.ade) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_52.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_52.2: <specific function> = specific_impl_function %impl.elem0.loc10_52.2, @Copy.WithSelf.Op(%Copy.facet.loc10_52.3) [symbolic = %specific_impl_fn.loc10_52.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.0bb)) -> out %return.param: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.d4bbe0.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.0bb) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.945) = name_ref start, @IntRange.%field_decl.loc22 [concrete = @IntRange.%field_decl.loc22]
// CHECK:STDOUT:     %.loc10_52.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.d4bbe0.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_52.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.d4bbe0.1) = acquire_value %.loc10_52.1
// CHECK:STDOUT:     %impl.elem0.loc10_52.1: @IntRange.as.Iterate.impl.NewCursor.%.loc10_52.6 (%.ade) = impl_witness_access constants.%Copy.lookup_impl_witness.923, element0 [symbolic = %impl.elem0.loc10_52.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:     %bound_method.loc10_52.1: <bound method> = bound_method %.loc10_52.2, %impl.elem0.loc10_52.1
// CHECK:STDOUT:     %Copy.facet.loc10_52.1: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc10_52.3: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc10_52.1 [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %Copy.facet.loc10_52.2: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc10_52.4: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc10_52.2 [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_52.1: <specific function> = specific_impl_function %impl.elem0.loc10_52.1, @Copy.WithSelf.Op(constants.%Copy.facet.ad7) [symbolic = %specific_impl_fn.loc10_52.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:     %bound_method.loc10_52.2: <bound method> = bound_method %.loc10_52.2, %specific_impl_fn.loc10_52.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.d4bbe0.1) = call %bound_method.loc10_52.2(%.loc10_52.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.711)]
// CHECK:STDOUT:   %self.param_patt.loc11_13.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.711) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.b7f)]
// CHECK:STDOUT:   %self.patt.loc11_13.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.711) = wrapper_binding_pattern self, %self.param_patt.loc11_13.2 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.835f4)]
// CHECK:STDOUT:   %Int.loc11_41.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %pattern_type.loc11_29: type = pattern_type %Int.loc11_41.1 [symbolic = %pattern_type.loc11_29 (constants.%pattern_type.5515d5.1)]
// CHECK:STDOUT:   %cursor.param_patt.loc11_29.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = ref_param_pattern [symbolic = %cursor.param_patt.loc11_29.2 (constants.%cursor.param_patt.a08)]
// CHECK:STDOUT:   %cursor.patt.loc11_29.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = wrapper_binding_pattern cursor, %cursor.param_patt.loc11_29.2 [symbolic = %cursor.patt.loc11_29.2 (constants.%cursor.patt.019)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.f5e)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.79f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.923)]
// CHECK:STDOUT:   %facet_value.loc11: %facet_type.d61 = facet_value %Int.loc11_41.1, (%SubobjectDestroy.lookup_impl_witness, %DestroyImpl.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value.loc11 (constants.%facet_value.f1f)]
// CHECK:STDOUT:   %.loc11_72.3: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value.loc11) [symbolic = %.loc11_72.3 (constants.%.19b)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.f1f)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_72.1: %OptionalStorage.type = facet_value %Int.loc11_41.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:   %Optional.loc11_72.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.loc11_72.1 (constants.%Optional.b5c)]
// CHECK:STDOUT:   %.loc11_72.4: Core.Form = init_form %Optional.loc11_72.1 [symbolic = %.loc11_72.4 (constants.%.940)]
// CHECK:STDOUT:   %pattern_type.loc11_72: type = pattern_type %Optional.loc11_72.1 [symbolic = %pattern_type.loc11_72 (constants.%pattern_type.47f)]
// CHECK:STDOUT:   %return.param_patt.loc11_72.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.47f) = out_param_pattern [symbolic = %return.param_patt.loc11_72.2 (constants.%return.param_patt.972)]
// CHECK:STDOUT:   %return.patt.loc11_44.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.47f) = return_slot_pattern %return.param_patt.loc11_72.2, %Optional.loc11_72.1 [symbolic = %return.patt.loc11_44.2 (constants.%return.patt.733)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_13: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_13 (constants.%require_complete.cca)]
// CHECK:STDOUT:   %require_complete.loc11_29: <witness> = require_complete_type %Int.loc11_41.1 [symbolic = %require_complete.loc11_29 (constants.%require_complete.d0a0eb.1)]
// CHECK:STDOUT:   %require_complete.loc11_44: <witness> = require_complete_type %Optional.loc11_72.1 [symbolic = %require_complete.loc11_44 (constants.%require_complete.710)]
// CHECK:STDOUT:   %value.patt.loc12_16.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = ref_binding_pattern value [symbolic = %value.patt.loc12_16.2 (constants.%value.patt.f0c)]
// CHECK:STDOUT:   %value.var_patt.loc12_7.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = var_pattern %value.patt.loc12_16.2 [symbolic = %value.var_patt.loc12_7.2 (constants.%value.var_patt)]
// CHECK:STDOUT:   %.loc12_32.4: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc12_32.4 (constants.%.cbd)]
// CHECK:STDOUT:   %Copy.facet.loc12_32.3: %Copy.type = facet_value %Int.loc11_41.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc12_32.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a18)]
// CHECK:STDOUT:   %.loc12_32.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc12_32.3 [symbolic = %.loc12_32.5 (constants.%.ade)]
// CHECK:STDOUT:   %impl.elem0.loc12_32.2: @IntRange.as.Iterate.impl.Next.%.loc12_32.5 (%.ade) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_32.2: <specific function> = specific_impl_function %impl.elem0.loc12_32.2, @Copy.WithSelf.Op(%Copy.facet.loc12_32.3) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_41.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.945)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_41.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.a49)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.9ab)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_41.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.e39)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e39) = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic = %assoc0 (constants.%assoc0.5ea)]
// CHECK:STDOUT:   %.loc13_17.2: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.97c(%N, %N) [symbolic = %.loc13_17.2 (constants.%.ad9)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.ef8, @Int.as.OrderedWith.impl.97c(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.2aa)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.a49) = facet_value %Int.loc11_41.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet.e79)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.loc11_41.1, %OrderedWith.facet) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.160)]
// CHECK:STDOUT:   %.loc13_17.3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.3 (constants.%.31f)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.97c(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.6a9)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.6a9) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.546)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.4b7)]
// CHECK:STDOUT:   %.loc14_9.3: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic = %.loc14_9.3 (constants.%.ae8)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet.loc14_9.3: %Inc.type = facet_value %Int.loc11_41.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.loc14_9.3) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.7c7)]
// CHECK:STDOUT:   %.loc14_9.4: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %Inc.facet.loc14_9.3 [symbolic = %.loc14_9.4 (constants.%.bfa)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.4 (%.bfa) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.fc5)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.WithSelf.Op(%Inc.facet.loc14_9.3) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.b3e)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.36c)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.36c) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.a4c)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Int.loc11_41.1, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.cbf)]
// CHECK:STDOUT:   %.loc12_7.5: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc12_7.5 (constants.%.811)]
// CHECK:STDOUT:   %facet_value.loc12_7.5: %facet_type.8a3 = facet_value %Int.loc11_41.1, (%SubobjectDestroy.lookup_impl_witness, %DestroyImpl.lookup_impl_witness) [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(%facet_value.loc12_7.5) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.6d7)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.145)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.145) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.125)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.0bb), %cursor.param: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1)) -> out %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = var_storage %value.var_patt.loc12_7.1
// CHECK:STDOUT:     %cursor.ref.loc12: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = acquire_value %cursor.ref.loc12
// CHECK:STDOUT:     %impl.elem0.loc12_32.1: @IntRange.as.Iterate.impl.Next.%.loc12_32.5 (%.ade) = impl_witness_access constants.%Copy.lookup_impl_witness.923, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:     %bound_method.loc12_32.1: <bound method> = bound_method %.loc12_32.1, %impl.elem0.loc12_32.1
// CHECK:STDOUT:     %Copy.facet.loc12_32.1: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc12_32.2: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc12_32.1 [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %Copy.facet.loc12_32.2: %Copy.type = facet_value constants.%Int.d4bbe0.1, (constants.%Copy.lookup_impl_witness.923) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %.loc12_32.3: %Copy.type = converted constants.%Int.d4bbe0.1, %Copy.facet.loc12_32.2 [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.ad7)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_32.1: <specific function> = specific_impl_function %impl.elem0.loc12_32.1, @Copy.WithSelf.Op(constants.%Copy.facet.ad7) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:     %bound_method.loc12_32.2: <bound method> = bound_method %.loc12_32.1, %specific_impl_fn.loc12_32.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = call %bound_method.loc12_32.2(%.loc12_32.1)
// CHECK:STDOUT:     assign %value.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %.loc12_27: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, %.loc12_27 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = wrapper_binding value, %value.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt.loc12_16.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = ref_binding_pattern value [symbolic = %value.patt.loc12_16.2 (constants.%value.patt.f0c)]
// CHECK:STDOUT:       %value.var_patt.loc12_7.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.5515d5.1) = var_pattern %value.patt.loc12_16.1 [symbolic = %value.var_patt.loc12_7.2 (constants.%value.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.0bb) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.945) = name_ref end, @IntRange.%field_decl.loc23 [concrete = @IntRange.%field_decl.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = acquire_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.d4bbe0.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.a49)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e39) = specific_constant imports.%Core.import_ref.8cd, @OrderedWith.WithSelf(constants.%Int.d4bbe0.1, constants.%Self.635) [symbolic = %assoc0 (constants.%assoc0.5ea)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e39) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.5ea)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.3 (%.31f) = impl_witness_access constants.%OrderedWith.impl_witness.2aa, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.546)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.4b7)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = acquire_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = name_ref cursor, %cursor
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.4 (%.bfa) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.fc5)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %cursor.ref.loc14, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %Inc.facet.loc14_9.1: %Inc.type = facet_value constants.%Int.d4bbe0.1, (constants.%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %.loc14_9.1: %Inc.type = converted constants.%Int.d4bbe0.1, %Inc.facet.loc14_9.1 [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %Inc.facet.loc14_9.2: %Inc.type = facet_value constants.%Int.d4bbe0.1, (constants.%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %.loc14_9.2: %Inc.type = converted constants.%Int.d4bbe0.1, %Inc.facet.loc14_9.2 [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.WithSelf.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.b3e)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %cursor.ref.loc14, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %Inc.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_9.2(%cursor.ref.loc14)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %.loc15_39: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, %.loc15_39 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_41: %OptionalStorage.type = facet_value %Int.loc15, (constants.%OptionalStorage.lookup_impl_witness.f1f) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %.loc15_41: %OptionalStorage.type = converted %Int.loc15, %OptionalStorage.facet.loc15_41 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.loc11_72.1 (constants.%Optional.b5c)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.36c) = specific_constant imports.%Core.import_ref.300, @Optional(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.Some (constants.%Optional.Some.a4c)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.36c) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.a4c)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.1: %OptionalStorage.type = facet_value constants.%Int.d4bbe0.1, (constants.%OptionalStorage.lookup_impl_witness.f1f) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %.loc15_53.1: %OptionalStorage.type = converted constants.%Int.d4bbe0.1, %OptionalStorage.facet.loc15_53.1 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.2: %OptionalStorage.type = facet_value constants.%Int.d4bbe0.1, (constants.%OptionalStorage.lookup_impl_witness.f1f) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %.loc15_53.2: %OptionalStorage.type = converted constants.%Int.d4bbe0.1, %OptionalStorage.facet.loc15_53.2 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_72.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.d4bbe0.1) = acquire_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) to %.loc11_72.1 = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48)
// CHECK:STDOUT:     %facet_value.loc12_7.1: %facet_type.8a3 = facet_value constants.%Int.d4bbe0.1, (constants.%SubobjectDestroy.lookup_impl_witness.f5e, constants.%DestroyImpl.lookup_impl_witness.79f) [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %.loc12_7.1: %facet_type.8a3 = converted constants.%Int.d4bbe0.1, %facet_value.loc12_7.1 [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %facet_value.loc12_7.2: %facet_type.8a3 = facet_value constants.%Int.d4bbe0.1, (constants.%SubobjectDestroy.lookup_impl_witness.f5e, constants.%DestroyImpl.lookup_impl_witness.79f) [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %.loc12_7.2: %facet_type.8a3 = converted constants.%Int.d4bbe0.1, %facet_value.loc12_7.2 [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.959) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.6d7)]
// CHECK:STDOUT:     %SelfDestruct.call.loc12_7.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%value.var)
// CHECK:STDOUT:     return %Optional.Some.call to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %.loc17_39: Core.IntLiteral = specific_constant @IntRange.%N.loc4_17.2, @IntRange(constants.%N) [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, %.loc17_39 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc17: %OptionalStorage.type = facet_value %Int.loc17, (constants.%OptionalStorage.lookup_impl_witness.f1f) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %.loc17_41: %OptionalStorage.type = converted %Int.loc17, %OptionalStorage.facet.loc17 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.b7c)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.loc11_72.1 (constants.%Optional.b5c)]
// CHECK:STDOUT:     %.loc17_42: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.145) = specific_constant imports.%Core.import_ref.987, @Optional(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.None (constants.%Optional.None.125)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.145) = name_ref None, %.loc17_42 [symbolic = %Optional.None (constants.%Optional.None.125)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.b7c) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_72.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) = splice_block %return.param {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.b5c) to %.loc11_72.2 = call %Optional.None.specific_fn.loc17_42.1()
// CHECK:STDOUT:     %facet_value.loc12_7.3: %facet_type.8a3 = facet_value constants.%Int.d4bbe0.1, (constants.%SubobjectDestroy.lookup_impl_witness.f5e, constants.%DestroyImpl.lookup_impl_witness.79f) [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %.loc12_7.3: %facet_type.8a3 = converted constants.%Int.d4bbe0.1, %facet_value.loc12_7.3 [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %facet_value.loc12_7.4: %facet_type.8a3 = facet_value constants.%Int.d4bbe0.1, (constants.%SubobjectDestroy.lookup_impl_witness.f5e, constants.%DestroyImpl.lookup_impl_witness.79f) [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %.loc12_7.4: %facet_type.8a3 = converted constants.%Int.d4bbe0.1, %facet_value.loc12_7.4 [symbolic = %facet_value.loc12_7.5 (constants.%facet_value.959)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.959) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.6d7)]
// CHECK:STDOUT:     %SelfDestruct.call.loc12_7.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%value.var)
// CHECK:STDOUT:     return %Optional.None.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> out %return.param: %IntRange.773 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.773]
// CHECK:STDOUT:   %.loc27_22: %IntRange.Make.type.3ab = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.582]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.3ab = name_ref Make, %.loc27_22 [concrete = constants.%IntRange.Make.582]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc26_34.1: ref %IntRange.773 = splice_block %return.param {}
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.773 to %.loc26_34.1 = call %IntRange.Make.specific_fn(%.loc27_28.2, %end.ref)
// CHECK:STDOUT:   return %IntRange.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc4_17.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.120
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.646
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0a0eb.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.945
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.fc0
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.ae3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.5515d5.1
// CHECK:STDOUT:   %start.param_patt.loc5_16.2 => constants.%start.param_patt.b10
// CHECK:STDOUT:   %start.patt.loc5_16.2 => constants.%start.patt.2ad
// CHECK:STDOUT:   %end.param_patt.loc5_34.2 => constants.%end.param_patt.dde
// CHECK:STDOUT:   %end.patt.loc5_34.2 => constants.%end.patt.962
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.147
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.711
// CHECK:STDOUT:   %return.param_patt.loc5_52.2 => constants.%return.param_patt.53d
// CHECK:STDOUT:   %return.patt.loc5_49.2 => constants.%return.patt.195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%SubobjectDestroy.lookup_impl_witness.f5e
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.cbf
// CHECK:STDOUT:   %.loc9_54.1 => constants.%.811
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.lookup_impl_witness.79f
// CHECK:STDOUT:   %facet_value.loc9_54.1 => constants.%facet_value.959
// CHECK:STDOUT:   %.loc9_85.1 => constants.%.cbd
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.923
// CHECK:STDOUT:   %facet_value.loc9_85.1 => constants.%facet_value.f1f
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type.48d
// CHECK:STDOUT:   %.loc9_87 => constants.%.712
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2 => constants.%Iterate.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type => constants.%IntRange.as.Iterate.impl.NewCursor.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor => constants.%IntRange.as.Iterate.impl.NewCursor
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type => constants.%IntRange.as.Iterate.impl.Next.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next => constants.%IntRange.as.Iterate.impl.Next
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.NewCursor(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.711
// CHECK:STDOUT:   %self.param_patt.loc10_18.2 => constants.%self.param_patt.b7f
// CHECK:STDOUT:   %self.patt.loc10_18.2 => constants.%self.patt.835f4
// CHECK:STDOUT:   %Int.loc10_37.1 => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %.loc10_37.1 => constants.%.c504e9.1
// CHECK:STDOUT:   %pattern_type.loc10_37 => constants.%pattern_type.5515d5.1
// CHECK:STDOUT:   %return.param_patt.loc10_37.2 => constants.%return.param_patt.a51197.1
// CHECK:STDOUT:   %return.patt.loc10_24.2 => constants.%return.patt.740
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.711
// CHECK:STDOUT:   %self.param_patt.loc11_13.2 => constants.%self.param_patt.b7f
// CHECK:STDOUT:   %self.patt.loc11_13.2 => constants.%self.patt.835f4
// CHECK:STDOUT:   %Int.loc11_41.1 => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %pattern_type.loc11_29 => constants.%pattern_type.5515d5.1
// CHECK:STDOUT:   %cursor.param_patt.loc11_29.2 => constants.%cursor.param_patt.a08
// CHECK:STDOUT:   %cursor.patt.loc11_29.2 => constants.%cursor.patt.019
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%SubobjectDestroy.lookup_impl_witness.f5e
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.lookup_impl_witness.79f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.923
// CHECK:STDOUT:   %facet_value.loc11 => constants.%facet_value.f1f
// CHECK:STDOUT:   %.loc11_72.3 => constants.%.19b
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.f1f
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_72.1 => constants.%OptionalStorage.facet.b7c
// CHECK:STDOUT:   %Optional.loc11_72.1 => constants.%Optional.b5c
// CHECK:STDOUT:   %.loc11_72.4 => constants.%.940
// CHECK:STDOUT:   %pattern_type.loc11_72 => constants.%pattern_type.47f
// CHECK:STDOUT:   %return.param_patt.loc11_72.2 => constants.%return.param_patt.972
// CHECK:STDOUT:   %return.patt.loc11_44.2 => constants.%return.patt.733
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.patt.loc4_17.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.3ab
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.582
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.773
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.85f
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.7e8
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.a5c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.d94
// CHECK:STDOUT:   %start.param_patt.loc5_16.2 => constants.%start.param_patt.2c9
// CHECK:STDOUT:   %start.patt.loc5_16.2 => constants.%start.patt.57a
// CHECK:STDOUT:   %end.param_patt.loc5_34.2 => constants.%end.param_patt.4c3
// CHECK:STDOUT:   %end.patt.loc5_34.2 => constants.%end.patt.9b2
// CHECK:STDOUT:   %IntRange => constants.%IntRange.773
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.bef
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.9c3
// CHECK:STDOUT:   %return.param_patt.loc5_52.2 => constants.%return.param_patt.82a
// CHECK:STDOUT:   %return.patt.loc5_49.2 => constants.%return.patt.9e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc5_49 => constants.%complete_type.a5c
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.7e8
// CHECK:STDOUT:   %.loc6_22.3 => constants.%.72f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.8af
// CHECK:STDOUT:   %Copy.facet.loc6_22.3 => constants.%Copy.facet.95d
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.f76
// CHECK:STDOUT:   %.loc6_22.4 => constants.%.2b9
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.cae
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic]
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0 [symbolic]
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.type: type = parameterized_class_name_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.fc0: type = struct_type {.start: %Int.d4bbe0.1, .end: %Int.d4bbe0.1} [symbolic]
// CHECK:STDOUT:   %complete_type.ae3: <witness> = complete_type_witness %struct_type.start.end.fc0 [symbolic]
// CHECK:STDOUT:   %IntRange.0bb: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.elem.945: type = unbound_element_type %IntRange.0bb, %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %require_complete.d0a0eb.1: <witness> = require_complete_type %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.120: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.646: %IntRange.Make.type.120 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.711: type = pattern_type %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %return.param_patt.53d: %pattern_type.711 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.9da: %pattern_type.711 = return_slot_pattern %return.param_patt.53d, invalid [symbolic]
// CHECK:STDOUT:   %.147: Core.Form = init_form %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %pattern_type.5515d5.1: type = pattern_type %Int.d4bbe0.1 [symbolic]
// CHECK:STDOUT:   %end.param_patt.dde: %pattern_type.5515d5.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %end.patt.962: %pattern_type.5515d5.1 = wrapper_binding_pattern end, %end.param_patt.dde [symbolic]
// CHECK:STDOUT:   %start.param_patt: %pattern_type.5515d5.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %start.patt: %pattern_type.5515d5.1 = wrapper_binding_pattern start, %start.param_patt [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.923: <witness> = lookup_impl_witness %Int.d4bbe0.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.d4bbe0.1, (%Copy.lookup_impl_witness.923) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a18: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.ade: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a18, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.6c8: %.ade = impl_witness_access %Copy.lookup_impl_witness.923, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d42: <specific function> = specific_impl_function %impl.elem0.6c8, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.cbd: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.cca: <witness> = require_complete_type %IntRange.0bb [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.773: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.3ab: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.582: %IntRange.Make.type.3ab = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.elem.85f: type = unbound_element_type %IntRange.773, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.7e8: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a5c: <witness> = complete_type_witness %struct_type.start.end.7e8 [concrete]
// CHECK:STDOUT:   %pattern_type.9c3: type = pattern_type %IntRange.773 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9c3 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9c3 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc5_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6e0: %SubobjectDestroy.type = facet_value %IntRange.773, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.acc: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6e0) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.ece = facet_value %IntRange.773, (%DestroyImpl.impl_witness.acc, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IntRange: %IntRange.type = import_ref Main//lib, IntRange, loaded [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %Main.Range: %Range.type = import_ref Main//lib, Range, loaded [concrete = constants.%Range]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Main.import_ref.6b552a.1: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.1f6: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.ae3)]
// CHECK:STDOUT:   %Main.import_ref.b1d = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a59 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.799 = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.546 = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.6b552a.2: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Main.import_ref.531: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Main//lib, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Main.import_ref.531), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntRange = imports.%Main.IntRange
// CHECK:STDOUT:     .Range = imports.%Main.Range
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .Read = %Read.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Read.decl: %Read.type = fn_decl @Read [concrete = constants.%Read] {
// CHECK:STDOUT:     %y.patt.loc4_18.1: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic = %y.patt.loc4_18.2 (constants.%y.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y.loc4_18.2: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_18.1 (constants.%y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.6b552a.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.120)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.120) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.646)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.d0a0eb.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.945)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.d4bbe0.1), .end: @IntRange.%Int (%Int.d4bbe0.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fc0)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.ae3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.1f6
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b1d
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.a59
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.799
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.546
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Read(%y.loc4_18.2: Core.IntLiteral) {
// CHECK:STDOUT:   %y.patt.loc4_18.2: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic = %y.patt.loc4_18.2 (constants.%y.patt)]
// CHECK:STDOUT:   %y.loc4_18.1: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_18.1 (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y.loc4_18.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_38.3: <bound method> = bound_method %y.loc4_18.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2: init %i32 = call %bound_method.loc5_38.3(%y.loc4_18.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.var: ref %IntRange.773 = var_storage %x.var_patt
// CHECK:STDOUT:     %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:     %y.ref: Core.IntLiteral = name_ref y, %y.loc4_18.2 [symbolic = %y.loc4_18.1 (constants.%y)]
// CHECK:STDOUT:     %.loc5_3.1: ref %IntRange.773 = splice_block %x.var {}
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc5_38.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_38.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1: init %i32 = call %bound_method.loc5_38.2(%y.ref) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.2: %i32 = converted %y.ref, %.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Range.call: init %IntRange.773 to %.loc5_3.1 = call %Range.ref(%.loc5_38.2)
// CHECK:STDOUT:     assign %x.var, %Range.call
// CHECK:STDOUT:     %.loc5_28: type = splice_block %IntRange [concrete = constants.%IntRange.773] {
// CHECK:STDOUT:       %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.773]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %IntRange.773 = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.9c3 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.9c3 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %facet_value.loc5_3.1: %facet_type.ece = facet_value constants.%IntRange.773, (constants.%DestroyImpl.impl_witness.acc, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc5_3.2: %facet_type.ece = converted constants.%IntRange.773, %facet_value.loc5_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:     %facet_value.loc5_3.2: %facet_type.ece = facet_value constants.%IntRange.773, (constants.%DestroyImpl.impl_witness.acc, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc5_3.3: %facet_type.ece = converted constants.%IntRange.773, %facet_value.loc5_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:     %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.6b552a.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.5515d5.1)]
// CHECK:STDOUT:   %start.param_patt: @IntRange.Make.%pattern_type.1 (%pattern_type.5515d5.1) = value_param_pattern [symbolic = %start.param_patt (constants.%start.param_patt)]
// CHECK:STDOUT:   %start.patt: @IntRange.Make.%pattern_type.1 (%pattern_type.5515d5.1) = wrapper_binding_pattern start, %start.param_patt [symbolic = %start.patt (constants.%start.patt)]
// CHECK:STDOUT:   %end.param_patt: @IntRange.Make.%pattern_type.1 (%pattern_type.5515d5.1) = value_param_pattern [symbolic = %end.param_patt (constants.%end.param_patt.dde)]
// CHECK:STDOUT:   %end.patt: @IntRange.Make.%pattern_type.1 (%pattern_type.5515d5.1) = wrapper_binding_pattern end, %end.param_patt [symbolic = %end.patt (constants.%end.patt.962)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.0bb)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %IntRange [symbolic = %.1 (constants.%.147)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.711)]
// CHECK:STDOUT:   %return.param_patt: @IntRange.Make.%pattern_type.2 (%pattern_type.711) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.53d)]
// CHECK:STDOUT:   %return.patt: @IntRange.Make.%pattern_type.2 (%pattern_type.711) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.9da)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %Int [symbolic = %require_complete.1 (constants.%require_complete.d0a0eb.1)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %IntRange [symbolic = %require_complete.2 (constants.%require_complete.cca)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.d4bbe0.1), .end: @IntRange.Make.%Int (%Int.d4bbe0.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fc0)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.2 (constants.%.cbd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.923)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a18)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.3 (constants.%.ade)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.3 (%.ade) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.6c8)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.d42)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc5_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc5_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc5_3.3(%self.param: ref %struct_type.start.end.7e8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc5_3.4(%self.param: ref %IntRange.773) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Read(constants.%y) {
// CHECK:STDOUT:   %y.patt.loc4_18.2 => constants.%y.patt
// CHECK:STDOUT:   %y.loc4_18.1 => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.120
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.646
// CHECK:STDOUT:   %Int => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0a0eb.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.945
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.fc0
// CHECK:STDOUT:   %complete_type => constants.%complete_type.ae3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.d4bbe0.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.5515d5.1
// CHECK:STDOUT:   %start.param_patt => constants.%start.param_patt
// CHECK:STDOUT:   %start.patt => constants.%start.patt
// CHECK:STDOUT:   %end.param_patt => constants.%end.param_patt.dde
// CHECK:STDOUT:   %end.patt => constants.%end.patt.962
// CHECK:STDOUT:   %IntRange => constants.%IntRange.0bb
// CHECK:STDOUT:   %.1 => constants.%.147
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.711
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.53d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.9da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.3ab
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.582
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.773
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.85f
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.7e8
// CHECK:STDOUT:   %complete_type => constants.%complete_type.a5c
// CHECK:STDOUT: }
// CHECK:STDOUT:

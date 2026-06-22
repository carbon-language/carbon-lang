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

class IntRange(N:! Core.IntLiteral) {
  fn Make(start: Core.Int(N), end: Core.Int(N)) -> Self {
    return {.start = start, .end = end};
  }

  impl as Core.Iterate where .CursorType = Core.Int(N) and .ElementType = Core.Int(N) {
    fn NewCursor(self) -> Core.Int(N) { return self.start; }
    fn Next(self, cursor: Core.Int(N)*) -> Core.Optional(Core.Int(N)) {
      var value: Core.Int(N) = *cursor;
      if (value < self.end) {
        ++*cursor;
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

fn Read(y:! Core.IntLiteral) {
  var unused x: IntRange(32) = Range(y);
}


// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.2c4: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.142cb7.1: type = pattern_type %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %start.param_patt.b2c: %pattern_type.142cb7.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %start.patt.9ef: %pattern_type.142cb7.1 = at_binding_pattern start, %start.param_patt.b2c [symbolic]
// CHECK:STDOUT:   %end.param_patt.8ab: %pattern_type.142cb7.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %end.patt.960: %pattern_type.142cb7.1 = at_binding_pattern end, %end.param_patt.8ab [symbolic]
// CHECK:STDOUT:   %.5c3: Core.Form = init_form %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %pattern_type.1ea: type = pattern_type %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %return.param_patt.262: %pattern_type.1ea = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.dbf: %pattern_type.1ea = return_slot_pattern %return.param_patt.262, %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.522: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.8ef: %IntRange.Make.type.522 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.304: type = fn_type @Optional.Some, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Some.2a0: %Optional.Some.type.304 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.None.type.bdc: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.01e: %Optional.None.type.bdc = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.35d [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.c91: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.dd9 [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.543: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.b37, element1 [symbolic_self]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.173: <witness> = lookup_impl_witness %Int.67af24.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7bb: %Destroy.type = facet_value %Int.67af24.1, (%Destroy.lookup_impl_witness.173) [symbolic]
// CHECK:STDOUT:   %assoc0.0f0: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.ebd7 [concrete]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %require_complete.4dfb92.1: <witness> = require_complete_type %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %return.param_patt.5a2679.1: %pattern_type.142cb7.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %.df5d8d.1: Core.Form = init_form %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.df6: <witness> = lookup_impl_witness %Int.67af24.1, @Copy [symbolic]
// CHECK:STDOUT:   %.a53: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.f48 = facet_value %Int.67af24.1, (%Destroy.lookup_impl_witness.173, %Copy.lookup_impl_witness.df6) [symbolic]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.543 = %Destroy.facet.7bb and %impl.elem0.903 = %facet_value> [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.as.Iterate.impl.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.975: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %self.param_patt.5cc: %pattern_type.1ea = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.13b: %pattern_type.1ea = at_binding_pattern self, %self.param_patt.5cc [symbolic]
// CHECK:STDOUT:   %return.patt.434: %pattern_type.142cb7.1 = return_slot_pattern %return.param_patt.5a2679.1, %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.41e: type = ptr_type %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.833: type = pattern_type %ptr.41e [symbolic]
// CHECK:STDOUT:   %cursor.param_patt.328: %pattern_type.833 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %cursor.patt.652: %pattern_type.833 = at_binding_pattern cursor, %cursor.param_patt.328 [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.a32: <witness> = lookup_impl_witness %Int.67af24.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %.17c: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.b99: %OptionalStorage.type = facet_value %Int.67af24.1, (%OptionalStorage.lookup_impl_witness.a32) [symbolic]
// CHECK:STDOUT:   %Optional.5ec: type = class_type @Optional, @Optional(%OptionalStorage.facet.b99) [symbolic]
// CHECK:STDOUT:   %.3f3: Core.Form = init_form %Optional.5ec [symbolic]
// CHECK:STDOUT:   %pattern_type.b38: type = pattern_type %Optional.5ec [symbolic]
// CHECK:STDOUT:   %return.param_patt.4bb: %pattern_type.b38 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.b38 = return_slot_pattern %return.param_patt.4bb, %Optional.5ec [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %start.patt.477: %pattern_type.142cb7.1 = ref_binding_pattern start [symbolic]
// CHECK:STDOUT:   %IntRange.elem.bed: type = unbound_element_type %IntRange.2c4, %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %end.patt.535: %pattern_type.142cb7.1 = ref_binding_pattern end [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.d4d: type = struct_type {.start: %Int.67af24.1, .end: %Int.67af24.1} [symbolic]
// CHECK:STDOUT:   %complete_type.aa1: <witness> = complete_type_witness %struct_type.start.end.d4d [symbolic]
// CHECK:STDOUT:   %require_complete.5d9: <witness> = require_complete_type %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %Copy.facet.f12: %Copy.type = facet_value %Int.67af24.1, (%Copy.lookup_impl_witness.df6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b83: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.f12) [symbolic]
// CHECK:STDOUT:   %.c60: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b83, %Copy.facet.f12 [symbolic]
// CHECK:STDOUT:   %impl.elem0.284: %.c60 = impl_witness_access %Copy.lookup_impl_witness.df6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f9f: <specific function> = specific_impl_function %impl.elem0.284, @Copy.WithSelf.Op(%Copy.facet.f12) [symbolic]
// CHECK:STDOUT:   %require_complete.f79: <witness> = require_complete_type %ptr.41e [symbolic]
// CHECK:STDOUT:   %Optional.None.type.2aa: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.b99) [symbolic]
// CHECK:STDOUT:   %Optional.None.fb7: %Optional.None.type.2aa = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.422: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.b99) [symbolic]
// CHECK:STDOUT:   %Optional.Some.bcf: %Optional.Some.type.422 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.63c: <witness> = require_complete_type %Optional.5ec [symbolic]
// CHECK:STDOUT:   %value.patt.505: %pattern_type.142cb7.1 = ref_binding_pattern value [symbolic]
// CHECK:STDOUT:   %value.var_patt: %pattern_type.142cb7.1 = var_pattern %value.patt.505 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ad8: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.ad8 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32a: type = facet_type <@OrderedWith, @OrderedWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.7df: %OrderedWith.type.32a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.7df) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.9e1: %OrderedWith.WithSelf.Less.type.b87 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.6f8: type = facet_type <@OrderedWith, @OrderedWith(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e6d: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.67af24.1) [symbolic]
// CHECK:STDOUT:   %assoc0.ebc: %OrderedWith.assoc_type.e6d = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic]
// CHECK:STDOUT:   %require_complete.4c8: <witness> = require_complete_type %OrderedWith.type.6f8 [symbolic]
// CHECK:STDOUT:   %assoc0.13c: %OrderedWith.assoc_type.1d2 = assoc_entity element0, imports.%Core.import_ref.447 [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.89b: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.ab3(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.050: %Int.as.OrderedWith.impl.Less.type.89b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.339: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.5c1, @Int.as.OrderedWith.impl.ab3(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.f5c: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.ab3(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.13c: %Int.as.OrderedWith.impl.Less.type.f5c = struct_value () [symbolic]
// CHECK:STDOUT:   %.4a8: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.ab3(%N, %N) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet.bf1: %OrderedWith.type.6f8 = facet_value %Int.67af24.1, (%OrderedWith.impl_witness.339) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c02: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.67af24.1, %OrderedWith.facet.bf1) [symbolic]
// CHECK:STDOUT:   %.08f: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c02, %OrderedWith.facet.bf1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.6ce: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.13c, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.67af24.1, @Inc [symbolic]
// CHECK:STDOUT:   %.f62: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.67af24.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.36a: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %.952: type = fn_type_with_self_type %Inc.WithSelf.Op.type.36a, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.7b5: %.952 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ed1: <specific function> = specific_impl_function %impl.elem0.7b5, @Inc.WithSelf.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.bcf, @Optional.Some(%OptionalStorage.facet.b99) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8f1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7bb) [symbolic]
// CHECK:STDOUT:   %.2c6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8f1, %Destroy.facet.7bb [symbolic]
// CHECK:STDOUT:   %impl.elem0.604: %.2c6 = impl_witness_access %Destroy.lookup_impl_witness.173, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.693: <specific function> = specific_impl_function %impl.elem0.604, @Destroy.WithSelf.Op(%Destroy.facet.7bb) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.fb7, @Optional.None(%OptionalStorage.facet.b99) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %end.param_patt.c55: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %end.patt.367: %pattern_type.6b6 = at_binding_pattern end, %end.param_patt.c55 [concrete]
// CHECK:STDOUT:   %IntRange.bbd: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %.9fd: Core.Form = init_form %IntRange.bbd [concrete]
// CHECK:STDOUT:   %pattern_type.0f3: type = pattern_type %IntRange.bbd [concrete]
// CHECK:STDOUT:   %return.param_patt.108: %pattern_type.0f3 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.7d9: %pattern_type.0f3 = return_slot_pattern %return.param_patt.108, %IntRange.bbd [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.d27: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.9ef: %IntRange.Make.type.d27 = struct_value () [concrete]
// CHECK:STDOUT:   %start.patt.c23: %pattern_type.6b6 = ref_binding_pattern start [concrete]
// CHECK:STDOUT:   %IntRange.elem.6b1: type = unbound_element_type %IntRange.bbd, %i32 [concrete]
// CHECK:STDOUT:   %end.patt.46f: %pattern_type.6b6 = ref_binding_pattern end [concrete]
// CHECK:STDOUT:   %struct_type.start.end.e93: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ec7: <witness> = complete_type_witness %struct_type.start.end.e93 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %start.param_patt.619: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %start.patt.304: %pattern_type.6b6 = at_binding_pattern start, %start.param_patt.619 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.9ef, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f1: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.9f1) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet.9f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %.cf6: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.987: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet.987 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Copy = %Core.Copy.e5d
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Iterate: type = import_ref Core//prelude/iterate, Iterate, loaded [concrete = constants.%Iterate.type]
// CHECK:STDOUT:   %Core.import_ref.96b: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.0f0]
// CHECK:STDOUT:   %Core.import_ref.5be: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1.c91]
// CHECK:STDOUT:   %Core.import_ref.6e4: @Optional.%Optional.None.type (%Optional.None.type.bdc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.01e)]
// CHECK:STDOUT:   %Core.import_ref.84ba: @Optional.%Optional.Some.type (%Optional.Some.type.304) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.2a0)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: %Destroy.type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.ebd7: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %facet_type.f48 = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Copy.e5d: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.ad8 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.ef7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.13c)]
// CHECK:STDOUT:   %Core.import_ref.e7a: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.b87) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.9e1)]
// CHECK:STDOUT:   %Core.import_ref.447 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c1c: @Int.as.OrderedWith.impl.ab3.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.89b) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.ab3.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.050)]
// CHECK:STDOUT:   %Core.import_ref.30c = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.664 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9d4 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.5c1 = impl_witness_table (%Core.import_ref.c1c, %Core.import_ref.30c, %Core.import_ref.664, %Core.import_ref.9d4), @Int.as.OrderedWith.impl.ab3 [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
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
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_17.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: %Range.type = fn_decl @Range [concrete = constants.%Range] {
// CHECK:STDOUT:     %end.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%end.param_patt.c55]
// CHECK:STDOUT:     %end.patt: %pattern_type.6b6 = at_binding_pattern end, %end.param_patt [concrete = constants.%end.patt.367]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0f3 = out_param_pattern [concrete = constants.%return.param_patt.108]
// CHECK:STDOUT:     %return.patt: %pattern_type.0f3 = return_slot_pattern %return.param_patt, %IntRange.loc26 [concrete = constants.%return.patt.7d9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc26: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc26: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.bbd]
// CHECK:STDOUT:     %.loc26_34.2: Core.Form = init_form %IntRange.loc26 [concrete = constants.%.9fd]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %end: %i32 = value_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.bbd = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.bbd = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.173)]
// CHECK:STDOUT:   %Destroy.facet.loc9_54.1: %Destroy.type = facet_value %Int.loc9_54.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:   %.loc9_85.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc9_85.1 (constants.%.a53)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.df6)]
// CHECK:STDOUT:   %facet_value.loc9_85.1: %facet_type.f48 = facet_value %Int.loc9_54.1, (%Destroy.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.543 = %Destroy.facet.loc9_54.1 and constants.%impl.elem0.903 = %facet_value.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.975)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type (constants.%IntRange.as.Iterate.impl.NewCursor.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type (constants.%IntRange.as.Iterate.impl.Next.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %.loc9_24 {
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = fn_decl @IntRange.as.Iterate.impl.NewCursor [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)] {
// CHECK:STDOUT:       %self.param_patt.loc10_18.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.1ea) = value_param_pattern [symbolic = %self.param_patt.loc10_18.2 (constants.%self.param_patt.5cc)]
// CHECK:STDOUT:       %self.patt.loc10_18.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.1ea) = at_binding_pattern self, %self.param_patt.loc10_18.1 [symbolic = %self.patt.loc10_18.2 (constants.%self.patt.13b)]
// CHECK:STDOUT:       %return.param_patt.loc10_37.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.142cb7.1) = out_param_pattern [symbolic = %return.param_patt.loc10_37.2 (constants.%return.param_patt.5a2679.1)]
// CHECK:STDOUT:       %return.patt.loc10_24.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.142cb7.1) = return_slot_pattern %return.param_patt.loc10_37.1, %Int.loc10_37.2 [symbolic = %return.patt.loc10_24.2 (constants.%return.patt.434)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_37.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:       %.loc10_37.2: Core.Form = init_form %Int.loc10_37.2 [symbolic = %.loc10_37.1 (constants.%.df5d8d.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.2c4) = value_param call_param0
// CHECK:STDOUT:       %.loc10_18.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.2c4)] {
// CHECK:STDOUT:         %.loc10_18.2: type = specific_constant constants.%IntRange.2c4, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_18.2 [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.2c4) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.67af24.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.67af24.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.param_patt.loc11_13.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.1ea) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.5cc)]
// CHECK:STDOUT:       %self.patt.loc11_13.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.1ea) = at_binding_pattern self, %self.param_patt.loc11_13.1 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.13b)]
// CHECK:STDOUT:       %cursor.param_patt.loc11_25.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.833) = value_param_pattern [symbolic = %cursor.param_patt.loc11_25.2 (constants.%cursor.param_patt.328)]
// CHECK:STDOUT:       %cursor.patt.loc11_25.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.833) = at_binding_pattern cursor, %cursor.param_patt.loc11_25.1 [symbolic = %cursor.patt.loc11_25.2 (constants.%cursor.patt.652)]
// CHECK:STDOUT:       %return.param_patt.loc11_69.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_69 (%pattern_type.b38) = out_param_pattern [symbolic = %return.param_patt.loc11_69.2 (constants.%return.param_patt.4bb)]
// CHECK:STDOUT:       %return.patt.loc11_41.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_69 (%pattern_type.b38) = return_slot_pattern %return.param_patt.loc11_69.1, %Optional.loc11_69.2 [symbolic = %return.patt.loc11_41.2 (constants.%return.patt.427)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_58: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_62: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_67: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_68: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_69.2: %OptionalStorage.type = facet_value %Int.loc11_68, (constants.%OptionalStorage.lookup_impl_witness.a32) [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:       %.loc11_69.5: %OptionalStorage.type = converted %Int.loc11_68, %OptionalStorage.facet.loc11_69.2 [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:       %Optional.loc11_69.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.loc11_69.1 (constants.%Optional.5ec)]
// CHECK:STDOUT:       %.loc11_69.6: Core.Form = init_form %Optional.loc11_69.2 [symbolic = %.loc11_69.4 (constants.%.3f3)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.2c4) = value_param call_param0
// CHECK:STDOUT:       %.loc11_13.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.2c4)] {
// CHECK:STDOUT:         %.loc11_13.2: type = specific_constant constants.%IntRange.2c4, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_13.2 [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.2c4) = value_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_38.1 (%ptr.41e) = value_param call_param1
// CHECK:STDOUT:       %.loc11_38: type = splice_block %ptr.loc11_38.2 [symbolic = %ptr.loc11_38.1 (constants.%ptr.41e)] {
// CHECK:STDOUT:         %Core.ref.loc11_27: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_31: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_36: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_37.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:         %ptr.loc11_38.2: type = ptr_type %Int.loc11_37.2 [symbolic = %ptr.loc11_38.1 (constants.%ptr.41e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.%ptr.loc11_38.1 (%ptr.41e) = value_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, %IntRange.as.Iterate.impl.NewCursor.decl, %IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness.loc9_87.1: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: %Destroy.type = impl_witness_assoc_constant constants.%Destroy.facet.7bb [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %facet_type.f48 = impl_witness_assoc_constant constants.%facet_value [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .NewCursor = %IntRange.as.Iterate.impl.NewCursor.decl
// CHECK:STDOUT:     .Next = %IntRange.as.Iterate.impl.Next.decl
// CHECK:STDOUT:     extend %.loc9_24
// CHECK:STDOUT:     witness = %Iterate.impl_witness.loc9_87.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc4_17.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_17.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc4_17.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_17.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.522)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.522) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.8ef)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N.loc4_17.1) [symbolic = %Int (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.4dfb92.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int [symbolic = %pattern_type (constants.%pattern_type.142cb7.1)]
// CHECK:STDOUT:   %start.patt: @IntRange.%pattern_type (%pattern_type.142cb7.1) = ref_binding_pattern start [symbolic = %start.patt (constants.%start.patt.477)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_17.1) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.bed)]
// CHECK:STDOUT:   %end.patt: @IntRange.%pattern_type (%pattern_type.142cb7.1) = ref_binding_pattern end [symbolic = %end.patt (constants.%end.patt.535)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.67af24.1), .end: @IntRange.%Int (%Int.67af24.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.d4d)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.522) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.8ef)] {
// CHECK:STDOUT:       %start.param_patt.loc5_16.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = value_param_pattern [symbolic = %start.param_patt.loc5_16.2 (constants.%start.param_patt.b2c)]
// CHECK:STDOUT:       %start.patt.loc5_16.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = at_binding_pattern start, %start.param_patt.loc5_16.1 [symbolic = %start.patt.loc5_16.2 (constants.%start.patt.9ef)]
// CHECK:STDOUT:       %end.param_patt.loc5_34.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = value_param_pattern [symbolic = %end.param_patt.loc5_34.2 (constants.%end.param_patt.8ab)]
// CHECK:STDOUT:       %end.patt.loc5_34.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = at_binding_pattern end, %end.param_patt.loc5_34.1 [symbolic = %end.patt.loc5_34.2 (constants.%end.patt.960)]
// CHECK:STDOUT:       %return.param_patt.loc5_52.1: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.1ea) = out_param_pattern [symbolic = %return.param_patt.loc5_52.2 (constants.%return.param_patt.262)]
// CHECK:STDOUT:       %return.patt.loc5_49.1: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.1ea) = return_slot_pattern %return.param_patt.loc5_52.1, %Self.ref [symbolic = %return.patt.loc5_49.2 (constants.%return.patt.dbf)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52.2: type = specific_constant constants.%IntRange.2c4, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52.2 [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:       %.loc5_52.3: Core.Form = init_form %Self.ref [symbolic = %.loc5_52.1 (constants.%.5c3)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.67af24.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = value_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.67af24.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = value_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.2c4) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.2c4) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.2c4 [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.35d]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.35d]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.5be [concrete = constants.%assoc1.c91]
// CHECK:STDOUT:       %impl.elem1: %Destroy.type = impl_witness_access constants.%Iterate.lookup_impl_witness.b37, element1 [symbolic_self = constants.%impl.elem1.543]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:       %Destroy.facet.loc9_54.2: %Destroy.type = facet_value %Int.loc9_54.2, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:       %.loc9_54: %Destroy.type = converted %Int.loc9_54.2, %Destroy.facet.loc9_54.2 [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.35d]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.96b [concrete = constants.%assoc0.0f0]
// CHECK:STDOUT:       %impl.elem0: %facet_type.f48 = impl_witness_access constants.%Iterate.lookup_impl_witness.b37, element0 [symbolic_self = constants.%impl.elem0.903]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:       %facet_value.loc9_85.2: %facet_type.f48 = facet_value %Int.loc9_85, (constants.%Destroy.lookup_impl_witness.173, constants.%Copy.lookup_impl_witness.df6) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_85.2: %facet_type.f48 = converted %Int.loc9_85, %facet_value.loc9_85.2 [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type %Iterate.ref
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc9_54
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_85.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc22: @IntRange.%IntRange.elem (%IntRange.elem.bed) = field_decl start, element0 [concrete]
// CHECK:STDOUT:     %.loc23: @IntRange.%IntRange.elem (%IntRange.elem.bed) = field_decl end, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.d4d [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.aa1)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.2c4
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %.loc22
// CHECK:STDOUT:     .end [private] = %.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.142cb7.1)]
// CHECK:STDOUT:   %start.param_patt.loc5_16.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = value_param_pattern [symbolic = %start.param_patt.loc5_16.2 (constants.%start.param_patt.b2c)]
// CHECK:STDOUT:   %start.patt.loc5_16.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = at_binding_pattern start, %start.param_patt.loc5_16.2 [symbolic = %start.patt.loc5_16.2 (constants.%start.patt.9ef)]
// CHECK:STDOUT:   %end.param_patt.loc5_34.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = value_param_pattern [symbolic = %end.param_patt.loc5_34.2 (constants.%end.param_patt.8ab)]
// CHECK:STDOUT:   %end.patt.loc5_34.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.142cb7.1) = at_binding_pattern end, %end.param_patt.loc5_34.2 [symbolic = %end.patt.loc5_34.2 (constants.%end.patt.960)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %.loc5_52.1: Core.Form = init_form %IntRange [symbolic = %.loc5_52.1 (constants.%.5c3)]
// CHECK:STDOUT:   %pattern_type.loc5_52: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_52 (constants.%pattern_type.1ea)]
// CHECK:STDOUT:   %return.param_patt.loc5_52.2: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.1ea) = out_param_pattern [symbolic = %return.param_patt.loc5_52.2 (constants.%return.param_patt.262)]
// CHECK:STDOUT:   %return.patt.loc5_49.2: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.1ea) = return_slot_pattern %return.param_patt.loc5_52.2, %IntRange [symbolic = %return.patt.loc5_49.2 (constants.%return.patt.dbf)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.4dfb92.1)]
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.5d9)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.d4d)]
// CHECK:STDOUT:   %.loc6_22.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc6_22.3 (constants.%.a53)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.df6)]
// CHECK:STDOUT:   %Copy.facet.loc6_22.3: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc6_22.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.b83)]
// CHECK:STDOUT:   %.loc6_22.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc6_22.3 [symbolic = %.loc6_22.4 (constants.%.c60)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.4 (%.c60) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.WithSelf.Op(%Copy.facet.loc6_22.3) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1)) -> out %return.param: @IntRange.Make.%IntRange (%IntRange.2c4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.d4d) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.4 (%.c60) = impl_witness_access constants.%Copy.lookup_impl_witness.df6, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %Copy.facet.loc6_22.1: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc6_22.1: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc6_22.1 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %Copy.facet.loc6_22.2: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc6_22.2: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc6_22.2 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.WithSelf.Op(constants.%Copy.facet.f12) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_22: init @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = class_element_access %return.param, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) to %.loc6_39.2 = in_place_init %Copy.WithSelf.Op.call.loc6_22
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.4 (%.c60) = impl_witness_access constants.%Copy.lookup_impl_witness.df6, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %Copy.facet.loc6_36.1: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc6_36.1: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc6_36.1 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %Copy.facet.loc6_36.2: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc6_36.2: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc6_36.2 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.WithSelf.Op(constants.%Copy.facet.f12) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) = class_element_access %return.param, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.67af24.1) to %.loc6_39.4 = in_place_init %Copy.WithSelf.Op.call.loc6_36
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.2c4) to %return.param = class_init (%.loc6_39.3, %.loc6_39.5)
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.2c4) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.1ea)]
// CHECK:STDOUT:   %self.param_patt.loc10_18.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.1ea) = value_param_pattern [symbolic = %self.param_patt.loc10_18.2 (constants.%self.param_patt.5cc)]
// CHECK:STDOUT:   %self.patt.loc10_18.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.1ea) = at_binding_pattern self, %self.param_patt.loc10_18.2 [symbolic = %self.patt.loc10_18.2 (constants.%self.patt.13b)]
// CHECK:STDOUT:   %Int.loc10_37.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %.loc10_37.1: Core.Form = init_form %Int.loc10_37.1 [symbolic = %.loc10_37.1 (constants.%.df5d8d.1)]
// CHECK:STDOUT:   %pattern_type.loc10_37: type = pattern_type %Int.loc10_37.1 [symbolic = %pattern_type.loc10_37 (constants.%pattern_type.142cb7.1)]
// CHECK:STDOUT:   %return.param_patt.loc10_37.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.142cb7.1) = out_param_pattern [symbolic = %return.param_patt.loc10_37.2 (constants.%return.param_patt.5a2679.1)]
// CHECK:STDOUT:   %return.patt.loc10_24.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.142cb7.1) = return_slot_pattern %return.param_patt.loc10_37.2, %Int.loc10_37.1 [symbolic = %return.patt.loc10_24.2 (constants.%return.patt.434)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_18: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_18 (constants.%require_complete.5d9)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_37.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.bed)]
// CHECK:STDOUT:   %require_complete.loc10_52: <witness> = require_complete_type %Int.loc10_37.1 [symbolic = %require_complete.loc10_52 (constants.%require_complete.4dfb92.1)]
// CHECK:STDOUT:   %.loc10_52.5: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc10_52.5 (constants.%.a53)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc10_37.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.df6)]
// CHECK:STDOUT:   %Copy.facet.loc10_52.3: %Copy.type = facet_value %Int.loc10_37.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_52.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.b83)]
// CHECK:STDOUT:   %.loc10_52.6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_52.3 [symbolic = %.loc10_52.6 (constants.%.c60)]
// CHECK:STDOUT:   %impl.elem0.loc10_52.2: @IntRange.as.Iterate.impl.NewCursor.%.loc10_52.6 (%.c60) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_52.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_52.2: <specific function> = specific_impl_function %impl.elem0.loc10_52.2, @Copy.WithSelf.Op(%Copy.facet.loc10_52.3) [symbolic = %specific_impl_fn.loc10_52.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.2c4)) -> out %return.param: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.67af24.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.2c4) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.bed) = name_ref start, @IntRange.%.loc22 [concrete = @IntRange.%.loc22]
// CHECK:STDOUT:     %.loc10_52.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.67af24.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_52.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.67af24.1) = acquire_value %.loc10_52.1
// CHECK:STDOUT:     %impl.elem0.loc10_52.1: @IntRange.as.Iterate.impl.NewCursor.%.loc10_52.6 (%.c60) = impl_witness_access constants.%Copy.lookup_impl_witness.df6, element0 [symbolic = %impl.elem0.loc10_52.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:     %bound_method.loc10_52.1: <bound method> = bound_method %.loc10_52.2, %impl.elem0.loc10_52.1
// CHECK:STDOUT:     %Copy.facet.loc10_52.1: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc10_52.3: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc10_52.1 [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %Copy.facet.loc10_52.2: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc10_52.4: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc10_52.2 [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_52.1: <specific function> = specific_impl_function %impl.elem0.loc10_52.1, @Copy.WithSelf.Op(constants.%Copy.facet.f12) [symbolic = %specific_impl_fn.loc10_52.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:     %bound_method.loc10_52.2: <bound method> = bound_method %.loc10_52.2, %specific_impl_fn.loc10_52.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.67af24.1) = call %bound_method.loc10_52.2(%.loc10_52.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.1ea)]
// CHECK:STDOUT:   %self.param_patt.loc11_13.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.1ea) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.5cc)]
// CHECK:STDOUT:   %self.patt.loc11_13.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.1ea) = at_binding_pattern self, %self.param_patt.loc11_13.2 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.13b)]
// CHECK:STDOUT:   %Int.loc11_37.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %ptr.loc11_38.1: type = ptr_type %Int.loc11_37.1 [symbolic = %ptr.loc11_38.1 (constants.%ptr.41e)]
// CHECK:STDOUT:   %pattern_type.loc11_25: type = pattern_type %ptr.loc11_38.1 [symbolic = %pattern_type.loc11_25 (constants.%pattern_type.833)]
// CHECK:STDOUT:   %cursor.param_patt.loc11_25.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.833) = value_param_pattern [symbolic = %cursor.param_patt.loc11_25.2 (constants.%cursor.param_patt.328)]
// CHECK:STDOUT:   %cursor.patt.loc11_25.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.833) = at_binding_pattern cursor, %cursor.param_patt.loc11_25.2 [symbolic = %cursor.patt.loc11_25.2 (constants.%cursor.patt.652)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_37.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.173)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_37.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.df6)]
// CHECK:STDOUT:   %facet_value: %facet_type.f48 = facet_value %Int.loc11_37.1, (%Destroy.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %.loc11_69.3: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value) [symbolic = %.loc11_69.3 (constants.%.17c)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_37.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.a32)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_69.1: %OptionalStorage.type = facet_value %Int.loc11_37.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:   %Optional.loc11_69.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc11_69.1) [symbolic = %Optional.loc11_69.1 (constants.%Optional.5ec)]
// CHECK:STDOUT:   %.loc11_69.4: Core.Form = init_form %Optional.loc11_69.1 [symbolic = %.loc11_69.4 (constants.%.3f3)]
// CHECK:STDOUT:   %pattern_type.loc11_69: type = pattern_type %Optional.loc11_69.1 [symbolic = %pattern_type.loc11_69 (constants.%pattern_type.b38)]
// CHECK:STDOUT:   %return.param_patt.loc11_69.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_69 (%pattern_type.b38) = out_param_pattern [symbolic = %return.param_patt.loc11_69.2 (constants.%return.param_patt.4bb)]
// CHECK:STDOUT:   %return.patt.loc11_41.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_69 (%pattern_type.b38) = return_slot_pattern %return.param_patt.loc11_69.2, %Optional.loc11_69.1 [symbolic = %return.patt.loc11_41.2 (constants.%return.patt.427)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_13: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_13 (constants.%require_complete.5d9)]
// CHECK:STDOUT:   %require_complete.loc11_25: <witness> = require_complete_type %ptr.loc11_38.1 [symbolic = %require_complete.loc11_25 (constants.%require_complete.f79)]
// CHECK:STDOUT:   %require_complete.loc11_41: <witness> = require_complete_type %Optional.loc11_69.1 [symbolic = %require_complete.loc11_41 (constants.%require_complete.63c)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %Int.loc11_37.1 [symbolic = %require_complete.loc12 (constants.%require_complete.4dfb92.1)]
// CHECK:STDOUT:   %pattern_type.loc12: type = pattern_type %Int.loc11_37.1 [symbolic = %pattern_type.loc12 (constants.%pattern_type.142cb7.1)]
// CHECK:STDOUT:   %value.patt.loc12_16.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.142cb7.1) = ref_binding_pattern value [symbolic = %value.patt.loc12_16.2 (constants.%value.patt.505)]
// CHECK:STDOUT:   %value.var_patt.loc12_7.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.142cb7.1) = var_pattern %value.patt.loc12_16.2 [symbolic = %value.var_patt.loc12_7.2 (constants.%value.var_patt)]
// CHECK:STDOUT:   %.loc12_32.5: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc12_32.5 (constants.%.a53)]
// CHECK:STDOUT:   %Copy.facet.loc12_32.3: %Copy.type = facet_value %Int.loc11_37.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc12_32.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.b83)]
// CHECK:STDOUT:   %.loc12_32.6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc12_32.3 [symbolic = %.loc12_32.6 (constants.%.c60)]
// CHECK:STDOUT:   %impl.elem0.loc12_32.2: @IntRange.as.Iterate.impl.Next.%.loc12_32.6 (%.c60) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_32.2: <specific function> = specific_impl_function %impl.elem0.loc12_32.2, @Copy.WithSelf.Op(%Copy.facet.loc12_32.3) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_37.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.bed)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_37.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.6f8)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.4c8)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_37.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.e6d)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e6d) = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic = %assoc0 (constants.%assoc0.ebc)]
// CHECK:STDOUT:   %.loc13_17.2: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.ab3(%N, %N) [symbolic = %.loc13_17.2 (constants.%.4a8)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.5c1, @Int.as.OrderedWith.impl.ab3(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.339)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.6f8) = facet_value %Int.loc11_37.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet.bf1)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.loc11_37.1, %OrderedWith.facet) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.c02)]
// CHECK:STDOUT:   %.loc13_17.3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.3 (constants.%.08f)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.ab3(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.f5c)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.f5c) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.13c)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.6ce)]
// CHECK:STDOUT:   %.loc14_9.3: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic = %.loc14_9.3 (constants.%.f62)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_37.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet.loc14_9.3: %Inc.type = facet_value %Int.loc11_37.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.loc14_9.3) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.36a)]
// CHECK:STDOUT:   %.loc14_9.4: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %Inc.facet.loc14_9.3 [symbolic = %.loc14_9.4 (constants.%.952)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.4 (%.952) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.7b5)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.WithSelf.Op(%Inc.facet.loc14_9.3) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.ed1)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc11_69.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.422)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.422) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.bcf)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc11_69.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.facet.loc12_7.5: %Destroy.type = facet_value %Int.loc11_37.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc12_7.5) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.8f1)]
// CHECK:STDOUT:   %.loc12_7.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc12_7.5 [symbolic = %.loc12_7.5 (constants.%.2c6)]
// CHECK:STDOUT:   %impl.elem0.loc12_7.3: @IntRange.as.Iterate.impl.Next.%.loc12_7.5 (%.2c6) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.604)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_7.3: <specific function> = specific_impl_function %impl.elem0.loc12_7.3, @Destroy.WithSelf.Op(%Destroy.facet.loc12_7.5) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.693)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc11_69.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.2aa)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.2aa) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.fb7)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc11_69.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.2c4), %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_38.1 (%ptr.41e)) -> out %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt.loc12_16.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.142cb7.1) = ref_binding_pattern value [symbolic = %value.patt.loc12_16.2 (constants.%value.patt.505)]
// CHECK:STDOUT:       %value.var_patt.loc12_7.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.142cb7.1) = var_pattern %value.patt.loc12_16.1 [symbolic = %value.var_patt.loc12_7.2 (constants.%value.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = var %value.var_patt.loc12_7.1
// CHECK:STDOUT:     %cursor.ref.loc12: @IntRange.as.Iterate.impl.Next.%ptr.loc11_38.1 (%ptr.41e) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = deref %cursor.ref.loc12
// CHECK:STDOUT:     %.loc12_32.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = acquire_value %.loc12_32.1
// CHECK:STDOUT:     %impl.elem0.loc12_32.1: @IntRange.as.Iterate.impl.Next.%.loc12_32.6 (%.c60) = impl_witness_access constants.%Copy.lookup_impl_witness.df6, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:     %bound_method.loc12_32.1: <bound method> = bound_method %.loc12_32.2, %impl.elem0.loc12_32.1
// CHECK:STDOUT:     %Copy.facet.loc12_32.1: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc12_32.3: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc12_32.1 [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %Copy.facet.loc12_32.2: %Copy.type = facet_value constants.%Int.67af24.1, (constants.%Copy.lookup_impl_witness.df6) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %.loc12_32.4: %Copy.type = converted constants.%Int.67af24.1, %Copy.facet.loc12_32.2 [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_32.1: <specific function> = specific_impl_function %impl.elem0.loc12_32.1, @Copy.WithSelf.Op(constants.%Copy.facet.f12) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:     %bound_method.loc12_32.2: <bound method> = bound_method %.loc12_32.2, %specific_impl_fn.loc12_32.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = call %bound_method.loc12_32.2(%.loc12_32.2)
// CHECK:STDOUT:     assign %value.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = ref_binding value, %value.var
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.2c4) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.bed) = name_ref end, @IntRange.%.loc23 [concrete = @IntRange.%.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = acquire_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.67af24.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.6f8)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e6d) = specific_constant imports.%Core.import_ref.ef7, @OrderedWith.WithSelf(constants.%Int.67af24.1, constants.%Self.7df) [symbolic = %assoc0 (constants.%assoc0.ebc)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e6d) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.ebc)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.3 (%.08f) = impl_witness_access constants.%OrderedWith.impl_witness.339, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.13c)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.6ce)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = acquire_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_38.1 (%ptr.41e) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc14_11: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = deref %cursor.ref.loc14
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.4 (%.952) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.7b5)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %.loc14_11, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %Inc.facet.loc14_9.1: %Inc.type = facet_value constants.%Int.67af24.1, (constants.%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %.loc14_9.1: %Inc.type = converted constants.%Int.67af24.1, %Inc.facet.loc14_9.1 [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %Inc.facet.loc14_9.2: %Inc.type = facet_value constants.%Int.67af24.1, (constants.%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %.loc14_9.2: %Inc.type = converted constants.%Int.67af24.1, %Inc.facet.loc14_9.2 [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.WithSelf.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.ed1)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %.loc14_11, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %Inc.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_9.2(%.loc14_11)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_41: %OptionalStorage.type = facet_value %Int.loc15, (constants.%OptionalStorage.lookup_impl_witness.a32) [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %.loc15_41: %OptionalStorage.type = converted %Int.loc15, %OptionalStorage.facet.loc15_41 [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.loc11_69.1 (constants.%Optional.5ec)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.422) = specific_constant imports.%Core.import_ref.84ba, @Optional(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.Some (constants.%Optional.Some.bcf)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.422) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.bcf)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.1: %OptionalStorage.type = facet_value constants.%Int.67af24.1, (constants.%OptionalStorage.lookup_impl_witness.a32) [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %.loc15_53.1: %OptionalStorage.type = converted constants.%Int.67af24.1, %OptionalStorage.facet.loc15_53.1 [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.2: %OptionalStorage.type = facet_value constants.%Int.67af24.1, (constants.%OptionalStorage.lookup_impl_witness.a32) [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %.loc15_53.2: %OptionalStorage.type = converted constants.%Int.67af24.1, %OptionalStorage.facet.loc15_53.2 [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_69.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_37.1 (%Int.67af24.1) = acquire_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) to %.loc11_69.1 = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48)
// CHECK:STDOUT:     %impl.elem0.loc12_7.1: @IntRange.as.Iterate.impl.Next.%.loc12_7.5 (%.2c6) = impl_witness_access constants.%Destroy.lookup_impl_witness.173, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.604)]
// CHECK:STDOUT:     %bound_method.loc12_7.1: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.1
// CHECK:STDOUT:     %Destroy.facet.loc12_7.1: %Destroy.type = facet_value constants.%Int.67af24.1, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %.loc12_7.1: %Destroy.type = converted constants.%Int.67af24.1, %Destroy.facet.loc12_7.1 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %Destroy.facet.loc12_7.2: %Destroy.type = facet_value constants.%Int.67af24.1, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %.loc12_7.2: %Destroy.type = converted constants.%Int.67af24.1, %Destroy.facet.loc12_7.2 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.1: <specific function> = specific_impl_function %impl.elem0.loc12_7.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.7bb) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.693)]
// CHECK:STDOUT:     %bound_method.loc12_7.2: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc12_7.1: init %empty_tuple.type = call %bound_method.loc12_7.2(%value.var)
// CHECK:STDOUT:     return %Optional.Some.call to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_37.1 (constants.%Int.67af24.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc17: %OptionalStorage.type = facet_value %Int.loc17, (constants.%OptionalStorage.lookup_impl_witness.a32) [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %.loc17_41: %OptionalStorage.type = converted %Int.loc17, %OptionalStorage.facet.loc17 [symbolic = %OptionalStorage.facet.loc11_69.1 (constants.%OptionalStorage.facet.b99)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.loc11_69.1 (constants.%Optional.5ec)]
// CHECK:STDOUT:     %.loc17_42: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.2aa) = specific_constant imports.%Core.import_ref.6e4, @Optional(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.None (constants.%Optional.None.fb7)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.2aa) = name_ref None, %.loc17_42 [symbolic = %Optional.None (constants.%Optional.None.fb7)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.b99) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_69.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) = splice_block %return.param {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_69.1 (%Optional.5ec) to %.loc11_69.2 = call %Optional.None.specific_fn.loc17_42.1()
// CHECK:STDOUT:     %impl.elem0.loc12_7.2: @IntRange.as.Iterate.impl.Next.%.loc12_7.5 (%.2c6) = impl_witness_access constants.%Destroy.lookup_impl_witness.173, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.604)]
// CHECK:STDOUT:     %bound_method.loc12_7.3: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.2
// CHECK:STDOUT:     %Destroy.facet.loc12_7.3: %Destroy.type = facet_value constants.%Int.67af24.1, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %.loc12_7.3: %Destroy.type = converted constants.%Int.67af24.1, %Destroy.facet.loc12_7.3 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %Destroy.facet.loc12_7.4: %Destroy.type = facet_value constants.%Int.67af24.1, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %.loc12_7.4: %Destroy.type = converted constants.%Int.67af24.1, %Destroy.facet.loc12_7.4 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.2: <specific function> = specific_impl_function %impl.elem0.loc12_7.2, @Destroy.WithSelf.Op(constants.%Destroy.facet.7bb) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.693)]
// CHECK:STDOUT:     %bound_method.loc12_7.4: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc12_7.2: init %empty_tuple.type = call %bound_method.loc12_7.4(%value.var)
// CHECK:STDOUT:     return %Optional.None.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> out %return.param: %IntRange.bbd {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.bbd]
// CHECK:STDOUT:   %.loc27_22: %IntRange.Make.type.d27 = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.9ef]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.d27 = name_ref Make, %.loc27_22 [concrete = constants.%IntRange.Make.9ef]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc26_34.1: ref %IntRange.bbd = splice_block %return.param {}
// CHECK:STDOUT:   %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.bbd to %.loc26_34.1 = call %IntRange.Make.specific_fn(%.loc27_28.2, %end.ref)
// CHECK:STDOUT:   return %IntRange.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc4_17.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.522
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.8ef
// CHECK:STDOUT:   %Int => constants.%Int.67af24.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4dfb92.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.142cb7.1
// CHECK:STDOUT:   %start.patt => constants.%start.patt.477
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.bed
// CHECK:STDOUT:   %end.patt => constants.%end.patt.535
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d4d
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.aa1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.67af24.1
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.142cb7.1
// CHECK:STDOUT:   %start.param_patt.loc5_16.2 => constants.%start.param_patt.b2c
// CHECK:STDOUT:   %start.patt.loc5_16.2 => constants.%start.patt.9ef
// CHECK:STDOUT:   %end.param_patt.loc5_34.2 => constants.%end.param_patt.8ab
// CHECK:STDOUT:   %end.patt.loc5_34.2 => constants.%end.patt.960
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.5c3
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.1ea
// CHECK:STDOUT:   %return.param_patt.loc5_52.2 => constants.%return.param_patt.262
// CHECK:STDOUT:   %return.patt.loc5_49.2 => constants.%return.patt.dbf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.67af24.1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.173
// CHECK:STDOUT:   %Destroy.facet.loc9_54.1 => constants.%Destroy.facet.7bb
// CHECK:STDOUT:   %.loc9_85.1 => constants.%.a53
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.df6
// CHECK:STDOUT:   %facet_value.loc9_85.1 => constants.%facet_value
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2 => constants.%Iterate.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.975
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type => constants.%IntRange.as.Iterate.impl.NewCursor.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor => constants.%IntRange.as.Iterate.impl.NewCursor
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type => constants.%IntRange.as.Iterate.impl.Next.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next => constants.%IntRange.as.Iterate.impl.Next
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.NewCursor(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.1ea
// CHECK:STDOUT:   %self.param_patt.loc10_18.2 => constants.%self.param_patt.5cc
// CHECK:STDOUT:   %self.patt.loc10_18.2 => constants.%self.patt.13b
// CHECK:STDOUT:   %Int.loc10_37.1 => constants.%Int.67af24.1
// CHECK:STDOUT:   %.loc10_37.1 => constants.%.df5d8d.1
// CHECK:STDOUT:   %pattern_type.loc10_37 => constants.%pattern_type.142cb7.1
// CHECK:STDOUT:   %return.param_patt.loc10_37.2 => constants.%return.param_patt.5a2679.1
// CHECK:STDOUT:   %return.patt.loc10_24.2 => constants.%return.patt.434
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.1ea
// CHECK:STDOUT:   %self.param_patt.loc11_13.2 => constants.%self.param_patt.5cc
// CHECK:STDOUT:   %self.patt.loc11_13.2 => constants.%self.patt.13b
// CHECK:STDOUT:   %Int.loc11_37.1 => constants.%Int.67af24.1
// CHECK:STDOUT:   %ptr.loc11_38.1 => constants.%ptr.41e
// CHECK:STDOUT:   %pattern_type.loc11_25 => constants.%pattern_type.833
// CHECK:STDOUT:   %cursor.param_patt.loc11_25.2 => constants.%cursor.param_patt.328
// CHECK:STDOUT:   %cursor.patt.loc11_25.2 => constants.%cursor.patt.652
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.173
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.df6
// CHECK:STDOUT:   %facet_value => constants.%facet_value
// CHECK:STDOUT:   %.loc11_69.3 => constants.%.17c
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.a32
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_69.1 => constants.%OptionalStorage.facet.b99
// CHECK:STDOUT:   %Optional.loc11_69.1 => constants.%Optional.5ec
// CHECK:STDOUT:   %.loc11_69.4 => constants.%.3f3
// CHECK:STDOUT:   %pattern_type.loc11_69 => constants.%pattern_type.b38
// CHECK:STDOUT:   %return.param_patt.loc11_69.2 => constants.%return.param_patt.4bb
// CHECK:STDOUT:   %return.patt.loc11_41.2 => constants.%return.patt.427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.patt.loc4_17.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.d27
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.9ef
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %start.patt => constants.%start.patt.c23
// CHECK:STDOUT:   %IntRange => constants.%IntRange.bbd
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.6b1
// CHECK:STDOUT:   %end.patt => constants.%end.patt.46f
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.e93
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.ec7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %start.param_patt.loc5_16.2 => constants.%start.param_patt.619
// CHECK:STDOUT:   %start.patt.loc5_16.2 => constants.%start.patt.304
// CHECK:STDOUT:   %end.param_patt.loc5_34.2 => constants.%end.param_patt.c55
// CHECK:STDOUT:   %end.patt.loc5_34.2 => constants.%end.patt.367
// CHECK:STDOUT:   %IntRange => constants.%IntRange.bbd
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.9fd
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.0f3
// CHECK:STDOUT:   %return.param_patt.loc5_52.2 => constants.%return.param_patt.108
// CHECK:STDOUT:   %return.patt.loc5_49.2 => constants.%return.patt.7d9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc5_49 => constants.%complete_type.ec7
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.e93
// CHECK:STDOUT:   %.loc6_22.3 => constants.%.cf6
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.0e0
// CHECK:STDOUT:   %Copy.facet.loc6_22.3 => constants.%Copy.facet.987
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d09
// CHECK:STDOUT:   %.loc6_22.4 => constants.%.7a6
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.4f6
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic]
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0 [symbolic]
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.d4d: type = struct_type {.start: %Int.67af24.1, .end: %Int.67af24.1} [symbolic]
// CHECK:STDOUT:   %complete_type.aa1: <witness> = complete_type_witness %struct_type.start.end.d4d [symbolic]
// CHECK:STDOUT:   %IntRange.2c4: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.142cb7.1: type = pattern_type %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %end.patt.535: %pattern_type.142cb7.1 = ref_binding_pattern end [symbolic]
// CHECK:STDOUT:   %IntRange.elem.bed: type = unbound_element_type %IntRange.2c4, %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %start.patt.477: %pattern_type.142cb7.1 = ref_binding_pattern start [symbolic]
// CHECK:STDOUT:   %require_complete.4dfb92.1: <witness> = require_complete_type %Int.67af24.1 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.522: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.8ef: %IntRange.Make.type.522 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.1ea: type = pattern_type %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %return.param_patt.262: %pattern_type.1ea = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.ca9: %pattern_type.1ea = return_slot_pattern %return.param_patt.262, invalid [symbolic]
// CHECK:STDOUT:   %.5c3: Core.Form = init_form %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %end.param_patt.8ab: %pattern_type.142cb7.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %end.patt.960: %pattern_type.142cb7.1 = at_binding_pattern end, %end.param_patt.8ab [symbolic]
// CHECK:STDOUT:   %start.param_patt: %pattern_type.142cb7.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %start.patt.9ef: %pattern_type.142cb7.1 = at_binding_pattern start, %start.param_patt [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.df6: <witness> = lookup_impl_witness %Int.67af24.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.67af24.1, (%Copy.lookup_impl_witness.df6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b83: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.c60: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b83, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.284: %.c60 = impl_witness_access %Copy.lookup_impl_witness.df6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f9f: <specific function> = specific_impl_function %impl.elem0.284, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.a53: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.5d9: <witness> = require_complete_type %IntRange.2c4 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.bbd: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.d27: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.9ef: %IntRange.Make.type.d27 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %start.patt.c23: %pattern_type.6b6 = ref_binding_pattern start [concrete]
// CHECK:STDOUT:   %IntRange.elem.6b1: type = unbound_element_type %IntRange.bbd, %i32 [concrete]
// CHECK:STDOUT:   %end.patt.46f: %pattern_type.6b6 = ref_binding_pattern end [concrete]
// CHECK:STDOUT:   %struct_type.start.end.e93: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ec7: <witness> = complete_type_witness %struct_type.start.end.e93 [concrete]
// CHECK:STDOUT:   %pattern_type.0f3: type = pattern_type %IntRange.bbd [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.0f3 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.0f3 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc5_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IntRange: %IntRange.type = import_ref Main//lib, IntRange, loaded [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %Main.Range: %Range.type = import_ref Main//lib, Range, loaded [concrete = constants.%Range]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Main.import_ref.6b552a.1: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.e06: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.aa1)]
// CHECK:STDOUT:   %Main.import_ref.663 = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.ac1 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.d41 = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.60f = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.6b552a.2: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %y.patt.loc4_10.1: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic = %y.patt.loc4_10.2 (constants.%y.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y.loc4_10.2: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_10.1 (constants.%y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.6b552a.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.522)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.522) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.8ef)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.4dfb92.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int [symbolic = %pattern_type (constants.%pattern_type.142cb7.1)]
// CHECK:STDOUT:   %start.patt: @IntRange.%pattern_type (%pattern_type.142cb7.1) = ref_binding_pattern start [symbolic = %start.patt (constants.%start.patt.477)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.bed)]
// CHECK:STDOUT:   %end.patt: @IntRange.%pattern_type (%pattern_type.142cb7.1) = ref_binding_pattern end [symbolic = %end.patt (constants.%end.patt.535)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.67af24.1), .end: @IntRange.%Int (%Int.67af24.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.d4d)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.e06
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.663
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.ac1
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.d41
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.60f
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Read(%y.loc4_10.2: Core.IntLiteral) {
// CHECK:STDOUT:   %y.patt.loc4_10.2: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic = %y.patt.loc4_10.2 (constants.%y.patt)]
// CHECK:STDOUT:   %y.loc4_10.1: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_10.1 (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y.loc4_10.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_38.3: <bound method> = bound_method %y.loc4_10.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2: init %i32 = call %bound_method.loc5_38.3(%y.loc4_10.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.0f3 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.0f3 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref %IntRange.bbd = var %x.var_patt
// CHECK:STDOUT:     %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:     %y.ref: Core.IntLiteral = name_ref y, %y.loc4_10.2 [symbolic = %y.loc4_10.1 (constants.%y)]
// CHECK:STDOUT:     %.loc5_3: ref %IntRange.bbd = splice_block %x.var {}
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc5_38.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_38.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1: init %i32 = call %bound_method.loc5_38.2(%y.ref) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.2: %i32 = converted %y.ref, %.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Range.call: init %IntRange.bbd to %.loc5_3 = call %Range.ref(%.loc5_38.2)
// CHECK:STDOUT:     assign %x.var, %Range.call
// CHECK:STDOUT:     %.loc5_28: type = splice_block %IntRange [concrete = constants.%IntRange.bbd] {
// CHECK:STDOUT:       %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.bbd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %IntRange.bbd = ref_binding x, %x.var
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.6b552a.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.67af24.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.142cb7.1)]
// CHECK:STDOUT:   %start.param_patt: @IntRange.Make.%pattern_type.1 (%pattern_type.142cb7.1) = value_param_pattern [symbolic = %start.param_patt (constants.%start.param_patt)]
// CHECK:STDOUT:   %start.patt: @IntRange.Make.%pattern_type.1 (%pattern_type.142cb7.1) = at_binding_pattern start, %start.param_patt [symbolic = %start.patt (constants.%start.patt.9ef)]
// CHECK:STDOUT:   %end.param_patt: @IntRange.Make.%pattern_type.1 (%pattern_type.142cb7.1) = value_param_pattern [symbolic = %end.param_patt (constants.%end.param_patt.8ab)]
// CHECK:STDOUT:   %end.patt: @IntRange.Make.%pattern_type.1 (%pattern_type.142cb7.1) = at_binding_pattern end, %end.param_patt [symbolic = %end.patt (constants.%end.patt.960)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.2c4)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %IntRange [symbolic = %.1 (constants.%.5c3)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.1ea)]
// CHECK:STDOUT:   %return.param_patt: @IntRange.Make.%pattern_type.2 (%pattern_type.1ea) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.262)]
// CHECK:STDOUT:   %return.patt: @IntRange.Make.%pattern_type.2 (%pattern_type.1ea) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.ca9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %Int [symbolic = %require_complete.1 (constants.%require_complete.4dfb92.1)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %IntRange [symbolic = %require_complete.2 (constants.%require_complete.5d9)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.67af24.1), .end: @IntRange.Make.%Int (%Int.67af24.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.d4d)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.2 (constants.%.a53)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.df6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.b83)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.3 (constants.%.c60)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.3 (%.c60) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.284)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.f9f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.3(%self.param: ref %struct_type.start.end.e93) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.4(%self.param: ref %IntRange.bbd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Read(constants.%y) {
// CHECK:STDOUT:   %y.patt.loc4_10.2 => constants.%y.patt
// CHECK:STDOUT:   %y.loc4_10.1 => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.522
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.8ef
// CHECK:STDOUT:   %Int => constants.%Int.67af24.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4dfb92.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.142cb7.1
// CHECK:STDOUT:   %start.patt => constants.%start.patt.477
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.bed
// CHECK:STDOUT:   %end.patt => constants.%end.patt.535
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d4d
// CHECK:STDOUT:   %complete_type => constants.%complete_type.aa1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.67af24.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.142cb7.1
// CHECK:STDOUT:   %start.param_patt => constants.%start.param_patt
// CHECK:STDOUT:   %start.patt => constants.%start.patt.9ef
// CHECK:STDOUT:   %end.param_patt => constants.%end.param_patt.8ab
// CHECK:STDOUT:   %end.patt => constants.%end.patt.960
// CHECK:STDOUT:   %IntRange => constants.%IntRange.2c4
// CHECK:STDOUT:   %.1 => constants.%.5c3
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.1ea
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.262
// CHECK:STDOUT:   %return.patt => constants.%return.patt.ca9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.d27
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.9ef
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %start.patt => constants.%start.patt.c23
// CHECK:STDOUT:   %IntRange => constants.%IntRange.bbd
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.6b1
// CHECK:STDOUT:   %end.patt => constants.%end.patt.46f
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.e93
// CHECK:STDOUT:   %complete_type => constants.%complete_type.ec7
// CHECK:STDOUT: }
// CHECK:STDOUT:

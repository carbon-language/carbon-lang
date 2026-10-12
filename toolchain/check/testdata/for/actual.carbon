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
// CHECK:STDOUT:   %IntRange.872: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.941e2b.1: type = pattern_type %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %start.param_patt.060: %pattern_type.941e2b.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %start.patt.8e2: %pattern_type.941e2b.1 = wrapper_binding_pattern start, %start.param_patt.060 [symbolic]
// CHECK:STDOUT:   %end.param_patt.406: %pattern_type.941e2b.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %end.patt.ddb: %pattern_type.941e2b.1 = wrapper_binding_pattern end, %end.param_patt.406 [symbolic]
// CHECK:STDOUT:   %.45d: Core.Form = init_form %IntRange.872 [symbolic]
// CHECK:STDOUT:   %pattern_type.2dd: type = pattern_type %IntRange.872 [symbolic]
// CHECK:STDOUT:   %return.param_patt.ef5: %pattern_type.2dd = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.ab0: %pattern_type.2dd = return_slot_pattern %return.param_patt.ef5, %IntRange.872 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.5ac: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.b41: %IntRange.Make.type.5ac = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.025: type = fn_type @Optional.Some, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Some.bde: %Optional.Some.type.025 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.None.type.d71: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.d97: %Optional.None.type.d71 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.frozen.bdd: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.bdd [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.c91: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.dd9 [concrete]
// CHECK:STDOUT:   %.052: <witness> = impl_self_witness %.Self.frozen.bdd, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.a93: %Destroy.type = impl_witness_access %.052, element1 [symbolic_self]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.38c: <witness> = lookup_impl_witness %Int.682b94.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1ae: %Destroy.type = facet_value %Int.682b94.1, (%Destroy.lookup_impl_witness.38c) [symbolic]
// CHECK:STDOUT:   %assoc0.0f0: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.ebd [concrete]
// CHECK:STDOUT:   %impl.elem0.3b7: %facet_type.f48 = impl_witness_access %.052, element0 [symbolic_self]
// CHECK:STDOUT:   %require_complete.aec7ec.1: <witness> = require_complete_type %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %return.param_patt.5c3fbb.1: %pattern_type.941e2b.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %.6c5f93.1: Core.Form = init_form %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.12f: <witness> = lookup_impl_witness %Int.682b94.1, @Copy [symbolic]
// CHECK:STDOUT:   %.7ff: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.f48 = facet_value %Int.682b94.1, (%Destroy.lookup_impl_witness.38c, %Copy.lookup_impl_witness.12f) [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.327: <witness> = impl_self_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.4a6: %Destroy.type = impl_witness_access %.327, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.188: %facet_type.f48 = impl_witness_access %.327, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.4a6 = %Destroy.facet.1ae and %impl.elem0.188 = %facet_value> [symbolic]
// CHECK:STDOUT:   %.60d: <witness> = impl_self_witness %IntRange.872, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.as.Iterate.impl.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %self.param_patt.9f4: %pattern_type.2dd = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.c27: %pattern_type.2dd = wrapper_binding_pattern self, %self.param_patt.9f4 [symbolic]
// CHECK:STDOUT:   %return.patt.a76: %pattern_type.941e2b.1 = return_slot_pattern %return.param_patt.5c3fbb.1, %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %cursor.param_patt.509: %pattern_type.941e2b.1 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %cursor.patt.cb0: %pattern_type.941e2b.1 = wrapper_binding_pattern cursor, %cursor.param_patt.509 [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.7dc: <witness> = lookup_impl_witness %Int.682b94.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %.e43: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.ed0: %OptionalStorage.type = facet_value %Int.682b94.1, (%OptionalStorage.lookup_impl_witness.7dc) [symbolic]
// CHECK:STDOUT:   %Optional.3e0: type = class_type @Optional, @Optional(%OptionalStorage.facet.ed0) [symbolic]
// CHECK:STDOUT:   %.a29: Core.Form = init_form %Optional.3e0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b05: type = pattern_type %Optional.3e0 [symbolic]
// CHECK:STDOUT:   %return.param_patt.546: %pattern_type.b05 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.56c: %pattern_type.b05 = return_slot_pattern %return.param_patt.546, %Optional.3e0 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IntRange.elem.850: type = unbound_element_type %IntRange.872, %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.9da: type = struct_type {.start: %Int.682b94.1, .end: %Int.682b94.1} [symbolic]
// CHECK:STDOUT:   %complete_type.3bd: <witness> = complete_type_witness %struct_type.start.end.9da [symbolic]
// CHECK:STDOUT:   %require_complete.1ff: <witness> = require_complete_type %IntRange.872 [symbolic]
// CHECK:STDOUT:   %Copy.facet.8a5: %Copy.type = facet_value %Int.682b94.1, (%Copy.lookup_impl_witness.12f) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c2e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.8a5) [symbolic]
// CHECK:STDOUT:   %.04e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c2e, %Copy.facet.8a5 [symbolic]
// CHECK:STDOUT:   %impl.elem0.7cc: %.04e = impl_witness_access %Copy.lookup_impl_witness.12f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dfb: <specific function> = specific_impl_function %impl.elem0.7cc, @Copy.WithSelf.Op(%Copy.facet.8a5) [symbolic]
// CHECK:STDOUT:   %Optional.None.type.459: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.ed0) [symbolic]
// CHECK:STDOUT:   %Optional.None.3a0: %Optional.None.type.459 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.1be: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.ed0) [symbolic]
// CHECK:STDOUT:   %Optional.Some.2d0: %Optional.Some.type.1be = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.ff9: <witness> = require_complete_type %Optional.3e0 [symbolic]
// CHECK:STDOUT:   %value.patt.a5e: %pattern_type.941e2b.1 = ref_binding_pattern value [symbolic]
// CHECK:STDOUT:   %value.var_patt: %pattern_type.941e2b.1 = var_pattern %value.patt.a5e [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ad8: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.ad8 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32a: type = facet_type <@OrderedWith, @OrderedWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.7df: %OrderedWith.type.32a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.7df) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.9e1: %OrderedWith.WithSelf.Less.type.b87 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.072: type = facet_type <@OrderedWith, @OrderedWith(%Int.682b94.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2d4: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.682b94.1) [symbolic]
// CHECK:STDOUT:   %assoc0.37d: %OrderedWith.assoc_type.2d4 = assoc_entity element0, imports.%Core.import_ref.e7a5 [symbolic]
// CHECK:STDOUT:   %require_complete.b85: <witness> = require_complete_type %OrderedWith.type.072 [symbolic]
// CHECK:STDOUT:   %assoc0.13c: %OrderedWith.assoc_type.1d2 = assoc_entity element0, imports.%Core.import_ref.447c [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.b4c: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.53b(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.229: %Int.as.OrderedWith.impl.Less.type.b4c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.898: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.25e, @Int.as.OrderedWith.impl.53b(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.c5d: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.53b(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.d0a: %Int.as.OrderedWith.impl.Less.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %.68d: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.53b(%N, %N) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet.3c1: %OrderedWith.type.072 = facet_value %Int.682b94.1, (%OrderedWith.impl_witness.898) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b28: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.682b94.1, %OrderedWith.facet.3c1) [symbolic]
// CHECK:STDOUT:   %.0c0: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b28, %OrderedWith.facet.3c1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.0ba: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.d0a, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.682b94.1, @Inc [symbolic]
// CHECK:STDOUT:   %.1d9: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.682b94.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.f1e: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %.07e: type = fn_type_with_self_type %Inc.WithSelf.Op.type.f1e, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.b82: %.07e = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7cc: <specific function> = specific_impl_function %impl.elem0.b82, @Inc.WithSelf.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.2d0, @Optional.Some(%OptionalStorage.facet.ed0) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.2b7: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.1ae) [symbolic]
// CHECK:STDOUT:   %.860: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.2b7, %Destroy.facet.1ae [symbolic]
// CHECK:STDOUT:   %impl.elem2.516: %.860 = impl_witness_access %Destroy.lookup_impl_witness.38c, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ee6: <specific function> = specific_impl_function %impl.elem2.516, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.1ae) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.3a0, @Optional.None(%OptionalStorage.facet.ed0) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %end.param_patt.23f: %pattern_type.df9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %end.patt.42e: %pattern_type.df9 = wrapper_binding_pattern end, %end.param_patt.23f [concrete]
// CHECK:STDOUT:   %IntRange.716: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %.1e6: Core.Form = init_form %IntRange.716 [concrete]
// CHECK:STDOUT:   %pattern_type.130: type = pattern_type %IntRange.716 [concrete]
// CHECK:STDOUT:   %return.param_patt.969: %pattern_type.130 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.026: %pattern_type.130 = return_slot_pattern %return.param_patt.969, %IntRange.716 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.65e: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.23f: %IntRange.Make.type.65e = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.elem.f91: type = unbound_element_type %IntRange.716, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.74c: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ec9: <witness> = complete_type_witness %struct_type.start.end.74c [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %start.param_patt.214: %pattern_type.df9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %start.patt.6cd: %pattern_type.df9 = wrapper_binding_pattern start, %start.param_patt.214 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.23f, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.604: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.604) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet.604 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %.417: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.6e9: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet.6e9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0fb: @Optional.%Optional.None.type (%Optional.None.type.d71) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.d97)]
// CHECK:STDOUT:   %Core.import_ref.4c3: @Optional.%Optional.Some.type (%Optional.Some.type.025) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.bde)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: %Destroy.type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.ebd: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %facet_type.f48 = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Copy.e5d: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.ad8 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.ef7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.13c)]
// CHECK:STDOUT:   %Core.import_ref.e7a5: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.b87) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.9e1)]
// CHECK:STDOUT:   %Core.import_ref.447c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9ae: @Int.as.OrderedWith.impl.53b.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.b4c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.53b.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.229)]
// CHECK:STDOUT:   %Core.import_ref.9e7 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b9c = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a78 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.25e = impl_witness_table (%Core.import_ref.9ae, %Core.import_ref.9e7, %Core.import_ref.b9c, %Core.import_ref.a78), @Int.as.OrderedWith.impl.53b [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
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
// CHECK:STDOUT:     %end.param_patt: %pattern_type.df9 = value_param_pattern [concrete = constants.%end.param_patt.23f]
// CHECK:STDOUT:     %end.patt: %pattern_type.df9 = wrapper_binding_pattern end, %end.param_patt [concrete = constants.%end.patt.42e]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.130 = out_param_pattern [concrete = constants.%return.param_patt.969]
// CHECK:STDOUT:     %return.patt: %pattern_type.130 = return_slot_pattern %return.param_patt, %IntRange.loc26 [concrete = constants.%return.patt.026]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc26: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc26: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.716]
// CHECK:STDOUT:     %.loc26_34.2: Core.Form = init_form %IntRange.loc26 [concrete = constants.%.1e6]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %end: %i32 = wrapper_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.716 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.716 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.38c)]
// CHECK:STDOUT:   %Destroy.facet.loc9_54.1: %Destroy.type = facet_value %Int.loc9_54.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:   %.loc9_85.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc9_85.1 (constants.%.7ff)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.12f)]
// CHECK:STDOUT:   %facet_value.loc9_85.1: %facet_type.f48 = facet_value %Int.loc9_54.1, (%Destroy.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.4a6 = %Destroy.facet.loc9_54.1 and constants.%impl.elem0.188 = %facet_value.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %.loc9_87: <witness> = impl_self_witness %IntRange, @Iterate [symbolic = %.loc9_87 (constants.%.60d)]
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
// CHECK:STDOUT:       %self.param_patt.loc10_18.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.2dd) = value_param_pattern [symbolic = %self.param_patt.loc10_18.2 (constants.%self.param_patt.9f4)]
// CHECK:STDOUT:       %self.patt.loc10_18.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.2dd) = wrapper_binding_pattern self, %self.param_patt.loc10_18.1 [symbolic = %self.patt.loc10_18.2 (constants.%self.patt.c27)]
// CHECK:STDOUT:       %return.param_patt.loc10_37.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.941e2b.1) = out_param_pattern [symbolic = %return.param_patt.loc10_37.2 (constants.%return.param_patt.5c3fbb.1)]
// CHECK:STDOUT:       %return.patt.loc10_24.1: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.941e2b.1) = return_slot_pattern %return.param_patt.loc10_37.1, %Int.loc10_37.2 [symbolic = %return.patt.loc10_24.2 (constants.%return.patt.a76)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_37.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_37.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       %.loc10_37.2: Core.Form = init_form %Int.loc10_37.2 [symbolic = %.loc10_37.1 (constants.%.6c5f93.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.872) = value_param call_param0
// CHECK:STDOUT:       %.loc10_18.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.872)] {
// CHECK:STDOUT:         %.loc10_18.2: type = specific_constant constants.%IntRange.872, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_18.2 [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.872) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.682b94.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.682b94.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.param_patt.loc11_13.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.2dd) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.9f4)]
// CHECK:STDOUT:       %self.patt.loc11_13.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.2dd) = wrapper_binding_pattern self, %self.param_patt.loc11_13.1 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.c27)]
// CHECK:STDOUT:       %cursor.param_patt.loc11_29.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = ref_param_pattern [symbolic = %cursor.param_patt.loc11_29.2 (constants.%cursor.param_patt.509)]
// CHECK:STDOUT:       %cursor.patt.loc11_29.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = wrapper_binding_pattern cursor, %cursor.param_patt.loc11_29.1 [symbolic = %cursor.patt.loc11_29.2 (constants.%cursor.patt.cb0)]
// CHECK:STDOUT:       %return.param_patt.loc11_72.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.b05) = out_param_pattern [symbolic = %return.param_patt.loc11_72.2 (constants.%return.param_patt.546)]
// CHECK:STDOUT:       %return.patt.loc11_44.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.b05) = return_slot_pattern %return.param_patt.loc11_72.1, %Optional.loc11_72.2 [symbolic = %return.patt.loc11_44.2 (constants.%return.patt.56c)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_47: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_61: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_65: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_70: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_71: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_72.2: %OptionalStorage.type = facet_value %Int.loc11_71, (constants.%OptionalStorage.lookup_impl_witness.7dc) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:       %.loc11_72.5: %OptionalStorage.type = converted %Int.loc11_71, %OptionalStorage.facet.loc11_72.2 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:       %Optional.loc11_72.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.loc11_72.1 (constants.%Optional.3e0)]
// CHECK:STDOUT:       %.loc11_72.6: Core.Form = init_form %Optional.loc11_72.2 [symbolic = %.loc11_72.4 (constants.%.a29)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.872) = value_param call_param0
// CHECK:STDOUT:       %.loc11_13.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.872)] {
// CHECK:STDOUT:         %.loc11_13.2: type = specific_constant constants.%IntRange.872, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_13.2 [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.872) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = ref_param call_param1
// CHECK:STDOUT:       %.loc11_41: type = splice_block %Int.loc11_41.2 [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)] {
// CHECK:STDOUT:         %Core.ref.loc11_31: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_35: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_40: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_41.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = wrapper_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, %IntRange.as.Iterate.impl.NewCursor.decl, %IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness.loc9_87.1: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: %Destroy.type = impl_witness_assoc_constant constants.%Destroy.facet.1ae [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %facet_type.f48 = impl_witness_assoc_constant constants.%facet_value [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
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
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_17.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.5ac)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.5ac) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.b41)]
// CHECK:STDOUT:   %Int.loc22_32.2: type = class_type @Int, @Int(%N.loc4_17.1) [symbolic = %Int.loc22_32.2 (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc22_32.2 [symbolic = %require_complete (constants.%require_complete.aec7ec.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_17.1) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc22_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.850)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int.loc22_32.2 (%Int.682b94.1), .end: @IntRange.%Int.loc22_32.2 (%Int.682b94.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.9da)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.3bd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.5ac) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.b41)] {
// CHECK:STDOUT:       %start.param_patt.loc5_16.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = value_param_pattern [symbolic = %start.param_patt.loc5_16.2 (constants.%start.param_patt.060)]
// CHECK:STDOUT:       %start.patt.loc5_16.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = wrapper_binding_pattern start, %start.param_patt.loc5_16.1 [symbolic = %start.patt.loc5_16.2 (constants.%start.patt.8e2)]
// CHECK:STDOUT:       %end.param_patt.loc5_34.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = value_param_pattern [symbolic = %end.param_patt.loc5_34.2 (constants.%end.param_patt.406)]
// CHECK:STDOUT:       %end.patt.loc5_34.1: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = wrapper_binding_pattern end, %end.param_patt.loc5_34.1 [symbolic = %end.patt.loc5_34.2 (constants.%end.patt.ddb)]
// CHECK:STDOUT:       %return.param_patt.loc5_52.1: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.2dd) = out_param_pattern [symbolic = %return.param_patt.loc5_52.2 (constants.%return.param_patt.ef5)]
// CHECK:STDOUT:       %return.patt.loc5_49.1: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.2dd) = return_slot_pattern %return.param_patt.loc5_52.1, %Self.ref [symbolic = %return.patt.loc5_49.2 (constants.%return.patt.ab0)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52.2: type = specific_constant constants.%IntRange.872, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52.2 [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:       %.loc5_52.3: Core.Form = init_form %Self.ref [symbolic = %.loc5_52.1 (constants.%.45d)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.682b94.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = wrapper_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.682b94.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = wrapper_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.872) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.872) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.872 [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self.frozen: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.bdd]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Iterate.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.bdd]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.5be [concrete = constants.%assoc1.c91]
// CHECK:STDOUT:       %impl.elem1.loc9_30.1: %Destroy.type = impl_witness_access constants.%.052, element1 [symbolic_self = constants.%impl.elem1.a93]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       %Destroy.facet.loc9_54.2: %Destroy.type = facet_value %Int.loc9_54.2, (constants.%Destroy.lookup_impl_witness.38c) [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:       %.loc9_54: %Destroy.type = converted %Int.loc9_54.2, %Destroy.facet.loc9_54.2 [symbolic = %Destroy.facet.loc9_54.1 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:       %rewrite.loc9_42.1 = requirement_rewrite %impl.elem1.loc9_30.1, %.loc9_54 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.bdd]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.96b [concrete = constants.%assoc0.0f0]
// CHECK:STDOUT:       %impl.elem0.loc9_60.1: %facet_type.f48 = impl_witness_access constants.%.052, element0 [symbolic_self = constants.%impl.elem0.3b7]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:       %facet_value.loc9_85.2: %facet_type.f48 = facet_value %Int.loc9_85, (constants.%Destroy.lookup_impl_witness.38c, constants.%Copy.lookup_impl_witness.12f) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_85.2: %facet_type.f48 = converted %Int.loc9_85, %facet_value.loc9_85.2 [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %rewrite.loc9_73.1 = requirement_rewrite %impl.elem0.loc9_60.1, %.loc9_85.2 [concrete]
// CHECK:STDOUT:       %impl.elem1.loc9_30.2: %Destroy.type = impl_witness_access constants.%.327, element1 [symbolic_self = constants.%impl.elem1.4a6]
// CHECK:STDOUT:       %rewrite.loc9_42.2 = requirement_rewrite %impl.elem1.loc9_30.2, %.loc9_54 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc9_60.2: %facet_type.f48 = impl_witness_access constants.%.327, element0 [symbolic_self = constants.%impl.elem0.188]
// CHECK:STDOUT:       %rewrite.loc9_73.2 = requirement_rewrite %impl.elem0.loc9_60.2, %.loc9_85.2 [concrete]
// CHECK:STDOUT:       %.loc9_24: type = where_expr [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Iterate.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc9_42.2 = requirement_rewrite %impl.elem1.loc9_30.2, %.loc9_54 [concrete]
// CHECK:STDOUT:         %rewrite.loc9_73.2 = requirement_rewrite %impl.elem0.loc9_60.2, %.loc9_85.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: <witness> = impl_self_witness @IntRange.as.Iterate.impl.%Self.ref, @Iterate [symbolic = @IntRange.as.Iterate.impl.%.loc9_87 (constants.%.60d)]
// CHECK:STDOUT:     %field_decl.loc22: @IntRange.%IntRange.elem (%IntRange.elem.850) = field_decl start, element0, %Int.loc22_32.1 in [concrete] {
// CHECK:STDOUT:       %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc22: Core.IntLiteral = name_ref N, %N.loc4_17.2 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc22_32.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.682b94.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %field_decl.loc23: @IntRange.%IntRange.elem (%IntRange.elem.850) = field_decl end, element1, %Int.loc23 in [concrete] {
// CHECK:STDOUT:       %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc23: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc23: Core.IntLiteral = name_ref N, %N.loc4_17.2 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc23: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.682b94.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.9da [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.3bd)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.872
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %field_decl.loc22
// CHECK:STDOUT:     .end [private] = %field_decl.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.941e2b.1)]
// CHECK:STDOUT:   %start.param_patt.loc5_16.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = value_param_pattern [symbolic = %start.param_patt.loc5_16.2 (constants.%start.param_patt.060)]
// CHECK:STDOUT:   %start.patt.loc5_16.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = wrapper_binding_pattern start, %start.param_patt.loc5_16.2 [symbolic = %start.patt.loc5_16.2 (constants.%start.patt.8e2)]
// CHECK:STDOUT:   %end.param_patt.loc5_34.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = value_param_pattern [symbolic = %end.param_patt.loc5_34.2 (constants.%end.param_patt.406)]
// CHECK:STDOUT:   %end.patt.loc5_34.2: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.941e2b.1) = wrapper_binding_pattern end, %end.param_patt.loc5_34.2 [symbolic = %end.patt.loc5_34.2 (constants.%end.patt.ddb)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %.loc5_52.1: Core.Form = init_form %IntRange [symbolic = %.loc5_52.1 (constants.%.45d)]
// CHECK:STDOUT:   %pattern_type.loc5_52: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_52 (constants.%pattern_type.2dd)]
// CHECK:STDOUT:   %return.param_patt.loc5_52.2: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.2dd) = out_param_pattern [symbolic = %return.param_patt.loc5_52.2 (constants.%return.param_patt.ef5)]
// CHECK:STDOUT:   %return.patt.loc5_49.2: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.2dd) = return_slot_pattern %return.param_patt.loc5_52.2, %IntRange [symbolic = %return.patt.loc5_49.2 (constants.%return.patt.ab0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.aec7ec.1)]
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.1ff)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.9da)]
// CHECK:STDOUT:   %.loc6_22.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc6_22.3 (constants.%.7ff)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.12f)]
// CHECK:STDOUT:   %Copy.facet.loc6_22.3: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc6_22.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c2e)]
// CHECK:STDOUT:   %.loc6_22.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc6_22.3 [symbolic = %.loc6_22.4 (constants.%.04e)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.4 (%.04e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.WithSelf.Op(%Copy.facet.loc6_22.3) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1)) -> out %return.param: @IntRange.Make.%IntRange (%IntRange.872) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.9da) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.4 (%.04e) = impl_witness_access constants.%Copy.lookup_impl_witness.12f, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %Copy.facet.loc6_22.1: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc6_22.1: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc6_22.1 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %Copy.facet.loc6_22.2: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc6_22.2: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc6_22.2 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.WithSelf.Op(constants.%Copy.facet.8a5) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_22: init @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = class_element_access %return.param, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) to %.loc6_39.2 = in_place_init %Copy.WithSelf.Op.call.loc6_22
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.4 (%.04e) = impl_witness_access constants.%Copy.lookup_impl_witness.12f, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %Copy.facet.loc6_36.1: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc6_36.1: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc6_36.1 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %Copy.facet.loc6_36.2: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc6_36.2: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc6_36.2 [symbolic = %Copy.facet.loc6_22.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.WithSelf.Op(constants.%Copy.facet.8a5) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) = class_element_access %return.param, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.682b94.1) to %.loc6_39.4 = in_place_init %Copy.WithSelf.Op.call.loc6_36
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.872) to %return.param = class_init (%.loc6_39.3, %.loc6_39.5)
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.872) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.2dd)]
// CHECK:STDOUT:   %self.param_patt.loc10_18.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.2dd) = value_param_pattern [symbolic = %self.param_patt.loc10_18.2 (constants.%self.param_patt.9f4)]
// CHECK:STDOUT:   %self.patt.loc10_18.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.2dd) = wrapper_binding_pattern self, %self.param_patt.loc10_18.2 [symbolic = %self.patt.loc10_18.2 (constants.%self.patt.c27)]
// CHECK:STDOUT:   %Int.loc10_37.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_37.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %.loc10_37.1: Core.Form = init_form %Int.loc10_37.1 [symbolic = %.loc10_37.1 (constants.%.6c5f93.1)]
// CHECK:STDOUT:   %pattern_type.loc10_37: type = pattern_type %Int.loc10_37.1 [symbolic = %pattern_type.loc10_37 (constants.%pattern_type.941e2b.1)]
// CHECK:STDOUT:   %return.param_patt.loc10_37.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.941e2b.1) = out_param_pattern [symbolic = %return.param_patt.loc10_37.2 (constants.%return.param_patt.5c3fbb.1)]
// CHECK:STDOUT:   %return.patt.loc10_24.2: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_37 (%pattern_type.941e2b.1) = return_slot_pattern %return.param_patt.loc10_37.2, %Int.loc10_37.1 [symbolic = %return.patt.loc10_24.2 (constants.%return.patt.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_18: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_18 (constants.%require_complete.1ff)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_37.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.850)]
// CHECK:STDOUT:   %require_complete.loc10_52: <witness> = require_complete_type %Int.loc10_37.1 [symbolic = %require_complete.loc10_52 (constants.%require_complete.aec7ec.1)]
// CHECK:STDOUT:   %.loc10_52.5: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc10_52.5 (constants.%.7ff)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc10_37.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.12f)]
// CHECK:STDOUT:   %Copy.facet.loc10_52.3: %Copy.type = facet_value %Int.loc10_37.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_52.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c2e)]
// CHECK:STDOUT:   %.loc10_52.6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_52.3 [symbolic = %.loc10_52.6 (constants.%.04e)]
// CHECK:STDOUT:   %impl.elem0.loc10_52.2: @IntRange.as.Iterate.impl.NewCursor.%.loc10_52.6 (%.04e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_52.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_52.2: <specific function> = specific_impl_function %impl.elem0.loc10_52.2, @Copy.WithSelf.Op(%Copy.facet.loc10_52.3) [symbolic = %specific_impl_fn.loc10_52.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.872)) -> out %return.param: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.682b94.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.872) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.850) = name_ref start, @IntRange.%field_decl.loc22 [concrete = @IntRange.%field_decl.loc22]
// CHECK:STDOUT:     %.loc10_52.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.682b94.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_52.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.682b94.1) = acquire_value %.loc10_52.1
// CHECK:STDOUT:     %impl.elem0.loc10_52.1: @IntRange.as.Iterate.impl.NewCursor.%.loc10_52.6 (%.04e) = impl_witness_access constants.%Copy.lookup_impl_witness.12f, element0 [symbolic = %impl.elem0.loc10_52.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:     %bound_method.loc10_52.1: <bound method> = bound_method %.loc10_52.2, %impl.elem0.loc10_52.1
// CHECK:STDOUT:     %Copy.facet.loc10_52.1: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc10_52.3: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc10_52.1 [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %Copy.facet.loc10_52.2: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc10_52.4: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc10_52.2 [symbolic = %Copy.facet.loc10_52.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_52.1: <specific function> = specific_impl_function %impl.elem0.loc10_52.1, @Copy.WithSelf.Op(constants.%Copy.facet.8a5) [symbolic = %specific_impl_fn.loc10_52.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:     %bound_method.loc10_52.2: <bound method> = bound_method %.loc10_52.2, %specific_impl_fn.loc10_52.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_37.1 (%Int.682b94.1) = call %bound_method.loc10_52.2(%.loc10_52.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.2dd)]
// CHECK:STDOUT:   %self.param_patt.loc11_13.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.2dd) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.9f4)]
// CHECK:STDOUT:   %self.patt.loc11_13.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.2dd) = wrapper_binding_pattern self, %self.param_patt.loc11_13.2 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.c27)]
// CHECK:STDOUT:   %Int.loc11_41.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %pattern_type.loc11_29: type = pattern_type %Int.loc11_41.1 [symbolic = %pattern_type.loc11_29 (constants.%pattern_type.941e2b.1)]
// CHECK:STDOUT:   %cursor.param_patt.loc11_29.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = ref_param_pattern [symbolic = %cursor.param_patt.loc11_29.2 (constants.%cursor.param_patt.509)]
// CHECK:STDOUT:   %cursor.patt.loc11_29.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = wrapper_binding_pattern cursor, %cursor.param_patt.loc11_29.2 [symbolic = %cursor.patt.loc11_29.2 (constants.%cursor.patt.cb0)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.38c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.12f)]
// CHECK:STDOUT:   %facet_value: %facet_type.f48 = facet_value %Int.loc11_41.1, (%Destroy.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %.loc11_72.3: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value) [symbolic = %.loc11_72.3 (constants.%.e43)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.7dc)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_72.1: %OptionalStorage.type = facet_value %Int.loc11_41.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:   %Optional.loc11_72.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.loc11_72.1 (constants.%Optional.3e0)]
// CHECK:STDOUT:   %.loc11_72.4: Core.Form = init_form %Optional.loc11_72.1 [symbolic = %.loc11_72.4 (constants.%.a29)]
// CHECK:STDOUT:   %pattern_type.loc11_72: type = pattern_type %Optional.loc11_72.1 [symbolic = %pattern_type.loc11_72 (constants.%pattern_type.b05)]
// CHECK:STDOUT:   %return.param_patt.loc11_72.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.b05) = out_param_pattern [symbolic = %return.param_patt.loc11_72.2 (constants.%return.param_patt.546)]
// CHECK:STDOUT:   %return.patt.loc11_44.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_72 (%pattern_type.b05) = return_slot_pattern %return.param_patt.loc11_72.2, %Optional.loc11_72.1 [symbolic = %return.patt.loc11_44.2 (constants.%return.patt.56c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_13: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_13 (constants.%require_complete.1ff)]
// CHECK:STDOUT:   %require_complete.loc11_29: <witness> = require_complete_type %Int.loc11_41.1 [symbolic = %require_complete.loc11_29 (constants.%require_complete.aec7ec.1)]
// CHECK:STDOUT:   %require_complete.loc11_44: <witness> = require_complete_type %Optional.loc11_72.1 [symbolic = %require_complete.loc11_44 (constants.%require_complete.ff9)]
// CHECK:STDOUT:   %value.patt.loc12_16.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = ref_binding_pattern value [symbolic = %value.patt.loc12_16.2 (constants.%value.patt.a5e)]
// CHECK:STDOUT:   %value.var_patt.loc12_7.2: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = var_pattern %value.patt.loc12_16.2 [symbolic = %value.var_patt.loc12_7.2 (constants.%value.var_patt)]
// CHECK:STDOUT:   %.loc12_32.4: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc12_32.4 (constants.%.7ff)]
// CHECK:STDOUT:   %Copy.facet.loc12_32.3: %Copy.type = facet_value %Int.loc11_41.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc12_32.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c2e)]
// CHECK:STDOUT:   %.loc12_32.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc12_32.3 [symbolic = %.loc12_32.5 (constants.%.04e)]
// CHECK:STDOUT:   %impl.elem0.loc12_32.2: @IntRange.as.Iterate.impl.Next.%.loc12_32.5 (%.04e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_32.2: <specific function> = specific_impl_function %impl.elem0.loc12_32.2, @Copy.WithSelf.Op(%Copy.facet.loc12_32.3) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_41.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.850)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_41.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.072)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.b85)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_41.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2d4)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2d4) = assoc_entity element0, imports.%Core.import_ref.e7a5 [symbolic = %assoc0 (constants.%assoc0.37d)]
// CHECK:STDOUT:   %.loc13_17.2: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.53b(%N, %N) [symbolic = %.loc13_17.2 (constants.%.68d)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.25e, @Int.as.OrderedWith.impl.53b(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.898)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.072) = facet_value %Int.loc11_41.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet.3c1)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.loc11_41.1, %OrderedWith.facet) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.b28)]
// CHECK:STDOUT:   %.loc13_17.3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.3 (constants.%.0c0)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.53b(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.c5d)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.c5d) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.d0a)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ba)]
// CHECK:STDOUT:   %.loc14_9.3: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic = %.loc14_9.3 (constants.%.1d9)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_41.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet.loc14_9.3: %Inc.type = facet_value %Int.loc11_41.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.loc14_9.3) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.f1e)]
// CHECK:STDOUT:   %.loc14_9.4: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %Inc.facet.loc14_9.3 [symbolic = %.loc14_9.4 (constants.%.07e)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.4 (%.07e) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.b82)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.WithSelf.Op(%Inc.facet.loc14_9.3) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.7cc)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.1be)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.1be) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.2d0)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.facet.loc12_7.5: %Destroy.type = facet_value %Int.loc11_41.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.loc12_7.5) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.2b7)]
// CHECK:STDOUT:   %.loc12_7.5: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc12_7.5 [symbolic = %.loc12_7.5 (constants.%.860)]
// CHECK:STDOUT:   %impl.elem2.loc12_7.3: @IntRange.as.Iterate.impl.Next.%.loc12_7.5 (%.860) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc12_7.3 (constants.%impl.elem2.516)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_7.3: <specific function> = specific_impl_function %impl.elem2.loc12_7.3, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.loc12_7.5) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.ee6)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.459)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.459) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.3a0)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc11_72.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.872), %cursor.param: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1)) -> out %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = var_storage %value.var_patt.loc12_7.1
// CHECK:STDOUT:     %cursor.ref.loc12: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = acquire_value %cursor.ref.loc12
// CHECK:STDOUT:     %impl.elem0.loc12_32.1: @IntRange.as.Iterate.impl.Next.%.loc12_32.5 (%.04e) = impl_witness_access constants.%Copy.lookup_impl_witness.12f, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:     %bound_method.loc12_32.1: <bound method> = bound_method %.loc12_32.1, %impl.elem0.loc12_32.1
// CHECK:STDOUT:     %Copy.facet.loc12_32.1: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc12_32.2: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc12_32.1 [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %Copy.facet.loc12_32.2: %Copy.type = facet_value constants.%Int.682b94.1, (constants.%Copy.lookup_impl_witness.12f) [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %.loc12_32.3: %Copy.type = converted constants.%Int.682b94.1, %Copy.facet.loc12_32.2 [symbolic = %Copy.facet.loc12_32.3 (constants.%Copy.facet.8a5)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_32.1: <specific function> = specific_impl_function %impl.elem0.loc12_32.1, @Copy.WithSelf.Op(constants.%Copy.facet.8a5) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:     %bound_method.loc12_32.2: <bound method> = bound_method %.loc12_32.1, %specific_impl_fn.loc12_32.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = call %bound_method.loc12_32.2(%.loc12_32.1)
// CHECK:STDOUT:     assign %value.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = wrapper_binding value, %value.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt.loc12_16.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = ref_binding_pattern value [symbolic = %value.patt.loc12_16.2 (constants.%value.patt.a5e)]
// CHECK:STDOUT:       %value.var_patt.loc12_7.1: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_29 (%pattern_type.941e2b.1) = var_pattern %value.patt.loc12_16.1 [symbolic = %value.var_patt.loc12_7.2 (constants.%value.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.872) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.850) = name_ref end, @IntRange.%field_decl.loc23 [concrete = @IntRange.%field_decl.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = acquire_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.682b94.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.072)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2d4) = specific_constant imports.%Core.import_ref.ef7, @OrderedWith.WithSelf(constants.%Int.682b94.1, constants.%Self.7df) [symbolic = %assoc0 (constants.%assoc0.37d)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2d4) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.37d)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.3 (%.0c0) = impl_witness_access constants.%OrderedWith.impl_witness.898, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.d0a)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ba)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = acquire_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = name_ref cursor, %cursor
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.4 (%.07e) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.b82)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %cursor.ref.loc14, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %Inc.facet.loc14_9.1: %Inc.type = facet_value constants.%Int.682b94.1, (constants.%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %.loc14_9.1: %Inc.type = converted constants.%Int.682b94.1, %Inc.facet.loc14_9.1 [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %Inc.facet.loc14_9.2: %Inc.type = facet_value constants.%Int.682b94.1, (constants.%Inc.lookup_impl_witness) [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %.loc14_9.2: %Inc.type = converted constants.%Int.682b94.1, %Inc.facet.loc14_9.2 [symbolic = %Inc.facet.loc14_9.3 (constants.%Inc.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.WithSelf.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.7cc)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %cursor.ref.loc14, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %Inc.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_9.2(%cursor.ref.loc14)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_41: %OptionalStorage.type = facet_value %Int.loc15, (constants.%OptionalStorage.lookup_impl_witness.7dc) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %.loc15_41: %OptionalStorage.type = converted %Int.loc15, %OptionalStorage.facet.loc15_41 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.loc11_72.1 (constants.%Optional.3e0)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.1be) = specific_constant imports.%Core.import_ref.4c3, @Optional(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.Some (constants.%Optional.Some.2d0)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.1be) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.2d0)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.1: %OptionalStorage.type = facet_value constants.%Int.682b94.1, (constants.%OptionalStorage.lookup_impl_witness.7dc) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %.loc15_53.1: %OptionalStorage.type = converted constants.%Int.682b94.1, %OptionalStorage.facet.loc15_53.1 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.2: %OptionalStorage.type = facet_value constants.%Int.682b94.1, (constants.%OptionalStorage.lookup_impl_witness.7dc) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %.loc15_53.2: %OptionalStorage.type = converted constants.%Int.682b94.1, %OptionalStorage.facet.loc15_53.2 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_72.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_41.1 (%Int.682b94.1) = acquire_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) to %.loc11_72.1 = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48)
// CHECK:STDOUT:     %impl.elem2.loc12_7.1: @IntRange.as.Iterate.impl.Next.%.loc12_7.5 (%.860) = impl_witness_access constants.%Destroy.lookup_impl_witness.38c, element2 [symbolic = %impl.elem2.loc12_7.3 (constants.%impl.elem2.516)]
// CHECK:STDOUT:     %bound_method.loc12_7.1: <bound method> = bound_method %value.var, %impl.elem2.loc12_7.1
// CHECK:STDOUT:     %Destroy.facet.loc12_7.1: %Destroy.type = facet_value constants.%Int.682b94.1, (constants.%Destroy.lookup_impl_witness.38c) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %.loc12_7.1: %Destroy.type = converted constants.%Int.682b94.1, %Destroy.facet.loc12_7.1 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %Destroy.facet.loc12_7.2: %Destroy.type = facet_value constants.%Int.682b94.1, (constants.%Destroy.lookup_impl_witness.38c) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %.loc12_7.2: %Destroy.type = converted constants.%Int.682b94.1, %Destroy.facet.loc12_7.2 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.1: <specific function> = specific_impl_function %impl.elem2.loc12_7.1, @Destroy.WithSelf.SelfDestruct(constants.%Destroy.facet.1ae) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.ee6)]
// CHECK:STDOUT:     %bound_method.loc12_7.2: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc12_7.1: init %empty_tuple.type = call %bound_method.loc12_7.2(%value.var)
// CHECK:STDOUT:     return %Optional.Some.call to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_41.1 (constants.%Int.682b94.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc17: %OptionalStorage.type = facet_value %Int.loc17, (constants.%OptionalStorage.lookup_impl_witness.7dc) [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %.loc17_41: %OptionalStorage.type = converted %Int.loc17, %OptionalStorage.facet.loc17 [symbolic = %OptionalStorage.facet.loc11_72.1 (constants.%OptionalStorage.facet.ed0)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.loc11_72.1 (constants.%Optional.3e0)]
// CHECK:STDOUT:     %.loc17_42: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.459) = specific_constant imports.%Core.import_ref.0fb, @Optional(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.None (constants.%Optional.None.3a0)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.459) = name_ref None, %.loc17_42 [symbolic = %Optional.None (constants.%Optional.None.3a0)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.ed0) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_72.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) = splice_block %return.param {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_72.1 (%Optional.3e0) to %.loc11_72.2 = call %Optional.None.specific_fn.loc17_42.1()
// CHECK:STDOUT:     %impl.elem2.loc12_7.2: @IntRange.as.Iterate.impl.Next.%.loc12_7.5 (%.860) = impl_witness_access constants.%Destroy.lookup_impl_witness.38c, element2 [symbolic = %impl.elem2.loc12_7.3 (constants.%impl.elem2.516)]
// CHECK:STDOUT:     %bound_method.loc12_7.3: <bound method> = bound_method %value.var, %impl.elem2.loc12_7.2
// CHECK:STDOUT:     %Destroy.facet.loc12_7.3: %Destroy.type = facet_value constants.%Int.682b94.1, (constants.%Destroy.lookup_impl_witness.38c) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %.loc12_7.3: %Destroy.type = converted constants.%Int.682b94.1, %Destroy.facet.loc12_7.3 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %Destroy.facet.loc12_7.4: %Destroy.type = facet_value constants.%Int.682b94.1, (constants.%Destroy.lookup_impl_witness.38c) [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %.loc12_7.4: %Destroy.type = converted constants.%Int.682b94.1, %Destroy.facet.loc12_7.4 [symbolic = %Destroy.facet.loc12_7.5 (constants.%Destroy.facet.1ae)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.2: <specific function> = specific_impl_function %impl.elem2.loc12_7.2, @Destroy.WithSelf.SelfDestruct(constants.%Destroy.facet.1ae) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.ee6)]
// CHECK:STDOUT:     %bound_method.loc12_7.4: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc12_7.2: init %empty_tuple.type = call %bound_method.loc12_7.4(%value.var)
// CHECK:STDOUT:     return %Optional.None.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> out %return.param: %IntRange.716 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.716]
// CHECK:STDOUT:   %.loc27_22: %IntRange.Make.type.65e = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.23f]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.65e = name_ref Make, %.loc27_22 [concrete = constants.%IntRange.Make.23f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc26_34.1: ref %IntRange.716 = splice_block %return.param {}
// CHECK:STDOUT:   %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.716 to %.loc26_34.1 = call %IntRange.Make.specific_fn(%.loc27_28.2, %end.ref)
// CHECK:STDOUT:   return %IntRange.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc4_17.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.5ac
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.b41
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%Int.682b94.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.aec7ec.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.850
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.9da
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.3bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.682b94.1
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.941e2b.1
// CHECK:STDOUT:   %start.param_patt.loc5_16.2 => constants.%start.param_patt.060
// CHECK:STDOUT:   %start.patt.loc5_16.2 => constants.%start.patt.8e2
// CHECK:STDOUT:   %end.param_patt.loc5_34.2 => constants.%end.param_patt.406
// CHECK:STDOUT:   %end.patt.loc5_34.2 => constants.%end.patt.ddb
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.45d
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %return.param_patt.loc5_52.2 => constants.%return.param_patt.ef5
// CHECK:STDOUT:   %return.patt.loc5_49.2 => constants.%return.patt.ab0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.682b94.1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.38c
// CHECK:STDOUT:   %Destroy.facet.loc9_54.1 => constants.%Destroy.facet.1ae
// CHECK:STDOUT:   %.loc9_85.1 => constants.%.7ff
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.12f
// CHECK:STDOUT:   %facet_value.loc9_85.1 => constants.%facet_value
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %.loc9_87 => constants.%.60d
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
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %self.param_patt.loc10_18.2 => constants.%self.param_patt.9f4
// CHECK:STDOUT:   %self.patt.loc10_18.2 => constants.%self.patt.c27
// CHECK:STDOUT:   %Int.loc10_37.1 => constants.%Int.682b94.1
// CHECK:STDOUT:   %.loc10_37.1 => constants.%.6c5f93.1
// CHECK:STDOUT:   %pattern_type.loc10_37 => constants.%pattern_type.941e2b.1
// CHECK:STDOUT:   %return.param_patt.loc10_37.2 => constants.%return.param_patt.5c3fbb.1
// CHECK:STDOUT:   %return.patt.loc10_24.2 => constants.%return.patt.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %self.param_patt.loc11_13.2 => constants.%self.param_patt.9f4
// CHECK:STDOUT:   %self.patt.loc11_13.2 => constants.%self.patt.c27
// CHECK:STDOUT:   %Int.loc11_41.1 => constants.%Int.682b94.1
// CHECK:STDOUT:   %pattern_type.loc11_29 => constants.%pattern_type.941e2b.1
// CHECK:STDOUT:   %cursor.param_patt.loc11_29.2 => constants.%cursor.param_patt.509
// CHECK:STDOUT:   %cursor.patt.loc11_29.2 => constants.%cursor.patt.cb0
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.38c
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.12f
// CHECK:STDOUT:   %facet_value => constants.%facet_value
// CHECK:STDOUT:   %.loc11_72.3 => constants.%.e43
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.7dc
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_72.1 => constants.%OptionalStorage.facet.ed0
// CHECK:STDOUT:   %Optional.loc11_72.1 => constants.%Optional.3e0
// CHECK:STDOUT:   %.loc11_72.4 => constants.%.a29
// CHECK:STDOUT:   %pattern_type.loc11_72 => constants.%pattern_type.b05
// CHECK:STDOUT:   %return.param_patt.loc11_72.2 => constants.%return.param_patt.546
// CHECK:STDOUT:   %return.patt.loc11_44.2 => constants.%return.patt.56c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.patt.loc4_17.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.65e
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.23f
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.716
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.f91
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.74c
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.ec9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.df9
// CHECK:STDOUT:   %start.param_patt.loc5_16.2 => constants.%start.param_patt.214
// CHECK:STDOUT:   %start.patt.loc5_16.2 => constants.%start.patt.6cd
// CHECK:STDOUT:   %end.param_patt.loc5_34.2 => constants.%end.param_patt.23f
// CHECK:STDOUT:   %end.patt.loc5_34.2 => constants.%end.patt.42e
// CHECK:STDOUT:   %IntRange => constants.%IntRange.716
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.1e6
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.130
// CHECK:STDOUT:   %return.param_patt.loc5_52.2 => constants.%return.param_patt.969
// CHECK:STDOUT:   %return.patt.loc5_49.2 => constants.%return.patt.026
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc5_49 => constants.%complete_type.ec9
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.74c
// CHECK:STDOUT:   %.loc6_22.3 => constants.%.417
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.4c3
// CHECK:STDOUT:   %Copy.facet.loc6_22.3 => constants.%Copy.facet.6e9
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.67d
// CHECK:STDOUT:   %.loc6_22.4 => constants.%.187
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.175
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
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.9da: type = struct_type {.start: %Int.682b94.1, .end: %Int.682b94.1} [symbolic]
// CHECK:STDOUT:   %complete_type.3bd: <witness> = complete_type_witness %struct_type.start.end.9da [symbolic]
// CHECK:STDOUT:   %IntRange.872: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.elem.850: type = unbound_element_type %IntRange.872, %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %require_complete.aec7ec.1: <witness> = require_complete_type %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.5ac: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.b41: %IntRange.Make.type.5ac = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.2dd: type = pattern_type %IntRange.872 [symbolic]
// CHECK:STDOUT:   %return.param_patt.ef5: %pattern_type.2dd = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.2ac: %pattern_type.2dd = return_slot_pattern %return.param_patt.ef5, invalid [symbolic]
// CHECK:STDOUT:   %.45d: Core.Form = init_form %IntRange.872 [symbolic]
// CHECK:STDOUT:   %pattern_type.941e2b.1: type = pattern_type %Int.682b94.1 [symbolic]
// CHECK:STDOUT:   %end.param_patt.406: %pattern_type.941e2b.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %end.patt.ddb: %pattern_type.941e2b.1 = wrapper_binding_pattern end, %end.param_patt.406 [symbolic]
// CHECK:STDOUT:   %start.param_patt: %pattern_type.941e2b.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %start.patt: %pattern_type.941e2b.1 = wrapper_binding_pattern start, %start.param_patt [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.12f: <witness> = lookup_impl_witness %Int.682b94.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.682b94.1, (%Copy.lookup_impl_witness.12f) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c2e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.04e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c2e, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.7cc: %.04e = impl_witness_access %Copy.lookup_impl_witness.12f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dfb: <specific function> = specific_impl_function %impl.elem0.7cc, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.7ff: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.1ff: <witness> = require_complete_type %IntRange.872 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.716: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.65e: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.23f: %IntRange.Make.type.65e = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.elem.f91: type = unbound_element_type %IntRange.716, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.74c: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ec9: <witness> = complete_type_witness %struct_type.start.end.74c [concrete]
// CHECK:STDOUT:   %pattern_type.130: type = pattern_type %IntRange.716 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.130 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.130 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.59b: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc8: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt.59b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc5_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc5_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc5_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.2f5: type = pattern_type %struct_type.start.end.74c [concrete]
// CHECK:STDOUT:   %self.param_patt.575: %pattern_type.2f5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.439: %pattern_type.2f5 = wrapper_binding_pattern self, %self.param_patt.575 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc5_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc5_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.180: %pattern_type.130 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.781: %pattern_type.130 = wrapper_binding_pattern self, %self.param_patt.180 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc5_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc5_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc5_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.c9b: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.3bd)]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.9a0 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.a9e: @IntRange.%IntRange.elem (%IntRange.elem.850) = import_ref Main//lib, loc22_20, loaded [concrete = %field_decl.c45]
// CHECK:STDOUT:   %Main.import_ref.e9b: @IntRange.%IntRange.elem (%IntRange.elem.850) = import_ref Main//lib, loc23_18, loaded [concrete = %field_decl.a49]
// CHECK:STDOUT:   %Main.import_ref.6b552a.2: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.8d11d7.1 = import_ref Main//lib, loc23_30, unloaded
// CHECK:STDOUT:   %field_decl.a49: @IntRange.%IntRange.elem (%IntRange.elem.850) = field_decl end, element1, %Main.import_ref.8d11d7.1 in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.8d11d7.2 = import_ref Main//lib, loc22_32, unloaded
// CHECK:STDOUT:   %field_decl.c45: @IntRange.%IntRange.elem (%IntRange.elem.850) = field_decl start, element0, %Main.import_ref.8d11d7.2 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc5_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc5_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc5_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2f5 = ref_param_pattern [concrete = constants.%self.param_patt.575]
// CHECK:STDOUT:     %self.patt: %pattern_type.2f5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.439]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.start.end.74c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.start.end.74c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc5_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2f5 = ref_param_pattern [concrete = constants.%self.param_patt.575]
// CHECK:STDOUT:     %self.patt: %pattern_type.2f5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.439]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.start.end.74c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.start.end.74c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc5_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.130 = ref_param_pattern [concrete = constants.%self.param_patt.180]
// CHECK:STDOUT:     %self.patt: %pattern_type.130 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.781]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %IntRange.716 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %IntRange.716 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc5_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.130 = ref_param_pattern [concrete = constants.%self.param_patt.180]
// CHECK:STDOUT:     %self.patt: %pattern_type.130 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.781]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %IntRange.716 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %IntRange.716 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.5ac)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.5ac) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.b41)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.aec7ec.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.850)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.682b94.1), .end: @IntRange.%Int (%Int.682b94.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.9da)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.3bd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.c9b
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.f42
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.9a0
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.a9e
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.e9b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Read(%y.loc4_18.2: Core.IntLiteral) {
// CHECK:STDOUT:   %y.patt.loc4_18.2: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [symbolic = %y.patt.loc4_18.2 (constants.%y.patt)]
// CHECK:STDOUT:   %y.loc4_18.1: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_18.1 (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y.loc4_18.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_38.3: <bound method> = bound_method %y.loc4_18.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2: init %i32 = call %bound_method.loc5_38.3(%y.loc4_18.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.var: ref %IntRange.716 = var_storage %x.var_patt
// CHECK:STDOUT:     %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:     %y.ref: Core.IntLiteral = name_ref y, %y.loc4_18.2 [symbolic = %y.loc4_18.1 (constants.%y)]
// CHECK:STDOUT:     %.loc5_3: ref %IntRange.716 = splice_block %x.var {}
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc5_38.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_38.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1: init %i32 = call %bound_method.loc5_38.2(%y.ref) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.2: %i32 = converted %y.ref, %.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Range.call: init %IntRange.716 to %.loc5_3 = call %Range.ref(%.loc5_38.2)
// CHECK:STDOUT:     assign %x.var, %Range.call
// CHECK:STDOUT:     %.loc5_28: type = splice_block %IntRange [concrete = constants.%IntRange.716] {
// CHECK:STDOUT:       %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.716]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %IntRange.716 = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.130 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.130 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %x.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%x.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.6b552a.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.682b94.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.941e2b.1)]
// CHECK:STDOUT:   %start.param_patt: @IntRange.Make.%pattern_type.1 (%pattern_type.941e2b.1) = value_param_pattern [symbolic = %start.param_patt (constants.%start.param_patt)]
// CHECK:STDOUT:   %start.patt: @IntRange.Make.%pattern_type.1 (%pattern_type.941e2b.1) = wrapper_binding_pattern start, %start.param_patt [symbolic = %start.patt (constants.%start.patt)]
// CHECK:STDOUT:   %end.param_patt: @IntRange.Make.%pattern_type.1 (%pattern_type.941e2b.1) = value_param_pattern [symbolic = %end.param_patt (constants.%end.param_patt.406)]
// CHECK:STDOUT:   %end.patt: @IntRange.Make.%pattern_type.1 (%pattern_type.941e2b.1) = wrapper_binding_pattern end, %end.param_patt [symbolic = %end.patt (constants.%end.patt.ddb)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.872)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %IntRange [symbolic = %.1 (constants.%.45d)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.2dd)]
// CHECK:STDOUT:   %return.param_patt: @IntRange.Make.%pattern_type.2 (%pattern_type.2dd) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.ef5)]
// CHECK:STDOUT:   %return.patt: @IntRange.Make.%pattern_type.2 (%pattern_type.2dd) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %Int [symbolic = %require_complete.1 (constants.%require_complete.aec7ec.1)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %IntRange [symbolic = %require_complete.2 (constants.%require_complete.1ff)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.682b94.1), .end: @IntRange.Make.%Int (%Int.682b94.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.9da)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.2 (constants.%.7ff)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.12f)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c2e)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.3 (constants.%.04e)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.3 (%.04e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.7cc)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.dfb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc5_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc5_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc5_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc5_3.2(%self.param: ref %struct_type.start.end.74c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_3.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc5_3.1: <bound method> = bound_method %.loc5_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc5_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc5_3.1(%.loc5_3.1)
// CHECK:STDOUT:   %.loc5_3.2: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc5_3.2: <bound method> = bound_method %.loc5_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc5_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc5_3.2(%.loc5_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc5_3.2(%self.param: ref %struct_type.start.end.74c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc5_3.2(%self.param: ref %struct_type.start.end.74c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc5_3.3(%self.param: ref %IntRange.716) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %end.ref: %IntRange.elem.f91 = name_ref end, imports.%Main.import_ref.e9b [concrete = imports.%field_decl.a49]
// CHECK:STDOUT:   %.loc5_3.1: ref %i32 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc5_3.1: <bound method> = bound_method %.loc5_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc5_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc5_3.1(%.loc5_3.1)
// CHECK:STDOUT:   %start.ref: %IntRange.elem.f91 = name_ref start, imports.%Main.import_ref.a9e [concrete = imports.%field_decl.c45]
// CHECK:STDOUT:   %.loc5_3.2: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc5_3.2: <bound method> = bound_method %.loc5_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc5_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc5_3.2(%.loc5_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc5_3.3(%self.param: ref %IntRange.716) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc5_3.3(%self.param: ref %IntRange.716) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.5ac
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.b41
// CHECK:STDOUT:   %Int => constants.%Int.682b94.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.aec7ec.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.850
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.9da
// CHECK:STDOUT:   %complete_type => constants.%complete_type.3bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.682b94.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.941e2b.1
// CHECK:STDOUT:   %start.param_patt => constants.%start.param_patt
// CHECK:STDOUT:   %start.patt => constants.%start.patt
// CHECK:STDOUT:   %end.param_patt => constants.%end.param_patt.406
// CHECK:STDOUT:   %end.patt => constants.%end.patt.ddb
// CHECK:STDOUT:   %IntRange => constants.%IntRange.872
// CHECK:STDOUT:   %.1 => constants.%.45d
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.ef5
// CHECK:STDOUT:   %return.patt => constants.%return.patt.2ac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.65e
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.23f
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.716
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.f91
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.74c
// CHECK:STDOUT:   %complete_type => constants.%complete_type.ec9
// CHECK:STDOUT: }
// CHECK:STDOUT:

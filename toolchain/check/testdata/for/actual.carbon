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

class IntRange(N:! Core.IntLiteral()) {
  fn Make(start: Core.Int(N), end: Core.Int(N)) -> Self {
    return {.start = start, .end = end};
  }

  impl as Core.Iterate where .CursorType = Core.Int(N) and .ElementType = Core.Int(N) {
    fn NewCursor[self: Self]() -> Core.Int(N) { return self.start; }
    fn Next[self: Self](cursor: Core.Int(N)*) -> Core.Optional(Core.Int(N)) {
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

fn Read(y:! Core.IntLiteral()) {
  var unused x: IntRange(32) = Range(y);
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.1b9: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.97e5cd.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.f31898.1: type = pattern_type %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %.57c: Core.Form = init_form %IntRange.1b9 [symbolic]
// CHECK:STDOUT:   %pattern_type.cef: type = pattern_type %IntRange.1b9 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.127: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.89b: %IntRange.Make.type.127 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %facet_type.97e: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.b1d: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.4e3: type = fn_type @Optional.Some, @Optional(%T.b1d) [symbolic]
// CHECK:STDOUT:   %Optional.Some.ed1: %Optional.Some.type.4e3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.None.type.e7a: type = fn_type @Optional.None, @Optional(%T.b1d) [symbolic]
// CHECK:STDOUT:   %Optional.None.3e8: %Optional.None.type.e7a = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.2a0: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.2a0 [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.567: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.a97 [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1a8: <witness> = lookup_impl_witness %.Self.2a0, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.84b: type = impl_witness_access %Iterate.lookup_impl_witness.1a8, element1 [symbolic_self]
// CHECK:STDOUT:   %assoc0.266: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.3b2 [concrete]
// CHECK:STDOUT:   %impl.elem0.867: %facet_type.97e = impl_witness_access %Iterate.lookup_impl_witness.1a8, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.47f: <witness> = lookup_impl_witness %Int.97e5cd.1, @Destroy [symbolic]
// CHECK:STDOUT:   %require_complete.22cb04.1: <witness> = require_complete_type %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6c1f60.1: Core.Form = init_form %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a6b: <witness> = lookup_impl_witness %Int.97e5cd.1, @Copy [symbolic]
// CHECK:STDOUT:   %.b6e: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.97e = facet_value %Int.97e5cd.1, (%Destroy.lookup_impl_witness.47f, %Copy.lookup_impl_witness.a6b) [symbolic]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.84b = %Int.97e5cd.1 and %impl.elem0.867 = %facet_value> [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.as.Iterate.impl.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.767: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.91e: type = ptr_type %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.0b0: type = pattern_type %ptr.91e [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.7d5: <witness> = lookup_impl_witness %Int.97e5cd.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %.6fc: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.214: %OptionalStorage.type = facet_value %Int.97e5cd.1, (%OptionalStorage.lookup_impl_witness.7d5) [symbolic]
// CHECK:STDOUT:   %Optional.8fc: type = class_type @Optional, @Optional(%OptionalStorage.facet.214) [symbolic]
// CHECK:STDOUT:   %.86f: Core.Form = init_form %Optional.8fc [symbolic]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %Optional.8fc [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IntRange.elem.777: type = unbound_element_type %IntRange.1b9, %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.7e8: type = struct_type {.start: %Int.97e5cd.1, .end: %Int.97e5cd.1} [symbolic]
// CHECK:STDOUT:   %complete_type.b28: <witness> = complete_type_witness %struct_type.start.end.7e8 [symbolic]
// CHECK:STDOUT:   %require_complete.20c: <witness> = require_complete_type %IntRange.1b9 [symbolic]
// CHECK:STDOUT:   %Copy.facet.a6d: %Copy.type = facet_value %Int.97e5cd.1, (%Copy.lookup_impl_witness.a6b) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.44f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.a6d) [symbolic]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %Copy.WithSelf.Op.type.44f, %Copy.facet.a6d [symbolic]
// CHECK:STDOUT:   %impl.elem0.553: %.979 = impl_witness_access %Copy.lookup_impl_witness.a6b, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e44: <specific function> = specific_impl_function %impl.elem0.553, @Copy.WithSelf.Op(%Copy.facet.a6d) [symbolic]
// CHECK:STDOUT:   %require_complete.f8a: <witness> = require_complete_type %ptr.91e [symbolic]
// CHECK:STDOUT:   %Optional.None.type.132: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.214) [symbolic]
// CHECK:STDOUT:   %Optional.None.065: %Optional.None.type.132 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.664: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.214) [symbolic]
// CHECK:STDOUT:   %Optional.Some.613: %Optional.Some.type.664 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.809: <witness> = require_complete_type %Optional.8fc [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.a39 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c17: type = facet_type <@OrderedWith, @OrderedWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.33c: %OrderedWith.type.c17 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eb: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.33c) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.cef: %OrderedWith.WithSelf.Less.type.123 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.0aa: type = facet_type <@OrderedWith, @OrderedWith(%Int.97e5cd.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.8ab: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.97e5cd.1) [symbolic]
// CHECK:STDOUT:   %assoc0.269: %OrderedWith.assoc_type.8ab = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic]
// CHECK:STDOUT:   %require_complete.dbf: <witness> = require_complete_type %OrderedWith.type.0aa [symbolic]
// CHECK:STDOUT:   %assoc0.e6a: %OrderedWith.assoc_type.2eb = assoc_entity element0, imports.%Core.import_ref.52f [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.cc0: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.e2c(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.c34: %Int.as.OrderedWith.impl.Less.type.cc0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.e33: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.000, @Int.as.OrderedWith.impl.e2c(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.d1e: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.e2c(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.282: %Int.as.OrderedWith.impl.Less.type.d1e = struct_value () [symbolic]
// CHECK:STDOUT:   %.6ea: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.e2c(%N, %N) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet.5b7: %OrderedWith.type.0aa = facet_value %Int.97e5cd.1, (%OrderedWith.impl_witness.e33) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.8d3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.97e5cd.1, %OrderedWith.facet.5b7) [symbolic]
// CHECK:STDOUT:   %.c57: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.8d3, %OrderedWith.facet.5b7 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.136: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.282, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.97e5cd.1, @Inc [symbolic]
// CHECK:STDOUT:   %.37d: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.97e5cd.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.5bc: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %.c54: type = fn_type_with_self_type %Inc.WithSelf.Op.type.5bc, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.873: %.c54 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e86: <specific function> = specific_impl_function %impl.elem0.873, @Inc.WithSelf.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.613, @Optional.Some(%OptionalStorage.facet.214) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.a10: %Destroy.type = facet_value %Int.97e5cd.1, (%Destroy.lookup_impl_witness.47f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d40: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a10) [symbolic]
// CHECK:STDOUT:   %.bb0: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d40, %Destroy.facet.a10 [symbolic]
// CHECK:STDOUT:   %impl.elem0.7ad: %.bb0 = impl_witness_access %Destroy.lookup_impl_witness.47f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c63: <specific function> = specific_impl_function %impl.elem0.7ad, @Destroy.WithSelf.Op(%Destroy.facet.a10) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.065, @Optional.None(%OptionalStorage.facet.214) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntRange.225: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %.ab6: Core.Form = init_form %IntRange.225 [concrete]
// CHECK:STDOUT:   %pattern_type.7af: type = pattern_type %IntRange.225 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.627: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.20a: %IntRange.Make.type.627 = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.elem.320: type = unbound_element_type %IntRange.225, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.a7c: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.7df: <witness> = complete_type_witness %struct_type.start.end.a7c [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.20a, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3f1: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.3f1) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet.3f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.fed: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f98: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6d3: %Int.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %.89b: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.ddd: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.83a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.83a, %Copy.facet.ddd [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.6d3, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Iterate: type = import_ref Core//prelude/iterate, Iterate, loaded [concrete = constants.%Iterate.type]
// CHECK:STDOUT:   %Core.import_ref.c99: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.266]
// CHECK:STDOUT:   %Core.import_ref.21c: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1.567]
// CHECK:STDOUT:   %Core.import_ref.a60: @Optional.%Optional.None.type (%Optional.None.type.e7a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.3e8)]
// CHECK:STDOUT:   %Core.import_ref.d2c: @Optional.%Optional.Some.type (%Optional.Some.type.4e3) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.ed1)]
// CHECK:STDOUT:   %Core.import_ref.a97: type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.3b2: %facet_type.97e = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %facet_type.97e = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.a39 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.246: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eb) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.e6a)]
// CHECK:STDOUT:   %Core.import_ref.9d2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.123) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.cef)]
// CHECK:STDOUT:   %Core.import_ref.52f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a8b: @Int.as.OrderedWith.impl.e2c.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.cc0) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.e2c.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.c34)]
// CHECK:STDOUT:   %Core.import_ref.2f3d = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c469 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.aebc = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.000 = impl_witness_table (%Core.import_ref.a8b, %Core.import_ref.2f3d, %Core.import_ref.c469, %Core.import_ref.aebc), @Int.as.OrderedWith.impl.e2c [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_36.1: type = splice_block %.loc4_36.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.3: type = converted %IntLiteral.call, %.loc4_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_17.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: %Range.type = fn_decl @Range [concrete = constants.%Range] {
// CHECK:STDOUT:     %end.param_patt: %pattern_type.17a = value_param_pattern [concrete]
// CHECK:STDOUT:     %end.patt: %pattern_type.17a = at_binding_pattern end, %end.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7af = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7af = return_slot_pattern %return.param_patt, %IntRange.loc26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc26: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc26: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.225]
// CHECK:STDOUT:     %.loc26_34.2: Core.Form = init_form %IntRange.loc26 [concrete = constants.%.ab6]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %end: %i32 = value_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.225 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.225 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %.loc9_85.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc9_85.1 (constants.%.b6e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.47f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.a6b)]
// CHECK:STDOUT:   %facet_value.loc9_85.1: %facet_type.97e = facet_value %Int.loc9_54.1, (%Destroy.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.84b = %Int.loc9_54.1 and constants.%impl.elem0.867 = %facet_value.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.767)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type (constants.%IntRange.as.Iterate.impl.NewCursor.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type (constants.%IntRange.as.Iterate.impl.Next.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %.loc9_24 {
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = fn_decl @IntRange.as.Iterate.impl.NewCursor [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)] {
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_22 (%pattern_type.cef) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_22 (%pattern_type.cef) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_45 (%pattern_type.f31898.1) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_45 (%pattern_type.f31898.1) = return_slot_pattern %return.param_patt, %Int.loc10_45.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_45.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_45.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:       %.loc10_45.2: Core.Form = init_form %Int.loc10_45.2 [symbolic = %.loc10_45.1 (constants.%.6c1f60.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.1b9) = value_param call_param0
// CHECK:STDOUT:       %.loc10_24.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.1b9)] {
// CHECK:STDOUT:         %.loc10_24.2: type = specific_constant constants.%IntRange.1b9, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_24.2 [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.1b9) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.97e5cd.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.97e5cd.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_17 (%pattern_type.cef) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_17 (%pattern_type.cef) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_31 (%pattern_type.0b0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %cursor.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_31 (%pattern_type.0b0) = at_binding_pattern cursor, %cursor.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_75 (%pattern_type.76e) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_75 (%pattern_type.76e) = return_slot_pattern %return.param_patt, %Optional.loc11_75.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_68: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_73: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_74: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_75.2: %OptionalStorage.type = facet_value %Int.loc11_74, (constants.%OptionalStorage.lookup_impl_witness.7d5) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:       %.loc11_75.5: %OptionalStorage.type = converted %Int.loc11_74, %OptionalStorage.facet.loc11_75.2 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:       %Optional.loc11_75.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.214) [symbolic = %Optional.loc11_75.1 (constants.%Optional.8fc)]
// CHECK:STDOUT:       %.loc11_75.6: Core.Form = init_form %Optional.loc11_75.2 [symbolic = %.loc11_75.4 (constants.%.86f)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.1b9) = value_param call_param0
// CHECK:STDOUT:       %.loc11_19.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.1b9)] {
// CHECK:STDOUT:         %.loc11_19.2: type = specific_constant constants.%IntRange.1b9, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_19.2 [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.1b9) = value_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.91e) = value_param call_param1
// CHECK:STDOUT:       %.loc11_44: type = splice_block %ptr.loc11_44.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.91e)] {
// CHECK:STDOUT:         %Core.ref.loc11_33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_37: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_42: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_43.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:         %ptr.loc11_44.2: type = ptr_type %Int.loc11_43.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.91e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.91e) = value_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, %IntRange.as.Iterate.impl.NewCursor.decl, %IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness.loc9_87.1: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: type = impl_witness_assoc_constant constants.%Int.97e5cd.1 [symbolic = %Int.loc9_54.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %facet_type.97e = impl_witness_assoc_constant constants.%facet_value [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
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
// CHECK:STDOUT:   %N.loc4_17.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_17.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_17.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.127)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.127) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.89b)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N.loc4_17.1) [symbolic = %Int (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.22cb04.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_17.1) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.777)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.97e5cd.1), .end: @IntRange.%Int (%Int.97e5cd.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.7e8)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.b28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.127) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.89b)] {
// CHECK:STDOUT:       %start.param_patt: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.f31898.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %start.patt: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.f31898.1) = at_binding_pattern start, %start.param_patt [concrete]
// CHECK:STDOUT:       %end.param_patt: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.f31898.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %end.patt: @IntRange.Make.%pattern_type.loc5_16 (%pattern_type.f31898.1) = at_binding_pattern end, %end.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.cef) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.Make.%pattern_type.loc5_52 (%pattern_type.cef) = return_slot_pattern %return.param_patt, %Self.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52.2: type = specific_constant constants.%IntRange.1b9, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52.2 [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:       %.loc5_52.3: Core.Form = init_form %Self.ref [symbolic = %.loc5_52.1 (constants.%.57c)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.97e5cd.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = value_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.97e5cd.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = value_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.1b9) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.1b9) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.1b9 [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2a0]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2a0]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.21c [concrete = constants.%assoc1.567]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%Iterate.lookup_impl_witness.1a8, element1 [symbolic_self = constants.%impl.elem1.84b]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2a0]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.c99 [concrete = constants.%assoc0.266]
// CHECK:STDOUT:       %impl.elem0: %facet_type.97e = impl_witness_access constants.%Iterate.lookup_impl_witness.1a8, element0 [symbolic_self = constants.%impl.elem0.867]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:       %facet_value.loc9_85.2: %facet_type.97e = facet_value %Int.loc9_85, (constants.%Destroy.lookup_impl_witness.47f, constants.%Copy.lookup_impl_witness.a6b) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_85.2: %facet_type.97e = converted %Int.loc9_85, %facet_value.loc9_85.2 [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type %Iterate.ref
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %Int.loc9_54.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_85.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc22: @IntRange.%IntRange.elem (%IntRange.elem.777) = field_decl start, element0 [concrete]
// CHECK:STDOUT:     %.loc23: @IntRange.%IntRange.elem (%IntRange.elem.777) = field_decl end, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.7e8 [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.b28)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.1b9
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %.loc22
// CHECK:STDOUT:     .end [private] = %.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.f31898.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %.loc5_52.1: Core.Form = init_form %IntRange [symbolic = %.loc5_52.1 (constants.%.57c)]
// CHECK:STDOUT:   %pattern_type.loc5_52: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_52 (constants.%pattern_type.cef)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.22cb04.1)]
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.20c)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.7e8)]
// CHECK:STDOUT:   %.loc6_22.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc6_22.1 (constants.%.b6e)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.a6b)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.a6d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.44f)]
// CHECK:STDOUT:   %.loc6_22.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc6_22.2 (constants.%.979)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.2 (%.979) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1)) -> out %return.param: @IntRange.Make.%IntRange (%IntRange.1b9) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.7e8) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.2 (%.979) = impl_witness_access constants.%Copy.lookup_impl_witness.a6b, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.WithSelf.Op(constants.%Copy.facet.a6d) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_22: init @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = class_element_access %return.param, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) to %.loc6_39.2 = in_place_init %Copy.WithSelf.Op.call.loc6_22
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.2 (%.979) = impl_witness_access constants.%Copy.lookup_impl_witness.a6b, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.WithSelf.Op(constants.%Copy.facet.a6d) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) = class_element_access %return.param, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.97e5cd.1) to %.loc6_39.4 = in_place_init %Copy.WithSelf.Op.call.loc6_36
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.1b9) to %return.param = class_init (%.loc6_39.3, %.loc6_39.5)
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.1b9) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %pattern_type.loc10_22: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_22 (constants.%pattern_type.cef)]
// CHECK:STDOUT:   %Int.loc10_45.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_45.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %.loc10_45.1: Core.Form = init_form %Int.loc10_45.1 [symbolic = %.loc10_45.1 (constants.%.6c1f60.1)]
// CHECK:STDOUT:   %pattern_type.loc10_45: type = pattern_type %Int.loc10_45.1 [symbolic = %pattern_type.loc10_45 (constants.%pattern_type.f31898.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_22: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_22 (constants.%require_complete.20c)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_45.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.777)]
// CHECK:STDOUT:   %require_complete.loc10_60: <witness> = require_complete_type %Int.loc10_45.1 [symbolic = %require_complete.loc10_60 (constants.%require_complete.22cb04.1)]
// CHECK:STDOUT:   %.loc10_60.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc10_60.3 (constants.%.b6e)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc10_45.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.a6b)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc10_45.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.a6d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.44f)]
// CHECK:STDOUT:   %.loc10_60.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc10_60.4 (constants.%.979)]
// CHECK:STDOUT:   %impl.elem0.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%.loc10_60.4 (%.979) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_60.2: <specific function> = specific_impl_function %impl.elem0.loc10_60.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.1b9)) -> out %return.param: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.97e5cd.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.1b9) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.777) = name_ref start, @IntRange.%.loc22 [concrete = @IntRange.%.loc22]
// CHECK:STDOUT:     %.loc10_60.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.97e5cd.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.97e5cd.1) = acquire_value %.loc10_60.1
// CHECK:STDOUT:     %impl.elem0.loc10_60.1: @IntRange.as.Iterate.impl.NewCursor.%.loc10_60.4 (%.979) = impl_witness_access constants.%Copy.lookup_impl_witness.a6b, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:     %bound_method.loc10_60.1: <bound method> = bound_method %.loc10_60.2, %impl.elem0.loc10_60.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_60.1: <specific function> = specific_impl_function %impl.elem0.loc10_60.1, @Copy.WithSelf.Op(constants.%Copy.facet.a6d) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:     %bound_method.loc10_60.2: <bound method> = bound_method %.loc10_60.2, %specific_impl_fn.loc10_60.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.97e5cd.1) = call %bound_method.loc10_60.2(%.loc10_60.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %pattern_type.loc11_17: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_17 (constants.%pattern_type.cef)]
// CHECK:STDOUT:   %Int.loc11_43.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %ptr.loc11_44.1: type = ptr_type %Int.loc11_43.1 [symbolic = %ptr.loc11_44.1 (constants.%ptr.91e)]
// CHECK:STDOUT:   %pattern_type.loc11_31: type = pattern_type %ptr.loc11_44.1 [symbolic = %pattern_type.loc11_31 (constants.%pattern_type.0b0)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.47f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.a6b)]
// CHECK:STDOUT:   %facet_value: %facet_type.97e = facet_value %Int.loc11_43.1, (%Destroy.lookup_impl_witness, %Copy.lookup_impl_witness) [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %.loc11_75.3: require_specific_def_type = require_specific_def @T.as_type.as.OptionalStorage.impl(%facet_value) [symbolic = %.loc11_75.3 (constants.%.6fc)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.7d5)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_75.1: %OptionalStorage.type = facet_value %Int.loc11_43.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:   %Optional.loc11_75.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.8fc)]
// CHECK:STDOUT:   %.loc11_75.4: Core.Form = init_form %Optional.loc11_75.1 [symbolic = %.loc11_75.4 (constants.%.86f)]
// CHECK:STDOUT:   %pattern_type.loc11_75: type = pattern_type %Optional.loc11_75.1 [symbolic = %pattern_type.loc11_75 (constants.%pattern_type.76e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_17: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_17 (constants.%require_complete.20c)]
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %ptr.loc11_44.1 [symbolic = %require_complete.loc11_31 (constants.%require_complete.f8a)]
// CHECK:STDOUT:   %require_complete.loc11_47: <witness> = require_complete_type %Optional.loc11_75.1 [symbolic = %require_complete.loc11_47 (constants.%require_complete.809)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %Int.loc11_43.1 [symbolic = %require_complete.loc12 (constants.%require_complete.22cb04.1)]
// CHECK:STDOUT:   %pattern_type.loc12: type = pattern_type %Int.loc11_43.1 [symbolic = %pattern_type.loc12 (constants.%pattern_type.f31898.1)]
// CHECK:STDOUT:   %.loc12_32.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc12_32.3 (constants.%.b6e)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc11_43.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.a6d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.44f)]
// CHECK:STDOUT:   %.loc12_32.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc12_32.4 (constants.%.979)]
// CHECK:STDOUT:   %impl.elem0.loc12_32.2: @IntRange.as.Iterate.impl.Next.%.loc12_32.4 (%.979) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_32.2: <specific function> = specific_impl_function %impl.elem0.loc12_32.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_43.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.777)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_43.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.0aa)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.dbf)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.8ab)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.8ab) = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic = %assoc0 (constants.%assoc0.269)]
// CHECK:STDOUT:   %.loc13_17.2: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.e2c(%N, %N) [symbolic = %.loc13_17.2 (constants.%.6ea)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.000, @Int.as.OrderedWith.impl.e2c(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.e33)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.0aa) = facet_value %Int.loc11_43.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet.5b7)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Int.loc11_43.1, %OrderedWith.facet) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.8d3)]
// CHECK:STDOUT:   %.loc13_17.3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.3 (constants.%.c57)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.e2c(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.d1e)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.d1e) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.282)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.136)]
// CHECK:STDOUT:   %.loc14_9.1: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic = %.loc14_9.1 (constants.%.37d)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.loc11_43.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.5bc)]
// CHECK:STDOUT:   %.loc14_9.2: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %Inc.facet [symbolic = %.loc14_9.2 (constants.%.c54)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.c54) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.873)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.WithSelf.Op(%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.e86)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.664)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.664) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.613)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc11_43.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.a10)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d40)]
// CHECK:STDOUT:   %.loc12_7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc12_7 (constants.%.bb0)]
// CHECK:STDOUT:   %impl.elem0.loc12_7.3: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.bb0) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.7ad)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_7.3: <specific function> = specific_impl_function %impl.elem0.loc12_7.3, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.c63)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.132)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.132) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.065)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.1b9), %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.91e)) -> out %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.f31898.1) = ref_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.var_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.f31898.1) = var_pattern %value.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = var %value.var_patt
// CHECK:STDOUT:     %cursor.ref.loc12: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.91e) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = deref %cursor.ref.loc12
// CHECK:STDOUT:     %.loc12_32.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = acquire_value %.loc12_32.1
// CHECK:STDOUT:     %impl.elem0.loc12_32.1: @IntRange.as.Iterate.impl.Next.%.loc12_32.4 (%.979) = impl_witness_access constants.%Copy.lookup_impl_witness.a6b, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.553)]
// CHECK:STDOUT:     %bound_method.loc12_32.1: <bound method> = bound_method %.loc12_32.2, %impl.elem0.loc12_32.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_32.1: <specific function> = specific_impl_function %impl.elem0.loc12_32.1, @Copy.WithSelf.Op(constants.%Copy.facet.a6d) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.e44)]
// CHECK:STDOUT:     %bound_method.loc12_32.2: <bound method> = bound_method %.loc12_32.2, %specific_impl_fn.loc12_32.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = call %bound_method.loc12_32.2(%.loc12_32.2)
// CHECK:STDOUT:     assign %value.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = ref_binding value, %value.var
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.1b9) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.777) = name_ref end, @IntRange.%.loc23 [concrete = @IntRange.%.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = acquire_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.97e5cd.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.0aa)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.8ab) = specific_constant imports.%Core.import_ref.246, @OrderedWith.WithSelf(constants.%Int.97e5cd.1, constants.%Self.33c) [symbolic = %assoc0 (constants.%assoc0.269)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.8ab) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.269)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.3 (%.c57) = impl_witness_access constants.%OrderedWith.impl_witness.e33, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.282)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.136)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = acquire_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.91e) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc14_11: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = deref %cursor.ref.loc14
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.c54) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.873)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %.loc14_11, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.WithSelf.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.e86)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %.loc14_11, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %Inc.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_9.2(%.loc14_11)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_17.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_41: %OptionalStorage.type = facet_value %Int.loc15, (constants.%OptionalStorage.lookup_impl_witness.7d5) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:     %.loc15_41: %OptionalStorage.type = converted %Int.loc15, %OptionalStorage.facet.loc15_41 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.214) [symbolic = %Optional.loc11_75.1 (constants.%Optional.8fc)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.664) = specific_constant imports.%Core.import_ref.d2c, @Optional(constants.%OptionalStorage.facet.214) [symbolic = %Optional.Some (constants.%Optional.Some.613)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.664) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.613)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53: %OptionalStorage.type = facet_value constants.%Int.97e5cd.1, (constants.%OptionalStorage.lookup_impl_witness.7d5) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:     %.loc15_53: %OptionalStorage.type = converted constants.%Int.97e5cd.1, %OptionalStorage.facet.loc15_53 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.214) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_75.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.97e5cd.1) = acquire_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) to %.loc11_75.1 = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48)
// CHECK:STDOUT:     %impl.elem0.loc12_7.1: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.bb0) = impl_witness_access constants.%Destroy.lookup_impl_witness.47f, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.7ad)]
// CHECK:STDOUT:     %bound_method.loc12_7.1: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.1: <specific function> = specific_impl_function %impl.elem0.loc12_7.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.a10) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.c63)]
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
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc17: %OptionalStorage.type = facet_value %Int.loc17, (constants.%OptionalStorage.lookup_impl_witness.7d5) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:     %.loc17_41: %OptionalStorage.type = converted %Int.loc17, %OptionalStorage.facet.loc17 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.214)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.214) [symbolic = %Optional.loc11_75.1 (constants.%Optional.8fc)]
// CHECK:STDOUT:     %.loc17_42: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.132) = specific_constant imports.%Core.import_ref.a60, @Optional(constants.%OptionalStorage.facet.214) [symbolic = %Optional.None (constants.%Optional.None.065)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.132) = name_ref None, %.loc17_42 [symbolic = %Optional.None (constants.%Optional.None.065)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.214) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_75.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) = splice_block %return.param {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.8fc) to %.loc11_75.2 = call %Optional.None.specific_fn.loc17_42.1()
// CHECK:STDOUT:     %impl.elem0.loc12_7.2: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.bb0) = impl_witness_access constants.%Destroy.lookup_impl_witness.47f, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.7ad)]
// CHECK:STDOUT:     %bound_method.loc12_7.3: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.2
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.2: <specific function> = specific_impl_function %impl.elem0.loc12_7.2, @Destroy.WithSelf.Op(constants.%Destroy.facet.a10) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.c63)]
// CHECK:STDOUT:     %bound_method.loc12_7.4: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc12_7.2: init %empty_tuple.type = call %bound_method.loc12_7.4(%value.var)
// CHECK:STDOUT:     return %Optional.None.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> out %return.param: %IntRange.225 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.225]
// CHECK:STDOUT:   %.loc27_22: %IntRange.Make.type.627 = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.20a]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.627 = name_ref Make, %.loc27_22 [concrete = constants.%IntRange.Make.20a]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc26_34.1: ref %IntRange.225 = splice_block %return.param {}
// CHECK:STDOUT:   %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.225 to %.loc26_34.1 = call %IntRange.Make.specific_fn(%.loc27_28.2, %end.ref)
// CHECK:STDOUT:   return %IntRange.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.127
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.89b
// CHECK:STDOUT:   %Int => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.22cb04.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.777
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.7e8
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.b28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.f31898.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.57c
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.cef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %.loc9_85.1 => constants.%.b6e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.47f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.a6b
// CHECK:STDOUT:   %facet_value.loc9_85.1 => constants.%facet_value
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2 => constants.%Iterate.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.767
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type => constants.%IntRange.as.Iterate.impl.NewCursor.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor => constants.%IntRange.as.Iterate.impl.NewCursor
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type => constants.%IntRange.as.Iterate.impl.Next.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next => constants.%IntRange.as.Iterate.impl.Next
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.NewCursor(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %pattern_type.loc10_22 => constants.%pattern_type.cef
// CHECK:STDOUT:   %Int.loc10_45.1 => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %.loc10_45.1 => constants.%.6c1f60.1
// CHECK:STDOUT:   %pattern_type.loc10_45 => constants.%pattern_type.f31898.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %pattern_type.loc11_17 => constants.%pattern_type.cef
// CHECK:STDOUT:   %Int.loc11_43.1 => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %ptr.loc11_44.1 => constants.%ptr.91e
// CHECK:STDOUT:   %pattern_type.loc11_31 => constants.%pattern_type.0b0
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness.47f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.a6b
// CHECK:STDOUT:   %facet_value => constants.%facet_value
// CHECK:STDOUT:   %.loc11_75.3 => constants.%.6fc
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.7d5
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_75.1 => constants.%OptionalStorage.facet.214
// CHECK:STDOUT:   %Optional.loc11_75.1 => constants.%Optional.8fc
// CHECK:STDOUT:   %.loc11_75.4 => constants.%.86f
// CHECK:STDOUT:   %pattern_type.loc11_75 => constants.%pattern_type.76e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.loc4_17.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.627
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.20a
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.225
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.320
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.a7c
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.7df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.17a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.225
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.ab6
// CHECK:STDOUT:   %pattern_type.loc5_52 => constants.%pattern_type.7af
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc5_49 => constants.%complete_type.7df
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.a7c
// CHECK:STDOUT:   %.loc6_22.1 => constants.%.89b
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f1f
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.ddd
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.83a
// CHECK:STDOUT:   %.loc6_22.2 => constants.%.2ec
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.6d3
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0 [symbolic]
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.97e5cd.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.7e8: type = struct_type {.start: %Int.97e5cd.1, .end: %Int.97e5cd.1} [symbolic]
// CHECK:STDOUT:   %complete_type.b28: <witness> = complete_type_witness %struct_type.start.end.7e8 [symbolic]
// CHECK:STDOUT:   %IntRange.1b9: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.elem.777: type = unbound_element_type %IntRange.1b9, %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %require_complete.22cb04.1: <witness> = require_complete_type %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.127: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.89b: %IntRange.Make.type.127 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.cef: type = pattern_type %IntRange.1b9 [symbolic]
// CHECK:STDOUT:   %.57c: Core.Form = init_form %IntRange.1b9 [symbolic]
// CHECK:STDOUT:   %pattern_type.f31898.1: type = pattern_type %Int.97e5cd.1 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a6b: <witness> = lookup_impl_witness %Int.97e5cd.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.97e5cd.1, (%Copy.lookup_impl_witness.a6b) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.44f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %Copy.WithSelf.Op.type.44f, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.553: %.979 = impl_witness_access %Copy.lookup_impl_witness.a6b, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e44: <specific function> = specific_impl_function %impl.elem0.553, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.b6e: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.20c: <witness> = require_complete_type %IntRange.1b9 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.225: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.627: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.20a: %IntRange.Make.type.627 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.elem.320: type = unbound_element_type %IntRange.225, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.a7c: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.7df: <witness> = complete_type_witness %struct_type.start.end.a7c [concrete]
// CHECK:STDOUT:   %pattern_type.7af: type = pattern_type %IntRange.225 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.4: type = fn_type @Destroy.Op.loc5_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.4: %Destroy.Op.type.3f5f78.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IntRange: %IntRange.type = import_ref Main//lib, IntRange, loaded [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %Main.Range: %Range.type = import_ref Main//lib, Range, loaded [concrete = constants.%Range]
// CHECK:STDOUT:   %Core.a3bffd.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Main.import_ref.6b552a.1: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.7d4: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.b28)]
// CHECK:STDOUT:   %Main.import_ref.ddc = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b52 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.603 = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.dee = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.6b552a.2: Core.IntLiteral = import_ref Main//lib, loc4_17, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntRange = imports.%Main.IntRange
// CHECK:STDOUT:     .Range = imports.%Main.Range
// CHECK:STDOUT:     .Core = imports.%Core.a3bffd.1
// CHECK:STDOUT:     .Read = %Read.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Read.decl: %Read.type = fn_decl @Read [concrete = constants.%Read] {
// CHECK:STDOUT:     %y.patt: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_29.1: type = splice_block %.loc4_29.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core.a3bffd.1 [concrete = imports.%Core.a3bffd.1]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_29.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_29.3: type = converted %IntLiteral.call, %.loc4_29.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y.loc4_10.2: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_10.1 (constants.%y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.6b552a.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.127)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.127) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.89b)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.22cb04.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.777)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.97e5cd.1), .end: @IntRange.%Int (%Int.97e5cd.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.7e8)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.b28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.7d4
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ddc
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.b52
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.603
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.dee
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Read(%y.loc4_10.2: Core.IntLiteral) {
// CHECK:STDOUT:   %y.loc4_10.1: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_10.1 (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y.loc4_10.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_38.3: <bound method> = bound_method %y.loc4_10.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2: init %i32 = call %bound_method.loc5_38.3(%y.loc4_10.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.7af = ref_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.7af = var_pattern %x.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref %IntRange.225 = var %x.var_patt
// CHECK:STDOUT:     %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:     %y.ref: Core.IntLiteral = name_ref y, %y.loc4_10.2 [symbolic = %y.loc4_10.1 (constants.%y)]
// CHECK:STDOUT:     %.loc5_3: ref %IntRange.225 = splice_block %x.var {}
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc5_38.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_38.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = %bound_method.loc5_38.3 (constants.%bound_method)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1: init %i32 = call %bound_method.loc5_38.2(%y.ref) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_38.2: %i32 = converted %y.ref, %.loc5_38.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_38.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Range.call: init %IntRange.225 to %.loc5_3 = call %Range.ref(%.loc5_38.2)
// CHECK:STDOUT:     assign %x.var, %Range.call
// CHECK:STDOUT:     %.loc5_28: type = splice_block %IntRange [concrete = constants.%IntRange.225] {
// CHECK:STDOUT:       %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.225]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %IntRange.225 = ref_binding x, %x.var
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op.f0ba5f.4
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.6b552a.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.97e5cd.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.f31898.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.1b9)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %IntRange [symbolic = %.1 (constants.%.57c)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.cef)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %Int [symbolic = %require_complete.1 (constants.%require_complete.22cb04.1)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %IntRange [symbolic = %require_complete.2 (constants.%require_complete.20c)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.97e5cd.1), .end: @IntRange.Make.%Int (%Int.97e5cd.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.7e8)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.2 (constants.%.b6e)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.a6b)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.44f)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.3 (constants.%.979)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.3 (%.979) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.553)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.e44)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.3(%self.param: ref %struct_type.start.end.a7c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.4(%self.param: ref %IntRange.225) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Read(constants.%y) {
// CHECK:STDOUT:   %y.loc4_10.1 => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.127
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.89b
// CHECK:STDOUT:   %Int => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.22cb04.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.777
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.7e8
// CHECK:STDOUT:   %complete_type => constants.%complete_type.b28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.97e5cd.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f31898.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.1b9
// CHECK:STDOUT:   %.1 => constants.%.57c
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.cef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.627
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.20a
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.225
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.320
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.a7c
// CHECK:STDOUT:   %complete_type => constants.%complete_type.7df
// CHECK:STDOUT: }
// CHECK:STDOUT:

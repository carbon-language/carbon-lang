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

fn Read() {
  let y:! Core.IntLiteral() = 43;
  var x: IntRange(32) = Range(y);
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.48b: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.b50e37.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.60f8e7.1: type = pattern_type %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.121: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.14e: %IntRange.Make.type.121 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %.Self.b7a: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.02e: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.9e6 [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.b7a [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.0cc: <witness> = lookup_impl_witness %.Self.b7a, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.aba: type = impl_witness_access %Iterate.lookup_impl_witness.0cc, element1 [symbolic_self]
// CHECK:STDOUT:   %assoc0.0f6: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.61e [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %impl.elem0.0b5: %Copy.type = impl_witness_access %Iterate.lookup_impl_witness.0cc, element0 [symbolic_self]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.e66303.1: <witness> = require_complete_type %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.035: <witness> = lookup_impl_witness %Int.b50e37.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.d3d: %Copy.type = facet_value %Int.b50e37.1, (%Copy.lookup_impl_witness.035) [symbolic]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.aba = %Int.b50e37.1 and %impl.elem0.0b5 = %Copy.facet.d3d> [symbolic]
// CHECK:STDOUT:   %require_complete.e76: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.3fe: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.193: type = fn_type @Optional.None, @Optional(%T.3fe) [symbolic]
// CHECK:STDOUT:   %Optional.None.dc7: %Optional.None.type.193 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.32b: type = fn_type @Optional.Some, @Optional(%T.3fe) [symbolic]
// CHECK:STDOUT:   %Optional.Some.249: %Optional.Some.type.32b = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.1cf: type = ptr_type %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.054: type = pattern_type %ptr.1cf [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.8c6: <witness> = lookup_impl_witness %Int.b50e37.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.ea4: %OptionalStorage.type = facet_value %Int.b50e37.1, (%OptionalStorage.lookup_impl_witness.8c6) [symbolic]
// CHECK:STDOUT:   %Optional.23f: type = class_type @Optional, @Optional(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %Optional.23f [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IntRange.elem.baf: type = unbound_element_type %IntRange.48b, %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.cb5: type = struct_type {.start: %Int.b50e37.1, .end: %Int.b50e37.1} [symbolic]
// CHECK:STDOUT:   %complete_type.4f0: <witness> = complete_type_witness %struct_type.start.end.cb5 [symbolic]
// CHECK:STDOUT:   %require_complete.b4c: <witness> = require_complete_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %.20c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.d3d [symbolic]
// CHECK:STDOUT:   %impl.elem0.625: %.20c = impl_witness_access %Copy.lookup_impl_witness.035, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c88: <specific function> = specific_impl_function %impl.elem0.625, @Copy.Op(%Copy.facet.d3d) [symbolic]
// CHECK:STDOUT:   %Optional.None.type.a39: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %Optional.None.b0b: %Optional.None.type.a39 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.74c: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %Optional.Some.37c: %Optional.Some.type.74c = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.b49: <witness> = require_complete_type %Optional.23f [symbolic]
// CHECK:STDOUT:   %require_complete.6ca: <witness> = require_complete_type %ptr.1cf [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.fc4: type = fn_type @OrderedWith.Less, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.bb3: %OrderedWith.Less.type.fc4 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72b: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.393: type = facet_type <@OrderedWith, @OrderedWith(%Int.b50e37.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.c6f: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.b50e37.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72a: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.b50e37.1) [symbolic]
// CHECK:STDOUT:   %assoc0.3f6: %OrderedWith.assoc_type.72a = assoc_entity element0, imports.%Core.import_ref.02d [symbolic]
// CHECK:STDOUT:   %require_complete.4f7: <witness> = require_complete_type %OrderedWith.type.393 [symbolic]
// CHECK:STDOUT:   %assoc0.7c2: %OrderedWith.assoc_type.72b = assoc_entity element0, imports.%Core.import_ref.e123aa.1 [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.e01: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.849(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.7c7: %Int.as.OrderedWith.impl.Less.type.e01 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.586: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.401, @Int.as.OrderedWith.impl.849(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.f52: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.849(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.af3: %Int.as.OrderedWith.impl.Less.type.f52 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.393 = facet_value %Int.b50e37.1, (%OrderedWith.impl_witness.586) [symbolic]
// CHECK:STDOUT:   %.db5: type = fn_type_with_self_type %OrderedWith.Less.type.c6f, %OrderedWith.facet [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.0ff: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.af3, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.Op.type: type = fn_type @Inc.Op [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.b50e37.1, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.b50e37.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.080: type = fn_type_with_self_type %Inc.Op.type, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.923: %.080 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.173: <specific function> = specific_impl_function %impl.elem0.923, @Inc.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.37c, @Optional.Some(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Int.b50e37.1, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.437: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.353: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.3be: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.353 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.b50e37.1, (%Destroy.impl_witness.437) [symbolic]
// CHECK:STDOUT:   %.200: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.3be, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.b0b, @Optional.None(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntRange.365: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d16: type = pattern_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.cef: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.0dc: %IntRange.Make.type.cef = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.elem.e33: type = unbound_element_type %IntRange.365, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.d0a: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c45: <witness> = complete_type_witness %struct_type.start.end.d0a [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.0dc, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c50: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c50 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.119: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.0f6]
// CHECK:STDOUT:   %Core.import_ref.ed6: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:   %Core.import_ref.9e6: type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.61e: %Copy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %Copy.type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f2f: @Optional.%Optional.None.type (%Optional.None.type.193) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.dc7)]
// CHECK:STDOUT:   %Core.import_ref.961: @Optional.%Optional.Some.type (%Optional.Some.type.32b) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.249)]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.128: @OrderedWith.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72b) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.%assoc0 (constants.%assoc0.7c2)]
// CHECK:STDOUT:   %Core.import_ref.02d: @OrderedWith.%OrderedWith.Less.type (%OrderedWith.Less.type.fc4) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.%OrderedWith.Less (constants.%OrderedWith.Less.bb3)]
// CHECK:STDOUT:   %Core.import_ref.e123aa.1 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b35: @Int.as.OrderedWith.impl.849.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.e01) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.849.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.7c7)]
// CHECK:STDOUT:   %Core.import_ref.2a1 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3bc = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8fd = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.401 = impl_witness_table (%Core.import_ref.b35, %Core.import_ref.2a1, %Core.import_ref.3bc, %Core.import_ref.8fd), @Int.as.OrderedWith.impl.849 [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.3: type = converted %IntLiteral.call, %.loc4_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_16.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: %Range.type = fn_decl @Range [concrete = constants.%Range] {
// CHECK:STDOUT:     %end.patt: %pattern_type.7ce = value_binding_pattern end [concrete]
// CHECK:STDOUT:     %end.param_patt: %pattern_type.7ce = value_param_pattern %end.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d16 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d16 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc26: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc26_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc26: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc26_15: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc26_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %end: %i32 = value_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.365 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.365 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet.loc9_85.1: %Copy.type = facet_value %Int.loc9_54.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.aba = %Int.loc9_54.1 and constants.%impl.elem0.0b5 = %Copy.facet.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.e76)]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic = %Iterate.impl_witness (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type (constants.%IntRange.as.Iterate.impl.NewCursor.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type (constants.%IntRange.as.Iterate.impl.Next.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %.loc9_24 {
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = fn_decl @IntRange.as.Iterate.impl.NewCursor [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.6de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_32 (%pattern_type.60f8e7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_32 (%pattern_type.60f8e7.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_45.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_45.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.48b) = value_param call_param0
// CHECK:STDOUT:       %.loc10_24.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.48b)] {
// CHECK:STDOUT:         %.loc10_24.2: type = specific_constant constants.%IntRange.48b, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_24.2 [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.48b) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.b50e37.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.b50e37.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.6de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %cursor.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.054) = value_binding_pattern cursor [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.054) = value_param_pattern %cursor.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_47 (%pattern_type.9f4) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_47 (%pattern_type.9f4) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_68: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_73: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_74: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_75.2: %OptionalStorage.type = facet_value %Int.loc11_74, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:       %.loc11_75: %OptionalStorage.type = converted %Int.loc11_74, %OptionalStorage.facet.loc11_75.2 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:       %Optional.loc11_75.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.loc11_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.48b) = value_param call_param0
// CHECK:STDOUT:       %.loc11_19.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.48b)] {
// CHECK:STDOUT:         %.loc11_19.2: type = specific_constant constants.%IntRange.48b, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_19.2 [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.48b) = value_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = value_param call_param1
// CHECK:STDOUT:       %.loc11_44: type = splice_block %ptr.loc11_44.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.1cf)] {
// CHECK:STDOUT:         %Core.ref.loc11_33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_37: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_42: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_43.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:         %ptr.loc11_44.2: type = ptr_type %Int.loc11_43.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.1cf)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = value_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .NewCursor = %IntRange.as.Iterate.impl.NewCursor.decl
// CHECK:STDOUT:     .Next = %IntRange.as.Iterate.impl.Next.decl
// CHECK:STDOUT:     witness = @IntRange.%Iterate.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_16.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_16.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.121)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.121) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.14e)]
// CHECK:STDOUT:   %Int.loc22_32.2: type = class_type @Int, @Int(%N.loc4_16.1) [symbolic = %Int.loc22_32.2 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc22_32.2 [symbolic = %require_complete (constants.%require_complete.e66303.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_16.1) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc22_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.baf)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int.loc22_32.2 (%Int.b50e37.1), .end: @IntRange.%Int.loc22_32.2 (%Int.b50e37.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.cb5)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.4f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.121) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.14e)] {
// CHECK:STDOUT:       %start.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_binding_pattern start [concrete]
// CHECK:STDOUT:       %start.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_param_pattern %start.patt, call_param0 [concrete]
// CHECK:STDOUT:       %end.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_binding_pattern end [concrete]
// CHECK:STDOUT:       %end.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_param_pattern %end.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.6de) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.6de) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52: type = specific_constant constants.%IntRange.48b, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52 [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.48b) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.48b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.48b [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.b7a]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.b7a]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.ed6 [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%Iterate.lookup_impl_witness.0cc, element1 [symbolic_self = constants.%impl.elem1.aba]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.b7a]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.119 [concrete = constants.%assoc0.0f6]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0: %Copy.type = impl_witness_access constants.%Iterate.lookup_impl_witness.0cc, element0 [symbolic_self = constants.%impl.elem0.0b5]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %Copy.facet.loc9_85.2: %Copy.type = facet_value %Int.loc9_85, (constants.%Copy.lookup_impl_witness.035) [symbolic = %Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:       %.loc9_85: %Copy.type = converted %Int.loc9_85, %Copy.facet.loc9_85.2 [symbolic = %Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr %.Self [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%Iterate.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %Int.loc9_54.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_85
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.decl, @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = @IntRange.as.Iterate.impl.%Iterate.impl_witness (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: type = impl_witness_assoc_constant constants.%Int.b50e37.1 [symbolic = @IntRange.as.Iterate.impl.%Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %Copy.type = impl_witness_assoc_constant constants.%Copy.facet.d3d [symbolic = @IntRange.as.Iterate.impl.%Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc22: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:     %Int.loc22_32.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %.loc22: @IntRange.%IntRange.elem (%IntRange.elem.baf) = field_decl start, element0 [concrete]
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc23: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc23: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:     %Int.loc23: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %.loc23: @IntRange.%IntRange.elem (%IntRange.elem.baf) = field_decl end, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.cb5 [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.4f0)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.48b
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %.loc22
// CHECK:STDOUT:     .end [private] = %.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %pattern_type.loc5_11: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_11 (constants.%pattern_type.60f8e7.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.loc5_49: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_49 (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.e66303.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.cb5)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %.loc6_22.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc6_22.2 (constants.%.20c)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.2 (%.20c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1)) -> %return.param: @IntRange.Make.%IntRange (%IntRange.48b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.cb5) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.2 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %.loc6_22.1: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = initialize_from %.loc6_22.1 to %.loc6_39.2
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.2 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = class_element_access %return, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = initialize_from %.loc6_36 to %.loc6_39.4
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.48b) = class_init (%.loc6_39.3, %.loc6_39.5), %return
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.48b) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %Int.loc10_45.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_45.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %pattern_type.loc10_32: type = pattern_type %Int.loc10_45.1 [symbolic = %pattern_type.loc10_32 (constants.%pattern_type.60f8e7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_22: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_22 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_45.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.baf)]
// CHECK:STDOUT:   %require_complete.loc10_60: <witness> = require_complete_type %Int.loc10_45.1 [symbolic = %require_complete.loc10_60 (constants.%require_complete.e66303.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc10_45.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc10_45.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %.loc10_60.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_60.4 (constants.%.20c)]
// CHECK:STDOUT:   %impl.elem0.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%.loc10_60.4 (%.20c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_60.2: <specific function> = specific_impl_function %impl.elem0.loc10_60.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.48b)) -> @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.b50e37.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.48b) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.baf) = name_ref start, @IntRange.%.loc22 [concrete = @IntRange.%.loc22]
// CHECK:STDOUT:     %.loc10_60.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.b50e37.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.b50e37.1) = acquire_value %.loc10_60.1
// CHECK:STDOUT:     %impl.elem0.loc10_60.1: @IntRange.as.Iterate.impl.NewCursor.%.loc10_60.4 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc10_60.1: <bound method> = bound_method %.loc10_60.2, %impl.elem0.loc10_60.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_60.1: <specific function> = specific_impl_function %impl.elem0.loc10_60.1, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc10_60.2: <bound method> = bound_method %.loc10_60.2, %specific_impl_fn.loc10_60.1
// CHECK:STDOUT:     %.loc10_60.3: init @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.b50e37.1) = call %bound_method.loc10_60.2(%.loc10_60.2)
// CHECK:STDOUT:     return %.loc10_60.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %Int.loc11_43.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %ptr.loc11_44.1: type = ptr_type %Int.loc11_43.1 [symbolic = %ptr.loc11_44.1 (constants.%ptr.1cf)]
// CHECK:STDOUT:   %pattern_type.loc11_25: type = pattern_type %ptr.loc11_44.1 [symbolic = %pattern_type.loc11_25 (constants.%pattern_type.054)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.8c6)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_75.1: %OptionalStorage.type = facet_value %Int.loc11_43.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:   %Optional.loc11_75.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:   %pattern_type.loc11_47: type = pattern_type %Optional.loc11_75.1 [symbolic = %pattern_type.loc11_47 (constants.%pattern_type.9f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_47: <witness> = require_complete_type %Optional.loc11_75.1 [symbolic = %require_complete.loc11_47 (constants.%require_complete.b49)]
// CHECK:STDOUT:   %require_complete.loc11_17: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_17 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %ptr.loc11_44.1 [symbolic = %require_complete.loc11_31 (constants.%require_complete.6ca)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %Int.loc11_43.1 [symbolic = %require_complete.loc12 (constants.%require_complete.e66303.1)]
// CHECK:STDOUT:   %pattern_type.loc12: type = pattern_type %Int.loc11_43.1 [symbolic = %pattern_type.loc12 (constants.%pattern_type.60f8e7.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc11_43.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %.loc12_32.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc12_32.4 (constants.%.20c)]
// CHECK:STDOUT:   %impl.elem0.loc12_32.2: @IntRange.as.Iterate.impl.Next.%.loc12_32.4 (%.20c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_32.2: <specific function> = specific_impl_function %impl.elem0.loc12_32.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_43.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.baf)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_43.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.393)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.4f7)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.72a)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72a) = assoc_entity element0, imports.%Core.import_ref.02d [symbolic = %assoc0 (constants.%assoc0.3f6)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.401, @Int.as.OrderedWith.impl.849(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.586)]
// CHECK:STDOUT:   %OrderedWith.Less.type: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.Less.type (constants.%OrderedWith.Less.type.c6f)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.393) = facet_value %Int.loc11_43.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet)]
// CHECK:STDOUT:   %.loc13_17.2: type = fn_type_with_self_type %OrderedWith.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.2 (constants.%.db5)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.849(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.f52)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.f52) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.af3)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ff)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.loc11_43.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %.loc14_9.2: type = fn_type_with_self_type constants.%Inc.Op.type, %Inc.facet [symbolic = %.loc14_9.2 (constants.%.080)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.080) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.923)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.Op(%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.173)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.74c)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.74c) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.37c)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Int.loc11_43.1, () [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.437)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc11_43.1, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc12_7: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc12_7 (constants.%.200)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.353)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @IntRange.as.Iterate.impl.Next.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.353) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.a39)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.a39) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.b0b)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.48b), %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf)) -> %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.60f8e7.1) = ref_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.var_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.60f8e7.1) = var_pattern %value.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = var %value.var_patt
// CHECK:STDOUT:     %cursor.ref.loc12: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = deref %cursor.ref.loc12
// CHECK:STDOUT:     %.loc12_32.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = acquire_value %.loc12_32.1
// CHECK:STDOUT:     %impl.elem0.loc12_32.1: @IntRange.as.Iterate.impl.Next.%.loc12_32.4 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc12_32.1: <bound method> = bound_method %.loc12_32.2, %impl.elem0.loc12_32.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_32.1: <specific function> = specific_impl_function %impl.elem0.loc12_32.1, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc12_32.2: <bound method> = bound_method %.loc12_32.2, %specific_impl_fn.loc12_32.1
// CHECK:STDOUT:     %.loc12_32.3: init @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = call %bound_method.loc12_32.2(%.loc12_32.2)
// CHECK:STDOUT:     assign %value.var, %.loc12_32.3
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = ref_binding value, %value.var
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.48b) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.baf) = name_ref end, @IntRange.%.loc23 [concrete = @IntRange.%.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = acquire_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.b50e37.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.393)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72a) = specific_constant imports.%Core.import_ref.128, @OrderedWith(constants.%Int.b50e37.1) [symbolic = %assoc0 (constants.%assoc0.3f6)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72a) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.3f6)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.2 (%.db5) = impl_witness_access constants.%OrderedWith.impl_witness.586, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.af3)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ff)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = acquire_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc14_11: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = deref %cursor.ref.loc14
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.080) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.923)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %.loc14_11, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.173)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %.loc14_11, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %addr.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = addr_of %.loc14_11
// CHECK:STDOUT:     %.loc14_9.1: init %empty_tuple.type = call %bound_method.loc14_9.2(%addr.loc14)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_41: %OptionalStorage.type = facet_value %Int.loc15, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc15_41: %OptionalStorage.type = converted %Int.loc15, %OptionalStorage.facet.loc15_41 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.loc11_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.74c) = specific_constant imports.%Core.import_ref.961, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.Some (constants.%Optional.Some.37c)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.74c) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.37c)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.1: %OptionalStorage.type = facet_value constants.%Int.b50e37.1, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc15_53.1: %OptionalStorage.type = converted constants.%Int.b50e37.1, %OptionalStorage.facet.loc15_53.1 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.2: %OptionalStorage.type = facet_value constants.%Int.b50e37.1, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc15_53.2: %OptionalStorage.type = converted constants.%Int.b50e37.1, %OptionalStorage.facet.loc15_53.2 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_47.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) = splice_block %return {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.b50e37.1) = acquire_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48) to %.loc11_47.1
// CHECK:STDOUT:     %impl.elem0.loc12_7.1: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.200) = impl_witness_access constants.%Destroy.impl_witness.437, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be)]
// CHECK:STDOUT:     %bound_method.loc12_7.1: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.1
// CHECK:STDOUT:     %specific_fn.loc12_7.1: <specific function> = specific_function %impl.elem0.loc12_7.1, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc12_7.2: <bound method> = bound_method %value.var, %specific_fn.loc12_7.1
// CHECK:STDOUT:     %addr.loc12_7.1: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = addr_of %value.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12_7.1: init %empty_tuple.type = call %bound_method.loc12_7.2(%addr.loc12_7.1)
// CHECK:STDOUT:     return %Optional.Some.call to %return
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc17: %OptionalStorage.type = facet_value %Int.loc17, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc17_41: %OptionalStorage.type = converted %Int.loc17, %OptionalStorage.facet.loc17 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.loc11_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:     %.loc17_42: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.a39) = specific_constant imports.%Core.import_ref.f2f, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.None (constants.%Optional.None.b0b)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.a39) = name_ref None, %.loc17_42 [symbolic = %Optional.None (constants.%Optional.None.b0b)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_47.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) = splice_block %return {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.23f) = call %Optional.None.specific_fn.loc17_42.1() to %.loc11_47.2
// CHECK:STDOUT:     %impl.elem0.loc12_7.2: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.200) = impl_witness_access constants.%Destroy.impl_witness.437, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be)]
// CHECK:STDOUT:     %bound_method.loc12_7.3: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.2
// CHECK:STDOUT:     %specific_fn.loc12_7.2: <specific function> = specific_function %impl.elem0.loc12_7.2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc12_7.4: <bound method> = bound_method %value.var, %specific_fn.loc12_7.2
// CHECK:STDOUT:     %addr.loc12_7.2: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.1cf) = addr_of %value.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12_7.2: init %empty_tuple.type = call %bound_method.loc12_7.4(%addr.loc12_7.2)
// CHECK:STDOUT:     return %Optional.None.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> %return.param: %IntRange.365 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:   %.loc27_22: %IntRange.Make.type.cef = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.0dc]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.cef = name_ref Make, %.loc27_22 [concrete = constants.%IntRange.Make.0dc]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc26_20: ref %IntRange.365 = splice_block %return {}
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.365 = call %IntRange.Make.specific_fn(%.loc27_28.2, %end.ref) to %.loc26_20
// CHECK:STDOUT:   return %IntRange.Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.121
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.14e
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e66303.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.baf
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.cb5
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.4f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.60f8e7.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.035
// CHECK:STDOUT:   %Copy.facet.loc9_85.1 => constants.%Copy.facet.d3d
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e76
// CHECK:STDOUT:   %Iterate.impl_witness => constants.%Iterate.impl_witness
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
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.6de
// CHECK:STDOUT:   %Int.loc10_45.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %pattern_type.loc10_32 => constants.%pattern_type.60f8e7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.6de
// CHECK:STDOUT:   %Int.loc11_43.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %ptr.loc11_44.1 => constants.%ptr.1cf
// CHECK:STDOUT:   %pattern_type.loc11_25 => constants.%pattern_type.054
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.8c6
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_75.1 => constants.%OptionalStorage.facet.ea4
// CHECK:STDOUT:   %Optional.loc11_75.1 => constants.%Optional.23f
// CHECK:STDOUT:   %pattern_type.loc11_47 => constants.%pattern_type.9f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.cef
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.0dc
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.e33
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.c45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.d16
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_49 => constants.%complete_type.c45
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.fb7
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.c50
// CHECK:STDOUT:   %.loc6_22.2 => constants.%.65f
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.dfd
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.f4f02c.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.fb5: type = struct_type {.start: %Int.f4f02c.1, .end: %Int.f4f02c.1} [symbolic]
// CHECK:STDOUT:   %complete_type.655: <witness> = complete_type_witness %struct_type.start.end.fb5 [symbolic]
// CHECK:STDOUT:   %IntRange.48b: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.121: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.14e: %IntRange.Make.type.121 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %pattern_type.e3e33e.1: type = pattern_type %Int.f4f02c.1 [symbolic]
// CHECK:STDOUT:   %require_complete.001e59.1: <witness> = require_complete_type %Int.f4f02c.1 [symbolic]
// CHECK:STDOUT:   %IntRange.elem.4f4: type = unbound_element_type %IntRange.48b, %Int.f4f02c.1 [symbolic]
// CHECK:STDOUT:   %require_complete.b4c: <witness> = require_complete_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d1a: <witness> = lookup_impl_witness %Int.f4f02c.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.f4f02c.1, (%Copy.lookup_impl_witness.d1a) [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.b90: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.305: %.b90 = impl_witness_access %Copy.lookup_impl_witness.d1a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3a3: <specific function> = specific_impl_function %impl.elem0.305, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.365: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.cef: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.0dc: %IntRange.Make.type.cef = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.elem.f21: type = unbound_element_type %IntRange.365, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.0fe: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a43: <witness> = complete_type_witness %struct_type.start.end.0fe [concrete]
// CHECK:STDOUT:   %pattern_type.d16: type = pattern_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.bd9: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8cf: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.09b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.04e, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.069: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.13e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.bd9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.09b) [concrete]
// CHECK:STDOUT:   %.1a5: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.13e [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.13e, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %IntRange.365, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.fef: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.875: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.fef = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.049: type = ptr_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.875, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IntRange: %IntRange.type = import_ref Main//lib, IntRange, loaded [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %Main.Range: %Range.type = import_ref Main//lib, Range, loaded [concrete = constants.%Range]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Main.import_ref.40af26.1: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.80d: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.655)]
// CHECK:STDOUT:   %Main.import_ref.b1b = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.765 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.997 = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.40af26.2: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7d5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8cf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.04e = impl_witness_table (%Core.import_ref.7d5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntRange = imports.%Main.IntRange
// CHECK:STDOUT:     .Range = imports.%Main.Range
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .Read = %Read.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Read.decl: %Read.type = fn_decl @Read [concrete = constants.%Read] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.40af26.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.121)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.121) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.14e)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.f4f02c.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.001e59.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.4f4)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.f4f02c.1), .end: @IntRange.%Int (%Int.f4f02c.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fb5)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.655)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.80d
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b1b
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.765
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.e5d
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.997
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Read() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete = constants.%int_43]
// CHECK:STDOUT:   %.loc5_27.1: type = splice_block %.loc5_27.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core.ece [concrete = imports.%Core.ece]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %IntLiteral.call, %.loc5_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0, %int_43 [symbolic = constants.%y]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d16 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d16 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %IntRange.365 = var %x.var_patt
// CHECK:STDOUT:   %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:   %y.ref: Core.IntLiteral = name_ref y, %y [symbolic = constants.%y]
// CHECK:STDOUT:   %.loc6_3: ref %IntRange.365 = splice_block %x.var {}
// CHECK:STDOUT:   %impl.elem0: %.1a5 = impl_witness_access constants.%ImplicitAs.impl_witness.09b, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.13e]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc6_31.2(%y.ref) [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call]
// CHECK:STDOUT:   %.loc6_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call]
// CHECK:STDOUT:   %.loc6_31.2: %i32 = converted %y.ref, %.loc6_31.1 [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call]
// CHECK:STDOUT:   %Range.call: init %IntRange.365 = call %Range.ref(%.loc6_31.2) to %.loc6_3
// CHECK:STDOUT:   assign %x.var, %Range.call
// CHECK:STDOUT:   %.loc6_21: type = splice_block %IntRange [concrete = constants.%IntRange.365] {
// CHECK:STDOUT:     %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %IntRange.365 = ref_binding x, %x.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %x.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.875
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.875, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_3: <bound method> = bound_method %x.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.049 = addr_of %x.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc6_3(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.40af26.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.f4f02c.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.e3e33e.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Int [symbolic = %require_complete.2 (constants.%require_complete.001e59.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.f4f02c.1), .end: @IntRange.Make.%Int (%Int.f4f02c.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fb5)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d1a)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.1: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.1 (constants.%.b90)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.1 (%.b90) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.305)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.3a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.121
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.14e
// CHECK:STDOUT:   %Int => constants.%Int.f4f02c.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.001e59.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.4f4
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.fb5
// CHECK:STDOUT:   %complete_type => constants.%complete_type.655
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.f4f02c.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.e3e33e.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.cef
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.0dc
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.f21
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.0fe
// CHECK:STDOUT:   %complete_type => constants.%complete_type.a43
// CHECK:STDOUT: }
// CHECK:STDOUT:

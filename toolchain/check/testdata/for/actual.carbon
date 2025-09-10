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
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.349: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.49d0e6.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.8963eb.1: type = pattern_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.dcd: type = pattern_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.51f: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.2ec: %IntRange.Make.type.51f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %.Self.d62: %Iterate.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.02e: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.9e6 [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.d62 [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.6c6: <witness> = lookup_impl_witness %.Self.d62, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.b8a: type = impl_witness_access %Iterate.lookup_impl_witness.6c6, element1 [symbolic_self]
// CHECK:STDOUT:   %assoc0.724: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.4f9 [concrete]
// CHECK:STDOUT:   %impl.elem0.70d: type = impl_witness_access %Iterate.lookup_impl_witness.6c6, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.b8a = %Int.49d0e6.1 and %impl.elem0.70d = %Int.49d0e6.1> [symbolic]
// CHECK:STDOUT:   %require_complete.f08: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.ef2: type = fn_type @Optional.None, @Optional(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Optional.None.fd6: %Optional.None.type.ef2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.b2c: type = fn_type @Optional.Some, @Optional(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Optional.Some.d0d: %Optional.Some.type.b2c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.784: type = ptr_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4dc: type = pattern_type %ptr.784 [symbolic]
// CHECK:STDOUT:   %Optional.708: type = class_type @Optional, @Optional(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.4b1: type = pattern_type %Optional.708 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.b4f426.1: <witness> = require_complete_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %IntRange.elem.e7c: type = unbound_element_type %IntRange.349, %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.606: <witness> = impl_witness @IntRange.%Destroy.impl_witness_table, @IntRange.as.Destroy.impl(%N) [symbolic]
// CHECK:STDOUT:   %ptr.710: type = ptr_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %pattern_type.99f: type = pattern_type %ptr.710 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.type: type = fn_type @IntRange.as.Destroy.impl.Op, @IntRange.as.Destroy.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op: %IntRange.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.78d: type = struct_type {.start: %Int.49d0e6.1, .end: %Int.49d0e6.1} [symbolic]
// CHECK:STDOUT:   %complete_type.9d5: <witness> = complete_type_witness %struct_type.start.end.78d [symbolic]
// CHECK:STDOUT:   %require_complete.524: <witness> = require_complete_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.73a: type = fn_type @Optional.None, @Optional(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %Optional.None.83e: %Optional.None.type.73a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.185: type = fn_type @Optional.Some, @Optional(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %Optional.Some.58a: %Optional.Some.type.185 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.b74: <witness> = require_complete_type %Optional.708 [symbolic]
// CHECK:STDOUT:   %require_complete.0f5: <witness> = require_complete_type %ptr.784 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = bind_symbolic_name Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.f19: type = fn_type @OrderedWith.Less, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.02e: %OrderedWith.Less.type.f19 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.03c: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.e9c: type = facet_type <@OrderedWith, @OrderedWith(%Int.49d0e6.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.8fe: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.d92: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %assoc0.2ae: %OrderedWith.assoc_type.d92 = assoc_entity element0, imports.%Core.import_ref.910 [symbolic]
// CHECK:STDOUT:   %require_complete.309: <witness> = require_complete_type %OrderedWith.type.e9c [symbolic]
// CHECK:STDOUT:   %assoc0.3c6: %OrderedWith.assoc_type.03c = assoc_entity element0, imports.%Core.import_ref.146ebd.1 [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = bind_symbolic_name M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.998: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.c2d(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.807: %Int.as.OrderedWith.impl.Less.type.998 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.e0a: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.d42, @Int.as.OrderedWith.impl.c2d(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.b26: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.c2d(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.ecc: %Int.as.OrderedWith.impl.Less.type.b26 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.e9c = facet_value %Int.49d0e6.1, (%OrderedWith.impl_witness.e0a) [symbolic]
// CHECK:STDOUT:   %.4e7: type = fn_type_with_self_type %OrderedWith.Less.type.8fe, %OrderedWith.facet [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.fd6: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.ecc, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.Op.type: type = fn_type @Inc.Op [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.49d0e6.1, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.49d0e6.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.c05: type = fn_type_with_self_type %Inc.Op.type, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1ab: %.c05 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8a8: <specific function> = specific_impl_function %impl.elem0.1ab, @Inc.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.58a, @Optional.Some(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.b15: <witness> = impl_witness imports.%Destroy.impl_witness_table.fea, @Int.as.Destroy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op: %Int.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.80c: %Destroy.type = facet_value %Int.49d0e6.1, (%Destroy.impl_witness.b15) [symbolic]
// CHECK:STDOUT:   %.11a: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.80c [symbolic]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Destroy.impl.Op, @Int.as.Destroy.impl.Op(%N) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.83e, @Optional.None(%Int.49d0e6.1) [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e4d: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.e4d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Iterate: type = import_ref Core//prelude/iterate, Iterate, loaded [concrete = constants.%Iterate.type]
// CHECK:STDOUT:   %Core.import_ref.1c9: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc12_18, loaded [concrete = constants.%assoc0.724]
// CHECK:STDOUT:   %Core.import_ref.ed6: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc13_17, loaded [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:   %Core.import_ref.9e6: type = import_ref Core//prelude/iterate, loc13_17, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4f9: type = import_ref Core//prelude/iterate, loc12_18, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f49: @Optional.%Optional.None.type (%Optional.None.type.ef2) = import_ref Core//prelude/iterate, inst137 [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.fd6)]
// CHECK:STDOUT:   %Core.import_ref.1a8: @Optional.%Optional.Some.type (%Optional.Some.type.b2c) = import_ref Core//prelude/iterate, inst138 [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.d0d)]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.1cc: @OrderedWith.%OrderedWith.assoc_type (%OrderedWith.assoc_type.03c) = import_ref Core//prelude/operators/comparison, loc26_44, loaded [symbolic = @OrderedWith.%assoc0 (constants.%assoc0.3c6)]
// CHECK:STDOUT:   %Core.import_ref.910: @OrderedWith.%OrderedWith.Less.type (%OrderedWith.Less.type.f19) = import_ref Core//prelude/operators/comparison, loc26_44, loaded [symbolic = @OrderedWith.%OrderedWith.Less (constants.%OrderedWith.Less.02e)]
// CHECK:STDOUT:   %Core.import_ref.146ebd.1 = import_ref Core//prelude/operators/comparison, loc26_44, unloaded
// CHECK:STDOUT:   %Core.import_ref.dea: @Int.as.OrderedWith.impl.c2d.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.998) = import_ref Core//prelude/types/int, loc52_46, loaded [symbolic = @Int.as.OrderedWith.impl.c2d.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.807)]
// CHECK:STDOUT:   %Core.import_ref.ce7 = import_ref Core//prelude/types/int, loc53_58, unloaded
// CHECK:STDOUT:   %Core.import_ref.fbf = import_ref Core//prelude/types/int, loc54_49, unloaded
// CHECK:STDOUT:   %Core.import_ref.0c6 = import_ref Core//prelude/types/int, loc55_61, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.d42 = impl_witness_table (%Core.import_ref.dea, %Core.import_ref.ce7, %Core.import_ref.fbf, %Core.import_ref.0c6), @Int.as.OrderedWith.impl.c2d [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.import_ref.16a: @Int.as.Destroy.impl.%Int.as.Destroy.impl.Op.type (%Int.as.Destroy.impl.Op.type) = import_ref Core//prelude/types/int, loc13_29, loaded [symbolic = @Int.as.Destroy.impl.%Int.as.Destroy.impl.Op (constants.%Int.as.Destroy.impl.Op)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.fea = impl_witness_table (%Core.import_ref.16a), @Int.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/types/int, loc20_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.3: type = converted %IntLiteral.call, %.loc4_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_16.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: %Range.type = fn_decl @Range [concrete = constants.%Range] {
// CHECK:STDOUT:     %end.patt: %pattern_type.7ce = binding_pattern end [concrete]
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
// CHECK:STDOUT:     %end: %i32 = bind_name end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.365 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.365 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.b8a = %Int.loc9_54.1 and constants.%impl.elem0.70d = %Int.loc9_54.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.f08)]
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
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.dcd) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.dcd) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_32 (%pattern_type.8963eb.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_32 (%pattern_type.8963eb.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_45.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_45.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.349) = value_param call_param0
// CHECK:STDOUT:       %.loc10_24.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.349)] {
// CHECK:STDOUT:         %.loc10_24.2: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_24.2 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.349) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.49d0e6.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.49d0e6.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.dcd) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.dcd) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %cursor.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.4dc) = binding_pattern cursor [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.4dc) = value_param_pattern %cursor.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_47 (%pattern_type.4b1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_47 (%pattern_type.4b1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_68: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_73: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_74: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %Optional.loc11_75.2: type = class_type @Optional, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.708)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.349) = value_param call_param0
// CHECK:STDOUT:       %.loc11_19.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.349)] {
// CHECK:STDOUT:         %.loc11_19.2: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_19.2 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.349) = bind_name self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = value_param call_param1
// CHECK:STDOUT:       %.loc11_44: type = splice_block %ptr.loc11_44.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.784)] {
// CHECK:STDOUT:         %Core.ref.loc11_33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_37: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_42: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_43.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:         %ptr.loc11_44.2: type = ptr_type %Int.loc11_43.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.784)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = bind_name cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) = return_slot %return.param
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
// CHECK:STDOUT: generic impl @IntRange.as.Destroy.impl(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @IntRange.%Destroy.impl_witness_table, @IntRange.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.606)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.type: type = fn_type @IntRange.as.Destroy.impl.Op, @IntRange.as.Destroy.impl(%N) [symbolic = %IntRange.as.Destroy.impl.Op.type (constants.%IntRange.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op: @IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op.type (%IntRange.as.Destroy.impl.Op.type) = struct_value () [symbolic = %IntRange.as.Destroy.impl.Op (constants.%IntRange.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @IntRange.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %IntRange.as.Destroy.impl.Op.decl: @IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op.type (%IntRange.as.Destroy.impl.Op.type) = fn_decl @IntRange.as.Destroy.impl.Op [symbolic = @IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op (constants.%IntRange.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Destroy.impl.Op.%pattern_type (%pattern_type.99f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Destroy.impl.Op.%pattern_type (%pattern_type.99f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_39.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @IntRange.as.Destroy.impl.Op.%ptr (%ptr.710) = value_param call_param0
// CHECK:STDOUT:       %.loc4_39.2: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.349)] {
// CHECK:STDOUT:         %.loc4_39.3: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_39.3 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Destroy.impl.Op.%ptr (%ptr.710) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %IntRange.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @IntRange.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_16.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_16.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.51f)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.51f) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.2ec)]
// CHECK:STDOUT:   %Int.loc22_32.2: type = class_type @Int, @Int(%N.loc4_16.1) [symbolic = %Int.loc22_32.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc22_32.2 [symbolic = %require_complete (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_16.1) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc22_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.e7c)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int.loc22_32.2 (%Int.49d0e6.1), .end: @IntRange.%Int.loc22_32.2 (%Int.49d0e6.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.78d)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.9d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.51f) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.2ec)] {
// CHECK:STDOUT:       %start.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = binding_pattern start [concrete]
// CHECK:STDOUT:       %start.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = value_param_pattern %start.patt, call_param0 [concrete]
// CHECK:STDOUT:       %end.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = binding_pattern end [concrete]
// CHECK:STDOUT:       %end.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = value_param_pattern %end.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.dcd) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.dcd) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = bind_name start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = bind_name end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.349) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.349) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.349 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d62]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.d62]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.ed6 [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%Iterate.lookup_impl_witness.6c6, element1 [symbolic_self = constants.%impl.elem1.b8a]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.d62]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.1c9 [concrete = constants.%assoc0.724]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Iterate.lookup_impl_witness.6c6, element0 [symbolic_self = constants.%impl.elem0.70d]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr %.Self [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%Iterate.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %Int.loc9_54.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %Int.loc9_85
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.decl, @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = @IntRange.as.Iterate.impl.%Iterate.impl_witness (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: type = impl_witness_assoc_constant constants.%Int.49d0e6.1 [symbolic = @IntRange.as.Iterate.impl.%Int.loc9_54.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: type = impl_witness_assoc_constant constants.%Int.49d0e6.1 [symbolic = @IntRange.as.Iterate.impl.%Int.loc9_54.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc22: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:     %Int.loc22_32.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %.loc22: @IntRange.%IntRange.elem (%IntRange.elem.e7c) = field_decl start, element0 [concrete]
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc23: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc23: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:     %Int.loc23: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %.loc23: @IntRange.%IntRange.elem (%IntRange.elem.e7c) = field_decl end, element1 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%IntRange.349 [symbolic = @IntRange.as.Destroy.impl.%IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:     impl_decl @IntRange.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op.decl), @IntRange.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @IntRange.as.Destroy.impl(constants.%N) [symbolic = @IntRange.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.606)]
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.78d [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.9d5)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.349
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %.loc22
// CHECK:STDOUT:     .end [private] = %.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %pattern_type.loc5_11: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_11 (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.loc5_49: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_49 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.78d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1)) -> %return.param: @IntRange.Make.%IntRange (%IntRange.349) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.78d) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = initialize_from %start.ref to %.loc6_39.2
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = class_element_access %return, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.49d0e6.1) = initialize_from %end.ref to %.loc6_39.4
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.349) = class_init (%.loc6_39.3, %.loc6_39.5), %return
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.349) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:   %Int.loc10_45.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_45.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %pattern_type.loc10_32: type = pattern_type %Int.loc10_45.1 [symbolic = %pattern_type.loc10_32 (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_22: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_22 (constants.%require_complete.524)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_45.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.e7c)]
// CHECK:STDOUT:   %require_complete.loc10_60: <witness> = require_complete_type %Int.loc10_45.1 [symbolic = %require_complete.loc10_60 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.349)) -> @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.49d0e6.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.349) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.e7c) = name_ref start, @IntRange.%.loc22 [concrete = @IntRange.%.loc22]
// CHECK:STDOUT:     %.loc10_60.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.49d0e6.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.49d0e6.1) = bind_value %.loc10_60.1
// CHECK:STDOUT:     return %.loc10_60.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:   %Int.loc11_43.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %ptr.loc11_44.1: type = ptr_type %Int.loc11_43.1 [symbolic = %ptr.loc11_44.1 (constants.%ptr.784)]
// CHECK:STDOUT:   %pattern_type.loc11_25: type = pattern_type %ptr.loc11_44.1 [symbolic = %pattern_type.loc11_25 (constants.%pattern_type.4dc)]
// CHECK:STDOUT:   %Optional.loc11_75.1: type = class_type @Optional, @Optional(%Int.loc11_43.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.708)]
// CHECK:STDOUT:   %pattern_type.loc11_47: type = pattern_type %Optional.loc11_75.1 [symbolic = %pattern_type.loc11_47 (constants.%pattern_type.4b1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_47: <witness> = require_complete_type %Optional.loc11_75.1 [symbolic = %require_complete.loc11_47 (constants.%require_complete.b74)]
// CHECK:STDOUT:   %require_complete.loc11_17: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_17 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %ptr.loc11_44.1 [symbolic = %require_complete.loc11_31 (constants.%require_complete.0f5)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %Int.loc11_43.1 [symbolic = %require_complete.loc12 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %pattern_type.loc12: type = pattern_type %Int.loc11_43.1 [symbolic = %pattern_type.loc12 (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_43.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.e7c)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_43.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.e9c)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.309)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.d92)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.d92) = assoc_entity element0, imports.%Core.import_ref.910 [symbolic = %assoc0 (constants.%assoc0.2ae)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.d42, @Int.as.OrderedWith.impl.c2d(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.e0a)]
// CHECK:STDOUT:   %OrderedWith.Less.type: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.Less.type (constants.%OrderedWith.Less.type.8fe)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.e9c) = facet_value %Int.loc11_43.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet)]
// CHECK:STDOUT:   %.loc13_17.2: type = fn_type_with_self_type %OrderedWith.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.2 (constants.%.4e7)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.c2d(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.b26)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.b26) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.ecc)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.fd6)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.loc11_43.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %.loc14_9.2: type = fn_type_with_self_type constants.%Inc.Op.type, %Inc.facet [symbolic = %.loc14_9.2 (constants.%.c05)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.c05) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.1ab)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.Op(%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.8a8)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%Int.loc11_43.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.185)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.185) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.58a)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%Int.loc11_43.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.fea, @Int.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.b15)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc11_43.1, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.80c)]
// CHECK:STDOUT:   %.loc12_7: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc12_7 (constants.%.11a)]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%N) [symbolic = %Int.as.Destroy.impl.Op.type (constants.%Int.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op: @IntRange.as.Iterate.impl.Next.%Int.as.Destroy.impl.Op.type (%Int.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Int.as.Destroy.impl.Op (constants.%Int.as.Destroy.impl.Op)]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Destroy.impl.Op, @Int.as.Destroy.impl.Op(%N) [symbolic = %Int.as.Destroy.impl.Op.specific_fn (constants.%Int.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%Int.loc11_43.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.73a)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.73a) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.83e)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%Int.loc11_43.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.349), %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784)) -> %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.8963eb.1) = binding_pattern value [concrete]
// CHECK:STDOUT:       %value.var_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.8963eb.1) = var_pattern %value.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = var %value.var_patt
// CHECK:STDOUT:     %cursor.ref.loc12: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = deref %cursor.ref.loc12
// CHECK:STDOUT:     %.loc12_32.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = bind_value %.loc12_32.1
// CHECK:STDOUT:     assign %value.var, %.loc12_32.2
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = bind_name value, %value.var
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.349) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.e7c) = name_ref end, @IntRange.%.loc23 [concrete = @IntRange.%.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = bind_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.49d0e6.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.e9c)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.d92) = specific_constant imports.%Core.import_ref.1cc, @OrderedWith(constants.%Int.49d0e6.1) [symbolic = %assoc0 (constants.%assoc0.2ae)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.d92) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.2ae)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.2 (%.4e7) = impl_witness_access constants.%OrderedWith.impl_witness.e0a, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.ecc)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.fd6)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = bind_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc14_11: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = deref %cursor.ref.loc14
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.c05) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.1ab)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %.loc14_11, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.8a8)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %.loc14_11, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %addr.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = addr_of %.loc14_11
// CHECK:STDOUT:     %.loc14_9.1: init %empty_tuple.type = call %bound_method.loc14_9.2(%addr.loc14)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.708)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.185) = specific_constant imports.%Core.import_ref.1a8, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.Some (constants.%Optional.Some.58a)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.185) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.58a)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = name_ref value, %value
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%Int.49d0e6.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_47.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) = splice_block %return {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.49d0e6.1) = bind_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48) to %.loc11_47.1
// CHECK:STDOUT:     %impl.elem0.loc12_7.1: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.11a) = impl_witness_access constants.%Destroy.impl_witness.b15, element0 [symbolic = %Int.as.Destroy.impl.Op (constants.%Int.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc12_7.1: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.1
// CHECK:STDOUT:     %specific_fn.loc12_7.1: <specific function> = specific_function %impl.elem0.loc12_7.1, @Int.as.Destroy.impl.Op(constants.%N) [symbolic = %Int.as.Destroy.impl.Op.specific_fn (constants.%Int.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc12_7.2: <bound method> = bound_method %value.var, %specific_fn.loc12_7.1
// CHECK:STDOUT:     %addr.loc12_7.1: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = addr_of %value.var
// CHECK:STDOUT:     %Int.as.Destroy.impl.Op.call.loc12_7.1: init %empty_tuple.type = call %bound_method.loc12_7.2(%addr.loc12_7.1)
// CHECK:STDOUT:     return %Optional.Some.call to %return
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.708)]
// CHECK:STDOUT:     %.loc17: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.73a) = specific_constant imports.%Core.import_ref.f49, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.None (constants.%Optional.None.83e)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.73a) = name_ref None, %.loc17 [symbolic = %Optional.None (constants.%Optional.None.83e)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%Int.49d0e6.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_47.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) = splice_block %return {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.708) = call %Optional.None.specific_fn.loc17_42.1() to %.loc11_47.2
// CHECK:STDOUT:     %impl.elem0.loc12_7.2: @IntRange.as.Iterate.impl.Next.%.loc12_7 (%.11a) = impl_witness_access constants.%Destroy.impl_witness.b15, element0 [symbolic = %Int.as.Destroy.impl.Op (constants.%Int.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc12_7.3: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.2
// CHECK:STDOUT:     %specific_fn.loc12_7.2: <specific function> = specific_function %impl.elem0.loc12_7.2, @Int.as.Destroy.impl.Op(constants.%N) [symbolic = %Int.as.Destroy.impl.Op.specific_fn (constants.%Int.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc12_7.4: <bound method> = bound_method %value.var, %specific_fn.loc12_7.2
// CHECK:STDOUT:     %addr.loc12_7.2: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.784) = addr_of %value.var
// CHECK:STDOUT:     %Int.as.Destroy.impl.Op.call.loc12_7.2: init %empty_tuple.type = call %bound_method.loc12_7.4(%addr.loc12_7.2)
// CHECK:STDOUT:     return %Optional.None.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Destroy.impl.Op(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %ptr: type = ptr_type %IntRange [symbolic = %ptr (constants.%ptr.710)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.99f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Destroy.impl.Op.%ptr (%ptr.710)) = "no_op";
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
// CHECK:STDOUT:   %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
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
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.51f
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.2ec
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b4f426.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.e7c
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.78d
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.9d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.8963eb.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.dcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f08
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
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.dcd
// CHECK:STDOUT:   %Int.loc10_45.1 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %pattern_type.loc10_32 => constants.%pattern_type.8963eb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.dcd
// CHECK:STDOUT:   %Int.loc11_43.1 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %ptr.loc11_44.1 => constants.%ptr.784
// CHECK:STDOUT:   %pattern_type.loc11_25 => constants.%pattern_type.4dc
// CHECK:STDOUT:   %Optional.loc11_75.1 => constants.%Optional.708
// CHECK:STDOUT:   %pattern_type.loc11_47 => constants.%pattern_type.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Destroy.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.606
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Destroy.impl.Op(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %ptr => constants.%ptr.710
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99f
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %y: Core.IntLiteral = bind_symbolic_name y, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int.7ff11f.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.434: type = struct_type {.start: %Int.7ff11f.1, .end: %Int.7ff11f.1} [symbolic]
// CHECK:STDOUT:   %complete_type.c76: <witness> = complete_type_witness %struct_type.start.end.434 [symbolic]
// CHECK:STDOUT:   %IntRange.349: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.51f: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.2ec: %IntRange.Make.type.51f = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.dcd: type = pattern_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %pattern_type.0ede7b.1: type = pattern_type %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %require_complete.ffde5f.1: <witness> = require_complete_type %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %IntRange.elem.ecb: type = unbound_element_type %IntRange.349, %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %require_complete.524: <witness> = require_complete_type %IntRange.349 [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.type.cf3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1d6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ed2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ba4, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cf3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ed2) [concrete]
// CHECK:STDOUT:   %.2c2: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e5b: <witness> = impl_witness imports.%Destroy.impl_witness_table.0cc, @IntRange.as.Destroy.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.type.e28: type = fn_type @IntRange.as.Destroy.impl.Op, @IntRange.as.Destroy.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.a7f: %IntRange.as.Destroy.impl.Op.type.e28 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.710: type = ptr_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %pattern_type.99f: type = pattern_type %ptr.710 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.cc2: <witness> = impl_witness imports.%Destroy.impl_witness_table.0cc, @IntRange.as.Destroy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.type.c5f: type = fn_type @IntRange.as.Destroy.impl.Op, @IntRange.as.Destroy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.754: %IntRange.as.Destroy.impl.Op.type.c5f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.049: type = ptr_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %pattern_type.37a: type = pattern_type %ptr.049 [concrete]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %IntRange.as.Destroy.impl.Op.754, @IntRange.as.Destroy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.f1e294.1: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.30f: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.c76)]
// CHECK:STDOUT:   %Main.import_ref.d13 = import_ref Main//lib, inst42 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d98 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.e58 = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.261 = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.f1e294.2: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0) = import_ref Core//prelude/types/int, loc20_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1d6)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ba4 = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.272: <witness> = import_ref Main//lib, loc4_39, loaded [symbolic = @IntRange.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.e5b)]
// CHECK:STDOUT:   %Main.import_ref.f1e294.3: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.dae: type = import_ref Main//lib, loc4_39, loaded [symbolic = @IntRange.as.Destroy.impl.%IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Main.import_ref.063: type = import_ref Main//lib, inst397 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.302: @IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op.type (%IntRange.as.Destroy.impl.Op.type.e28) = import_ref Main//lib, loc4_39, loaded [symbolic = @IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op (constants.%IntRange.as.Destroy.impl.Op.a7f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.0cc = impl_witness_table (%Main.import_ref.302), @IntRange.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f1e294.4: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
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
// CHECK:STDOUT: generic impl @IntRange.as.Destroy.impl(imports.%Main.import_ref.f1e294.3: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.0cc, @IntRange.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.e5b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.type: type = fn_type @IntRange.as.Destroy.impl.Op, @IntRange.as.Destroy.impl(%N) [symbolic = %IntRange.as.Destroy.impl.Op.type (constants.%IntRange.as.Destroy.impl.Op.type.e28)]
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op: @IntRange.as.Destroy.impl.%IntRange.as.Destroy.impl.Op.type (%IntRange.as.Destroy.impl.Op.type.e28) = struct_value () [symbolic = %IntRange.as.Destroy.impl.Op (constants.%IntRange.as.Destroy.impl.Op.a7f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.dae as imports.%Main.import_ref.063 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.272
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.f1e294.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.51f)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.51f) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.2ec)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.7ff11f.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.ffde5f.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.ecb)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.7ff11f.1), .end: @IntRange.%Int (%Int.7ff11f.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.434)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.c76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.30f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.d13
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.d98
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.e58
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.261
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core.ece [concrete = imports.%Core.ece]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %IntLiteral.call, %.loc5_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: Core.IntLiteral = bind_symbolic_name y, 0, %int_43 [symbolic = constants.%y]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d16 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d16 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %IntRange.365 = var %x.var_patt
// CHECK:STDOUT:   %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:   %y.ref: Core.IntLiteral = name_ref y, %y [symbolic = constants.%y]
// CHECK:STDOUT:   %.loc6_3: ref %IntRange.365 = splice_block %x.var {}
// CHECK:STDOUT:   %impl.elem0: %.2c2 = impl_witness_access constants.%ImplicitAs.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1]
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
// CHECK:STDOUT:   %x: ref %IntRange.365 = bind_name x, %x.var
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %x.var, constants.%IntRange.as.Destroy.impl.Op.754
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%IntRange.as.Destroy.impl.Op.754, @IntRange.as.Destroy.impl.Op(constants.%int_32) [concrete = constants.%IntRange.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_3: <bound method> = bound_method %x.var, %IntRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.049 = addr_of %x.var
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc6_3(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.f1e294.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.7ff11f.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.0ede7b.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Int [symbolic = %require_complete.2 (constants.%require_complete.ffde5f.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.7ff11f.1), .end: @IntRange.Make.%Int (%Int.7ff11f.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.434)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Destroy.impl.Op(imports.%Main.import_ref.f1e294.4: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %ptr: type = ptr_type %IntRange [symbolic = %ptr (constants.%ptr.710)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.99f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.51f
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.2ec
// CHECK:STDOUT:   %Int => constants.%Int.7ff11f.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ffde5f.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.ecb
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.434
// CHECK:STDOUT:   %complete_type => constants.%complete_type.c76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.7ff11f.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.0ede7b.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.dcd
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
// CHECK:STDOUT: specific @IntRange.as.Destroy.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.e5b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Destroy.impl.Op(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %ptr => constants.%ptr.710
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Destroy.impl(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.cc2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op.type => constants.%IntRange.as.Destroy.impl.Op.type.c5f
// CHECK:STDOUT:   %IntRange.as.Destroy.impl.Op => constants.%IntRange.as.Destroy.impl.Op.754
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Destroy.impl.Op(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %ptr => constants.%ptr.049
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.37a
// CHECK:STDOUT: }
// CHECK:STDOUT:

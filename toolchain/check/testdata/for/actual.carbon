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
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.349: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.49d0e6.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.8963eb.1: type = pattern_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.dcd: type = pattern_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %Make.type.51f: type = fn_type @Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Make.2ec: %Make.type.51f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Self.a96: %Iterate.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.ef1: %Iterate.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.02e: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.9e6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.935: type = facet_access_type %.Self.ef1 [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.65a: <witness> = lookup_impl_witness %.Self.ef1, @Iterate [symbolic_self]
// CHECK:STDOUT:   %Iterate.facet.7f1: %Iterate.type = facet_value %.Self.as_type.935, (%Iterate.lookup_impl_witness.65a) [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.164: type = impl_witness_access %Iterate.lookup_impl_witness.65a, element1 [symbolic_self]
// CHECK:STDOUT:   %assoc0.724: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.4f9 [concrete]
// CHECK:STDOUT:   %impl.elem0.19f: type = impl_witness_access %Iterate.lookup_impl_witness.65a, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.164 = %Int.49d0e6.1 and %impl.elem0.19f = %Int.49d0e6.1> [symbolic]
// CHECK:STDOUT:   %require_complete.d96: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %None.type.ef2: type = fn_type @None, @Optional(%T.8b3) [symbolic]
// CHECK:STDOUT:   %None.fd6: %None.type.ef2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Some.type.b2c: type = fn_type @Some, @Optional(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Some.d0d: %Some.type.b2c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %NewCursor.type.b02: type = fn_type @NewCursor.loc10, @Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %NewCursor.f32: %NewCursor.type.b02 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.784: type = ptr_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4dc: type = pattern_type %ptr.784 [symbolic]
// CHECK:STDOUT:   %Optional.708: type = class_type @Optional, @Optional(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.4b1: type = pattern_type %Optional.708 [symbolic]
// CHECK:STDOUT:   %Next.type.2c8: type = fn_type @Next.loc11, @Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Next.259: %Next.type.2c8 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.b4f426.1: <witness> = require_complete_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %IntRange.elem.e7c: type = unbound_element_type %IntRange.349, %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.78d: type = struct_type {.start: %Int.49d0e6.1, .end: %Int.49d0e6.1} [symbolic]
// CHECK:STDOUT:   %complete_type.9d5: <witness> = complete_type_witness %struct_type.start.end.78d [symbolic]
// CHECK:STDOUT:   %require_complete.524: <witness> = require_complete_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %None.type.73a: type = fn_type @None, @Optional(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %None.83e: %None.type.73a = struct_value () [symbolic]
// CHECK:STDOUT:   %Some.type.185: type = fn_type @Some, @Optional(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %Some.58a: %Some.type.185 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.b74: <witness> = require_complete_type %Optional.708 [symbolic]
// CHECK:STDOUT:   %require_complete.0f5: <witness> = require_complete_type %ptr.784 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = bind_symbolic_name Other, 0 [symbolic]
// CHECK:STDOUT:   %Less.type.f19: type = fn_type @Less.1, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %Less.02e: %Less.type.f19 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.03c: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.872: type = facet_type <@OrderedWith, @OrderedWith(%Int.49d0e6.1)> [symbolic]
// CHECK:STDOUT:   %Less.type.8fe: type = fn_type @Less.1, @OrderedWith(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.d92: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %assoc0.2ae: %OrderedWith.assoc_type.d92 = assoc_entity element0, imports.%Core.import_ref.910 [symbolic]
// CHECK:STDOUT:   %require_complete.66b6: <witness> = require_complete_type %OrderedWith.type.872 [symbolic]
// CHECK:STDOUT:   %assoc0.3c6: %OrderedWith.assoc_type.03c = assoc_entity element0, imports.%Core.import_ref.146ebd.1 [symbolic]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = bind_symbolic_name M, 1 [symbolic]
// CHECK:STDOUT:   %Less.type.2c7: type = fn_type @Less.2, @OrderedWith.impl.db3(%N, %M) [symbolic]
// CHECK:STDOUT:   %Less.a5a: %Less.type.2c7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.df5: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.2b2, @OrderedWith.impl.db3(%N, %N) [symbolic]
// CHECK:STDOUT:   %Less.type.98c: type = fn_type @Less.2, @OrderedWith.impl.db3(%N, %N) [symbolic]
// CHECK:STDOUT:   %Less.4be: %Less.type.98c = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.872 = facet_value %Int.49d0e6.1, (%OrderedWith.impl_witness.df5) [symbolic]
// CHECK:STDOUT:   %.99c: type = fn_type_with_self_type %Less.type.8fe, %OrderedWith.facet [symbolic]
// CHECK:STDOUT:   %Less.specific_fn.bb3: <specific function> = specific_function %Less.4be, @Less.2(%N, %N) [symbolic]
// CHECK:STDOUT:   %Op.type.e3a: type = fn_type @Op.95 [concrete]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.49d0e6.1, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.49d0e6.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.a0d: type = fn_type_with_self_type %Op.type.e3a, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.437: %.a0d = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a5f: <specific function> = specific_impl_function %impl.elem0.437, @Op.95(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Some.specific_fn: <specific function> = specific_function %Some.58a, @Some(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.96 [concrete]
// CHECK:STDOUT:   %ptr.2aa: type = ptr_type %Optional.708 [symbolic]
// CHECK:STDOUT:   %require_complete.a5e: <witness> = require_complete_type %ptr.2aa [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f9d: <witness> = lookup_impl_witness %Optional.708, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.be9: %Destroy.type = facet_value %Optional.708, (%Destroy.lookup_impl_witness.f9d) [symbolic]
// CHECK:STDOUT:   %.1ee: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.be9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5fd: %.1ee = impl_witness_access %Destroy.lookup_impl_witness.f9d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.79e: <specific function> = specific_impl_function %impl.elem0.5fd, @Op.96(%Destroy.facet.be9) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.34a: <witness> = lookup_impl_witness %Int.49d0e6.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.13d: %Destroy.type = facet_value %Int.49d0e6.1, (%Destroy.lookup_impl_witness.34a) [symbolic]
// CHECK:STDOUT:   %.bbc: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.13d [symbolic]
// CHECK:STDOUT:   %impl.elem0.fe8: %.bbc = impl_witness_access %Destroy.lookup_impl_witness.34a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f4e: <specific function> = specific_impl_function %impl.elem0.fe8, @Op.96(%Destroy.facet.13d) [symbolic]
// CHECK:STDOUT:   %None.specific_fn: <specific function> = specific_function %None.83e, @None(%Int.49d0e6.1) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntRange.365: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d16: type = pattern_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type.cef: type = fn_type @Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %Make.0dc: %Make.type.cef = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.elem.e33: type = unbound_element_type %IntRange.365, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.d0a: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c45: <witness> = complete_type_witness %struct_type.start.end.d0a [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.0dc, @Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Op.type.2d1: type = fn_type @Op.97, @Destroy.impl(%IntRange.365) [concrete]
// CHECK:STDOUT:   %Op.fa8: %Op.type.2d1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.049: type = ptr_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.df8: <specific function> = specific_function %Op.fa8, @Op.97(%IntRange.365) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .Optional = %Core.Optional
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
// CHECK:STDOUT:   %Core.import_ref.1c9: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc10_18, loaded [concrete = constants.%assoc0.724]
// CHECK:STDOUT:   %Core.import_ref.ed6: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc11_17, loaded [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:   %Core.import_ref.9e6: type = import_ref Core//prelude/iterate, loc11_17, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4f9: type = import_ref Core//prelude/iterate, loc10_18, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.2b7: @Optional.%None.type (%None.type.ef2) = import_ref Core//prelude/iterate, inst96 [indirect], loaded [symbolic = @Optional.%None (constants.%None.fd6)]
// CHECK:STDOUT:   %Core.import_ref.5c6: @Optional.%Some.type (%Some.type.b2c) = import_ref Core//prelude/iterate, inst97 [indirect], loaded [symbolic = @Optional.%Some (constants.%Some.d0d)]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.1cc: @OrderedWith.%OrderedWith.assoc_type (%OrderedWith.assoc_type.03c) = import_ref Core//prelude/operators/comparison, loc26_44, loaded [symbolic = @OrderedWith.%assoc0 (constants.%assoc0.3c6)]
// CHECK:STDOUT:   %Core.import_ref.910: @OrderedWith.%Less.type (%Less.type.f19) = import_ref Core//prelude/operators/comparison, loc26_44, loaded [symbolic = @OrderedWith.%Less (constants.%Less.02e)]
// CHECK:STDOUT:   %Core.import_ref.146ebd.1 = import_ref Core//prelude/operators/comparison, loc26_44, unloaded
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a5f: @OrderedWith.impl.db3.%Less.type (%Less.type.2c7) = import_ref Core//prelude/types/int, loc51_46, loaded [symbolic = @OrderedWith.impl.db3.%Less (constants.%Less.a5a)]
// CHECK:STDOUT:   %Core.import_ref.a58 = import_ref Core//prelude/types/int, loc52_58, unloaded
// CHECK:STDOUT:   %Core.import_ref.a39 = import_ref Core//prelude/types/int, loc53_49, unloaded
// CHECK:STDOUT:   %Core.import_ref.c4c = import_ref Core//prelude/types/int, loc54_61, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.2b2 = impl_witness_table (%Core.import_ref.a5f, %Core.import_ref.a58, %Core.import_ref.a39, %Core.import_ref.c4c), @OrderedWith.impl.db3 [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.3: type = converted %int_literal.make_type, %.loc4_36.2 [concrete = Core.IntLiteral]
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
// CHECK:STDOUT: generic impl @Iterate.impl(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Int.loc9_54.2: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.164 = %Int.loc9_54.2 and constants.%impl.elem0.19f = %Int.loc9_54.2> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.d96)]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @Iterate.impl(%N) [symbolic = %Iterate.impl_witness (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NewCursor.type: type = fn_type @NewCursor.loc10, @Iterate.impl(%N) [symbolic = %NewCursor.type (constants.%NewCursor.type.b02)]
// CHECK:STDOUT:   %NewCursor: @Iterate.impl.%NewCursor.type (%NewCursor.type.b02) = struct_value () [symbolic = %NewCursor (constants.%NewCursor.f32)]
// CHECK:STDOUT:   %Next.type: type = fn_type @Next.loc11, @Iterate.impl(%N) [symbolic = %Next.type (constants.%Next.type.2c8)]
// CHECK:STDOUT:   %Next: @Iterate.impl.%Next.type (%Next.type.2c8) = struct_value () [symbolic = %Next (constants.%Next.259)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %.loc9_24 {
// CHECK:STDOUT:     %NewCursor.decl: @Iterate.impl.%NewCursor.type (%NewCursor.type.b02) = fn_decl @NewCursor.loc10 [symbolic = @Iterate.impl.%NewCursor (constants.%NewCursor.f32)] {
// CHECK:STDOUT:       %self.patt: @NewCursor.loc10.%pattern_type.loc10_18 (%pattern_type.dcd) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @NewCursor.loc10.%pattern_type.loc10_18 (%pattern_type.dcd) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @NewCursor.loc10.%pattern_type.loc10_32 (%pattern_type.8963eb.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @NewCursor.loc10.%pattern_type.loc10_32 (%pattern_type.8963eb.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_45.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_45.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %self.param: @NewCursor.loc10.%IntRange (%IntRange.349) = value_param call_param0
// CHECK:STDOUT:       %.loc10_24.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.349)] {
// CHECK:STDOUT:         %.loc10_24.2: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_24.2 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @NewCursor.loc10.%IntRange (%IntRange.349) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @NewCursor.loc10.%Int.loc10_45.2 (%Int.49d0e6.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @NewCursor.loc10.%Int.loc10_45.2 (%Int.49d0e6.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Next.decl: @Iterate.impl.%Next.type (%Next.type.2c8) = fn_decl @Next.loc11 [symbolic = @Iterate.impl.%Next (constants.%Next.259)] {
// CHECK:STDOUT:       %self.patt: @Next.loc11.%pattern_type.loc11_13 (%pattern_type.dcd) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Next.loc11.%pattern_type.loc11_13 (%pattern_type.dcd) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %cursor.patt: @Next.loc11.%pattern_type.loc11_25 (%pattern_type.4dc) = binding_pattern cursor [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @Next.loc11.%pattern_type.loc11_25 (%pattern_type.4dc) = value_param_pattern %cursor.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @Next.loc11.%pattern_type.loc11_47 (%pattern_type.4b1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Next.loc11.%pattern_type.loc11_47 (%pattern_type.4b1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_68: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_73: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_74: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %Optional.loc11_75.1: type = class_type @Optional, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.loc11_75.2 (constants.%Optional.708)]
// CHECK:STDOUT:       %self.param: @Next.loc11.%IntRange (%IntRange.349) = value_param call_param0
// CHECK:STDOUT:       %.loc11_19.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.349)] {
// CHECK:STDOUT:         %.loc11_19.2: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_19.2 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Next.loc11.%IntRange (%IntRange.349) = bind_name self, %self.param
// CHECK:STDOUT:       %cursor.param: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = value_param call_param1
// CHECK:STDOUT:       %.loc11_44: type = splice_block %ptr.loc11_44.1 [symbolic = %ptr.loc11_44.2 (constants.%ptr.784)] {
// CHECK:STDOUT:         %Core.ref.loc11_33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_37: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_42: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_43.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:         %ptr.loc11_44.1: type = ptr_type %Int.loc11_43.1 [symbolic = %ptr.loc11_44.2 (constants.%ptr.784)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = bind_name cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @Next.loc11.%Optional.loc11_75.2 (%Optional.708) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Next.loc11.%Optional.loc11_75.2 (%Optional.708) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .NewCursor = %NewCursor.decl
// CHECK:STDOUT:     .Next = %Next.decl
// CHECK:STDOUT:     witness = @IntRange.%Iterate.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_16.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @IntRange(%N.loc4_16.1) [symbolic = %Make.type (constants.%Make.type.51f)]
// CHECK:STDOUT:   %Make: @IntRange.%Make.type (%Make.type.51f) = struct_value () [symbolic = %Make (constants.%Make.2ec)]
// CHECK:STDOUT:   %Int.loc22_32.2: type = class_type @Int, @Int(%N.loc4_16.1) [symbolic = %Int.loc22_32.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc22_32.2 [symbolic = %require_complete (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_16.1) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc22_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.e7c)]
// CHECK:STDOUT:   %struct_type.start.end.loc24_1.2: type = struct_type {.start: @IntRange.%Int.loc22_32.2 (%Int.49d0e6.1), .end: @IntRange.%Int.loc22_32.2 (%Int.49d0e6.1)} [symbolic = %struct_type.start.end.loc24_1.2 (constants.%struct_type.start.end.78d)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end.loc24_1.2 [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.9d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @IntRange.%Make.type (%Make.type.51f) = fn_decl @Make [symbolic = @IntRange.%Make (constants.%Make.2ec)] {
// CHECK:STDOUT:       %start.patt: @Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = binding_pattern start [concrete]
// CHECK:STDOUT:       %start.param_patt: @Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = value_param_pattern %start.patt, call_param0 [concrete]
// CHECK:STDOUT:       %end.patt: @Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = binding_pattern end [concrete]
// CHECK:STDOUT:       %end.param_patt: @Make.%pattern_type.loc5_11 (%pattern_type.8963eb.1) = value_param_pattern %end.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @Make.%pattern_type.loc5_49 (%pattern_type.dcd) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Make.%pattern_type.loc5_49 (%pattern_type.dcd) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52: type = specific_constant constants.%IntRange.349, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       %start.param: @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.1 [symbolic = %Int.loc5_28.2 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = bind_name start, %start.param
// CHECK:STDOUT:       %end.param: @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.2 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = bind_name end, %end.param
// CHECK:STDOUT:       %return.param: ref @Make.%IntRange (%IntRange.349) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Make.%IntRange (%IntRange.349) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.349 [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.ef1]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.ef1]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.ed6 [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.as_type.935]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.as_type.935]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%Iterate.lookup_impl_witness.65a, element1 [symbolic_self = constants.%impl.elem1.164]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.ef1]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.1c9 [concrete = constants.%assoc0.724]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.as_type.935]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.as_type.935]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Iterate.lookup_impl_witness.65a, element0 [symbolic_self = constants.%impl.elem0.19f]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr %.Self [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %Int.loc9_54.1
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %Int.loc9_85
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, @Iterate.impl.%NewCursor.decl, @Iterate.impl.%Next.decl), @Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness: <witness> = impl_witness %Iterate.impl_witness_table, @Iterate.impl(constants.%N) [symbolic = @Iterate.impl.%Iterate.impl_witness (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: type = impl_witness_assoc_constant constants.%Int.49d0e6.1 [symbolic = @Iterate.impl.%Int.loc9_54.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: type = impl_witness_assoc_constant constants.%Int.49d0e6.1 [symbolic = @Iterate.impl.%Int.loc9_54.2 (constants.%Int.49d0e6.1)]
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
// CHECK:STDOUT:     %struct_type.start.end.loc24_1.1: type = struct_type {.start: %Int.49d0e6.1, .end: %Int.49d0e6.1} [symbolic = %struct_type.start.end.loc24_1.2 (constants.%struct_type.start.end.78d)]
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness %struct_type.start.end.loc24_1.1 [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.9d5)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.349
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .start [private] = %.loc22
// CHECK:STDOUT:     .end [private] = %.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.2: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %pattern_type.loc5_11: type = pattern_type %Int.loc5_28.2 [symbolic = %pattern_type.loc5_11 (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.loc5_49: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_49 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.2 [symbolic = %require_complete.loc5_16 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @Make.%Int.loc5_28.2 (%Int.49d0e6.1), .end: @Make.%Int.loc5_28.2 (%Int.49d0e6.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.78d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @Make.%Int.loc5_28.2 (%Int.49d0e6.1), %end.param: @Make.%Int.loc5_28.2 (%Int.49d0e6.1)) -> %return.param: @Make.%IntRange (%IntRange.349) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @Make.%struct_type.start.end (%struct_type.start.end.78d) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc6_39.3: init @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = initialize_from %start.ref to %.loc6_39.2
// CHECK:STDOUT:     %.loc6_39.4: ref @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = class_element_access %return, element1
// CHECK:STDOUT:     %.loc6_39.5: init @Make.%Int.loc5_28.2 (%Int.49d0e6.1) = initialize_from %end.ref to %.loc6_39.4
// CHECK:STDOUT:     %.loc6_39.6: init @Make.%IntRange (%IntRange.349) = class_init (%.loc6_39.3, %.loc6_39.5), %return
// CHECK:STDOUT:     %.loc6_40: init @Make.%IntRange (%IntRange.349) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NewCursor.loc10(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:   %Int.loc10_45.2: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_45.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %pattern_type.loc10_32: type = pattern_type %Int.loc10_45.2 [symbolic = %pattern_type.loc10_32 (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_22: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_22 (constants.%require_complete.524)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_45.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.e7c)]
// CHECK:STDOUT:   %require_complete.loc10_60: <witness> = require_complete_type %Int.loc10_45.2 [symbolic = %require_complete.loc10_60 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @NewCursor.loc10.%IntRange (%IntRange.349)) -> @NewCursor.loc10.%Int.loc10_45.2 (%Int.49d0e6.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @NewCursor.loc10.%IntRange (%IntRange.349) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @NewCursor.loc10.%IntRange.elem (%IntRange.elem.e7c) = name_ref start, @IntRange.%.loc22 [concrete = @IntRange.%.loc22]
// CHECK:STDOUT:     %.loc10_60.1: ref @NewCursor.loc10.%Int.loc10_45.2 (%Int.49d0e6.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_60.2: @NewCursor.loc10.%Int.loc10_45.2 (%Int.49d0e6.1) = bind_value %.loc10_60.1
// CHECK:STDOUT:     return %.loc10_60.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Next.loc11(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:   %Int.loc11_43.2: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %ptr.loc11_44.2: type = ptr_type %Int.loc11_43.2 [symbolic = %ptr.loc11_44.2 (constants.%ptr.784)]
// CHECK:STDOUT:   %pattern_type.loc11_25: type = pattern_type %ptr.loc11_44.2 [symbolic = %pattern_type.loc11_25 (constants.%pattern_type.4dc)]
// CHECK:STDOUT:   %Optional.loc11_75.2: type = class_type @Optional, @Optional(%Int.loc11_43.2) [symbolic = %Optional.loc11_75.2 (constants.%Optional.708)]
// CHECK:STDOUT:   %pattern_type.loc11_47: type = pattern_type %Optional.loc11_75.2 [symbolic = %pattern_type.loc11_47 (constants.%pattern_type.4b1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_47.1: <witness> = require_complete_type %Optional.loc11_75.2 [symbolic = %require_complete.loc11_47.1 (constants.%require_complete.b74)]
// CHECK:STDOUT:   %require_complete.loc11_17: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_17 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %ptr.loc11_44.2 [symbolic = %require_complete.loc11_31 (constants.%require_complete.0f5)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %Int.loc11_43.2 [symbolic = %require_complete.loc12 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %pattern_type.loc12: type = pattern_type %Int.loc11_43.2 [symbolic = %pattern_type.loc12 (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_43.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.e7c)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_43.2)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.872)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.66b6)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_43.2) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.d92)]
// CHECK:STDOUT:   %assoc0: @Next.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.d92) = assoc_entity element0, imports.%Core.import_ref.910 [symbolic = %assoc0 (constants.%assoc0.2ae)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.2b2, @OrderedWith.impl.db3(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.df5)]
// CHECK:STDOUT:   %Less.type.loc13_17.1: type = fn_type @Less.1, @OrderedWith(%Int.loc11_43.2) [symbolic = %Less.type.loc13_17.1 (constants.%Less.type.8fe)]
// CHECK:STDOUT:   %OrderedWith.facet: @Next.loc11.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.872) = facet_value %Int.loc11_43.2, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet)]
// CHECK:STDOUT:   %.loc13_17.2: type = fn_type_with_self_type %Less.type.loc13_17.1, %OrderedWith.facet [symbolic = %.loc13_17.2 (constants.%.99c)]
// CHECK:STDOUT:   %Less.type.loc13_17.2: type = fn_type @Less.2, @OrderedWith.impl.db3(%N, %N) [symbolic = %Less.type.loc13_17.2 (constants.%Less.type.98c)]
// CHECK:STDOUT:   %Less: @Next.loc11.%Less.type.loc13_17.2 (%Less.type.98c) = struct_value () [symbolic = %Less (constants.%Less.4be)]
// CHECK:STDOUT:   %Less.specific_fn: <specific function> = specific_function %Less, @Less.2(%N, %N) [symbolic = %Less.specific_fn (constants.%Less.specific_fn.bb3)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.2, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.loc11_43.2, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %.loc14_9.2: type = fn_type_with_self_type constants.%Op.type.e3a, %Inc.facet [symbolic = %.loc14_9.2 (constants.%.a0d)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @Next.loc11.%.loc14_9.2 (%.a0d) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.437)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Op.95(%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.a5f)]
// CHECK:STDOUT:   %Some.type: type = fn_type @Some, @Optional(%Int.loc11_43.2) [symbolic = %Some.type (constants.%Some.type.185)]
// CHECK:STDOUT:   %Some: @Next.loc11.%Some.type (%Some.type.185) = struct_value () [symbolic = %Some (constants.%Some.58a)]
// CHECK:STDOUT:   %Some.specific_fn.loc15_42.2: <specific function> = specific_function %Some, @Some(%Int.loc11_43.2) [symbolic = %Some.specific_fn.loc15_42.2 (constants.%Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc11: <witness> = lookup_impl_witness %Optional.loc11_75.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc11 (constants.%Destroy.lookup_impl_witness.f9d)]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Optional.loc11_75.2, (%Destroy.lookup_impl_witness.loc11) [symbolic = %Destroy.facet.loc11 (constants.%Destroy.facet.be9)]
// CHECK:STDOUT:   %.loc11_47.6: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc11 [symbolic = %.loc11_47.6 (constants.%.1ee)]
// CHECK:STDOUT:   %impl.elem0.loc11_47.4: @Next.loc11.%.loc11_47.6 (%.1ee) = impl_witness_access %Destroy.lookup_impl_witness.loc11, element0 [symbolic = %impl.elem0.loc11_47.4 (constants.%impl.elem0.5fd)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_47.4: <specific function> = specific_impl_function %impl.elem0.loc11_47.4, @Op.96(%Destroy.facet.loc11) [symbolic = %specific_impl_fn.loc11_47.4 (constants.%specific_impl_fn.79e)]
// CHECK:STDOUT:   %ptr.loc11_47: type = ptr_type %Optional.loc11_75.2 [symbolic = %ptr.loc11_47 (constants.%ptr.2aa)]
// CHECK:STDOUT:   %require_complete.loc11_47.2: <witness> = require_complete_type %ptr.loc11_47 [symbolic = %require_complete.loc11_47.2 (constants.%require_complete.a5e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc12: <witness> = lookup_impl_witness %Int.loc11_43.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc12 (constants.%Destroy.lookup_impl_witness.34a)]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Int.loc11_43.2, (%Destroy.lookup_impl_witness.loc12) [symbolic = %Destroy.facet.loc12 (constants.%Destroy.facet.13d)]
// CHECK:STDOUT:   %.loc12_7.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc12 [symbolic = %.loc12_7.3 (constants.%.bbc)]
// CHECK:STDOUT:   %impl.elem0.loc12_7.3: @Next.loc11.%.loc12_7.3 (%.bbc) = impl_witness_access %Destroy.lookup_impl_witness.loc12, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.fe8)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_7.3: <specific function> = specific_impl_function %impl.elem0.loc12_7.3, @Op.96(%Destroy.facet.loc12) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.f4e)]
// CHECK:STDOUT:   %None.type: type = fn_type @None, @Optional(%Int.loc11_43.2) [symbolic = %None.type (constants.%None.type.73a)]
// CHECK:STDOUT:   %None: @Next.loc11.%None.type (%None.type.73a) = struct_value () [symbolic = %None (constants.%None.83e)]
// CHECK:STDOUT:   %None.specific_fn.loc17_42.2: <specific function> = specific_function %None, @None(%Int.loc11_43.2) [symbolic = %None.specific_fn.loc17_42.2 (constants.%None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Next.loc11.%IntRange (%IntRange.349), %cursor.param: @Next.loc11.%ptr.loc11_44.2 (%ptr.784)) -> %return.param: @Next.loc11.%Optional.loc11_75.2 (%Optional.708) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt: @Next.loc11.%pattern_type.loc12 (%pattern_type.8963eb.1) = binding_pattern value [concrete]
// CHECK:STDOUT:       %value.var_patt: @Next.loc11.%pattern_type.loc12 (%pattern_type.8963eb.1) = var_pattern %value.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = var %value.var_patt
// CHECK:STDOUT:     %cursor.ref.loc12: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = deref %cursor.ref.loc12
// CHECK:STDOUT:     %.loc12_32.2: @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = bind_value %.loc12_32.1
// CHECK:STDOUT:     assign %value.var, %.loc12_32.2
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = bind_name value, %value.var
// CHECK:STDOUT:     %value.ref.loc13: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @Next.loc11.%IntRange (%IntRange.349) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @Next.loc11.%IntRange.elem (%IntRange.elem.e7c) = name_ref end, @IntRange.%.loc23 [concrete = @IntRange.%.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = bind_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.49d0e6.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.872)]
// CHECK:STDOUT:     %.loc13_17.1: @Next.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.d92) = specific_constant imports.%Core.import_ref.1cc, @OrderedWith(constants.%Int.49d0e6.1) [symbolic = %assoc0 (constants.%assoc0.2ae)]
// CHECK:STDOUT:     %Less.ref: @Next.loc11.%OrderedWith.assoc_type (%OrderedWith.assoc_type.d92) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.2ae)]
// CHECK:STDOUT:     %impl.elem0.loc13: @Next.loc11.%.loc13_17.2 (%.99c) = impl_witness_access constants.%OrderedWith.impl_witness.df5, element0 [symbolic = %Less (constants.%Less.4be)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13, @Less.2(constants.%N, constants.%N) [symbolic = %Less.specific_fn (constants.%Less.specific_fn.bb3)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn
// CHECK:STDOUT:     %.loc13_11: @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = bind_value %value.ref.loc13
// CHECK:STDOUT:     %int.less: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %int.less
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %int.less, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc14_11: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = deref %cursor.ref.loc14
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @Next.loc11.%.loc14_9.2 (%.a0d) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.437)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %.loc14_11, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Op.95(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.a5f)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %.loc14_11, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %addr.loc14: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = addr_of %.loc14_11
// CHECK:STDOUT:     %.loc14_9.1: init %empty_tuple.type = call %bound_method.loc14_9.2(%addr.loc14)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.loc11_75.2 (constants.%Optional.708)]
// CHECK:STDOUT:     %.loc15_42: @Next.loc11.%Some.type (%Some.type.185) = specific_constant imports.%Core.import_ref.5c6, @Optional(constants.%Int.49d0e6.1) [symbolic = %Some (constants.%Some.58a)]
// CHECK:STDOUT:     %Some.ref: @Next.loc11.%Some.type (%Some.type.185) = name_ref Some, %.loc15_42 [symbolic = %Some (constants.%Some.58a)]
// CHECK:STDOUT:     %value.ref.loc15: ref @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = name_ref value, %value
// CHECK:STDOUT:     %Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Some(constants.%Int.49d0e6.1) [symbolic = %Some.specific_fn.loc15_42.2 (constants.%Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_47.1: ref @Next.loc11.%Optional.loc11_75.2 (%Optional.708) = splice_block %return {}
// CHECK:STDOUT:     %.loc15_48: @Next.loc11.%Int.loc11_43.2 (%Int.49d0e6.1) = bind_value %value.ref.loc15
// CHECK:STDOUT:     %Some.call: init @Next.loc11.%Optional.loc11_75.2 (%Optional.708) = call %Some.specific_fn.loc15_42.1(%.loc15_48) to %.loc11_47.1
// CHECK:STDOUT:     %impl.elem0.loc11_47.1: @Next.loc11.%.loc11_47.6 (%.1ee) = impl_witness_access constants.%Destroy.lookup_impl_witness.f9d, element0 [symbolic = %impl.elem0.loc11_47.4 (constants.%impl.elem0.5fd)]
// CHECK:STDOUT:     %bound_method.loc11_47.1: <bound method> = bound_method %.loc11_47.1, %impl.elem0.loc11_47.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_47.1: <specific function> = specific_impl_function %impl.elem0.loc11_47.1, @Op.96(constants.%Destroy.facet.be9) [symbolic = %specific_impl_fn.loc11_47.4 (constants.%specific_impl_fn.79e)]
// CHECK:STDOUT:     %bound_method.loc11_47.2: <bound method> = bound_method %.loc11_47.1, %specific_impl_fn.loc11_47.1
// CHECK:STDOUT:     %addr.loc11_47.1: @Next.loc11.%ptr.loc11_47 (%ptr.2aa) = addr_of %.loc11_47.1
// CHECK:STDOUT:     %.loc11_47.2: init %empty_tuple.type = call %bound_method.loc11_47.2(%addr.loc11_47.1)
// CHECK:STDOUT:     %impl.elem0.loc12_7.1: @Next.loc11.%.loc12_7.3 (%.bbc) = impl_witness_access constants.%Destroy.lookup_impl_witness.34a, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.fe8)]
// CHECK:STDOUT:     %bound_method.loc12_7.1: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.1: <specific function> = specific_impl_function %impl.elem0.loc12_7.1, @Op.96(constants.%Destroy.facet.13d) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.f4e)]
// CHECK:STDOUT:     %bound_method.loc12_7.2: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.1
// CHECK:STDOUT:     %addr.loc12_7.1: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = addr_of %value.var
// CHECK:STDOUT:     %.loc12_7.1: init %empty_tuple.type = call %bound_method.loc12_7.2(%addr.loc12_7.1)
// CHECK:STDOUT:     return %Some.call to %return
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%Int.49d0e6.1) [symbolic = %Optional.loc11_75.2 (constants.%Optional.708)]
// CHECK:STDOUT:     %.loc17: @Next.loc11.%None.type (%None.type.73a) = specific_constant imports.%Core.import_ref.2b7, @Optional(constants.%Int.49d0e6.1) [symbolic = %None (constants.%None.83e)]
// CHECK:STDOUT:     %None.ref: @Next.loc11.%None.type (%None.type.73a) = name_ref None, %.loc17 [symbolic = %None (constants.%None.83e)]
// CHECK:STDOUT:     %None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @None(constants.%Int.49d0e6.1) [symbolic = %None.specific_fn.loc17_42.2 (constants.%None.specific_fn)]
// CHECK:STDOUT:     %.loc11_47.3: ref @Next.loc11.%Optional.loc11_75.2 (%Optional.708) = splice_block %return {}
// CHECK:STDOUT:     %None.call: init @Next.loc11.%Optional.loc11_75.2 (%Optional.708) = call %None.specific_fn.loc17_42.1() to %.loc11_47.3
// CHECK:STDOUT:     %impl.elem0.loc11_47.2: @Next.loc11.%.loc11_47.6 (%.1ee) = impl_witness_access constants.%Destroy.lookup_impl_witness.f9d, element0 [symbolic = %impl.elem0.loc11_47.4 (constants.%impl.elem0.5fd)]
// CHECK:STDOUT:     %bound_method.loc11_47.3: <bound method> = bound_method %.loc11_47.3, %impl.elem0.loc11_47.2
// CHECK:STDOUT:     %specific_impl_fn.loc11_47.2: <specific function> = specific_impl_function %impl.elem0.loc11_47.2, @Op.96(constants.%Destroy.facet.be9) [symbolic = %specific_impl_fn.loc11_47.4 (constants.%specific_impl_fn.79e)]
// CHECK:STDOUT:     %bound_method.loc11_47.4: <bound method> = bound_method %.loc11_47.3, %specific_impl_fn.loc11_47.2
// CHECK:STDOUT:     %addr.loc11_47.2: @Next.loc11.%ptr.loc11_47 (%ptr.2aa) = addr_of %.loc11_47.3
// CHECK:STDOUT:     %.loc11_47.4: init %empty_tuple.type = call %bound_method.loc11_47.4(%addr.loc11_47.2)
// CHECK:STDOUT:     %impl.elem0.loc11_47.3: @Next.loc11.%.loc11_47.6 (%.1ee) = impl_witness_access constants.%Destroy.lookup_impl_witness.f9d, element0 [symbolic = %impl.elem0.loc11_47.4 (constants.%impl.elem0.5fd)]
// CHECK:STDOUT:     %bound_method.loc11_47.5: <bound method> = bound_method %.loc11_47.1, %impl.elem0.loc11_47.3
// CHECK:STDOUT:     %specific_impl_fn.loc11_47.3: <specific function> = specific_impl_function %impl.elem0.loc11_47.3, @Op.96(constants.%Destroy.facet.be9) [symbolic = %specific_impl_fn.loc11_47.4 (constants.%specific_impl_fn.79e)]
// CHECK:STDOUT:     %bound_method.loc11_47.6: <bound method> = bound_method %.loc11_47.1, %specific_impl_fn.loc11_47.3
// CHECK:STDOUT:     %addr.loc11_47.3: @Next.loc11.%ptr.loc11_47 (%ptr.2aa) = addr_of %.loc11_47.1
// CHECK:STDOUT:     %.loc11_47.5: init %empty_tuple.type = call %bound_method.loc11_47.6(%addr.loc11_47.3)
// CHECK:STDOUT:     %impl.elem0.loc12_7.2: @Next.loc11.%.loc12_7.3 (%.bbc) = impl_witness_access constants.%Destroy.lookup_impl_witness.34a, element0 [symbolic = %impl.elem0.loc12_7.3 (constants.%impl.elem0.fe8)]
// CHECK:STDOUT:     %bound_method.loc12_7.3: <bound method> = bound_method %value.var, %impl.elem0.loc12_7.2
// CHECK:STDOUT:     %specific_impl_fn.loc12_7.2: <specific function> = specific_impl_function %impl.elem0.loc12_7.2, @Op.96(constants.%Destroy.facet.13d) [symbolic = %specific_impl_fn.loc12_7.3 (constants.%specific_impl_fn.f4e)]
// CHECK:STDOUT:     %bound_method.loc12_7.4: <bound method> = bound_method %value.var, %specific_impl_fn.loc12_7.2
// CHECK:STDOUT:     %addr.loc12_7.2: @Next.loc11.%ptr.loc11_44.2 (%ptr.784) = addr_of %value.var
// CHECK:STDOUT:     %.loc12_7.2: init %empty_tuple.type = call %bound_method.loc12_7.4(%addr.loc12_7.2)
// CHECK:STDOUT:     return %None.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> %return.param: %IntRange.365 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:   %.loc27_22: %Make.type.cef = specific_constant @IntRange.%Make.decl, @IntRange(constants.%int_32) [concrete = constants.%Make.0dc]
// CHECK:STDOUT:   %Make.ref: %Make.type.cef = name_ref Make, %.loc27_22 [concrete = constants.%Make.0dc]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%int_32) [concrete = constants.%Make.specific_fn]
// CHECK:STDOUT:   %.loc26_20: ref %IntRange.365 = splice_block %return {}
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Make.call: init %IntRange.365 = call %Make.specific_fn(%.loc27_28.2, %end.ref) to %.loc26_20
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc26_20, constants.%Op.fa8
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.fa8, @Op.97(constants.%IntRange.365) [concrete = constants.%Op.specific_fn.df8]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %.loc26_20, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.049 = addr_of %.loc26_20
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc26(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.51f
// CHECK:STDOUT:   %Make => constants.%Make.2ec
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b4f426.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.e7c
// CHECK:STDOUT:   %struct_type.start.end.loc24_1.2 => constants.%struct_type.start.end.78d
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.9d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.2 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.8963eb.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.dcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CursorType(constants.%Self.a96) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CursorType(constants.%Iterate.facet.7f1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ElementType(constants.%Self.a96) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ElementType(constants.%Iterate.facet.7f1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %Int.loc9_54.2 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d96
// CHECK:STDOUT:   %Iterate.impl_witness => constants.%Iterate.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NewCursor.type => constants.%NewCursor.type.b02
// CHECK:STDOUT:   %NewCursor => constants.%NewCursor.f32
// CHECK:STDOUT:   %Next.type => constants.%Next.type.2c8
// CHECK:STDOUT:   %Next => constants.%Next.259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NewCursor.loc10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.dcd
// CHECK:STDOUT:   %Int.loc10_45.2 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %pattern_type.loc10_32 => constants.%pattern_type.8963eb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Next.loc11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.dcd
// CHECK:STDOUT:   %Int.loc11_43.2 => constants.%Int.49d0e6.1
// CHECK:STDOUT:   %ptr.loc11_44.2 => constants.%ptr.784
// CHECK:STDOUT:   %pattern_type.loc11_25 => constants.%pattern_type.4dc
// CHECK:STDOUT:   %Optional.loc11_75.2 => constants.%Optional.708
// CHECK:STDOUT:   %pattern_type.loc11_47 => constants.%pattern_type.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.cef
// CHECK:STDOUT:   %Make => constants.%Make.0dc
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.e33
// CHECK:STDOUT:   %struct_type.start.end.loc24_1.2 => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.c45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.2 => constants.%i32
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
// CHECK:STDOUT:   %Make.type.51f: type = fn_type @Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Make.2ec: %Make.type.51f = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.dcd: type = pattern_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %pattern_type.0ede7b.1: type = pattern_type %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %require_complete.ffde5f.1: <witness> = require_complete_type %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %IntRange.elem.ecb: type = unbound_element_type %IntRange.349, %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %require_complete.524: <witness> = require_complete_type %IntRange.349 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.365: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %Make.type.cef: type = fn_type @Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %Make.0dc: %Make.type.cef = struct_value () [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.9ba: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.6da: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0b2: type = fn_type @Convert.1, @ImplicitAs.impl.c81(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.6d7: %Convert.type.0b2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.256: type = facet_type <@OrderedWith, @OrderedWith(%Int.7ff11f.1)> [symbolic]
// CHECK:STDOUT:   %require_complete.10b: <witness> = require_complete_type %OrderedWith.type.256 [symbolic]
// CHECK:STDOUT:   %Less.type.016: type = fn_type @Less.1, @OrderedWith(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e53: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %ptr.58a: type = ptr_type %Int.7ff11f.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.6d0: type = pattern_type %ptr.58a [symbolic]
// CHECK:STDOUT:   %require_complete.656: <witness> = require_complete_type %ptr.58a [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e34: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e36, @ImplicitAs.impl.c81(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.ed5: type = fn_type @Convert.1, @ImplicitAs.impl.c81(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.16d: %Convert.type.ed5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2e7: %ImplicitAs.type.9ba = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e34) [concrete]
// CHECK:STDOUT:   %.d6a: type = fn_type_with_self_type %Convert.type.6da, %ImplicitAs.facet.2e7 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %y, %Convert.16d [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.16d, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked.58b: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.96 [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %.Self.36d: %Iterate.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.36d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a12: type = impl_witness_access %Iterate.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %Iterate.lookup_impl_witness, element1 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem0.a12 = %Int.7ff11f.1 and %impl.elem1 = %Int.7ff11f.1> [symbolic]
// CHECK:STDOUT:   %require_complete.6f5: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness imports.%Iterate.impl_witness_table, @Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %NewCursor.type: type = fn_type @NewCursor, @Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %NewCursor: %NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Next.type: type = fn_type @Next, @Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Next: %Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.e9f: type = class_type @Optional, @Optional(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.fd8: type = pattern_type %Optional.e9f [symbolic]
// CHECK:STDOUT:   %require_complete.5a1: <witness> = require_complete_type %Optional.e9f [symbolic]
// CHECK:STDOUT:   %assoc0.d1b: %OrderedWith.assoc_type.e53 = assoc_entity element0, imports.%Main.import_ref.014 [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.350: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.0e1, @OrderedWith.impl.5b6(%N, %N) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.256 = facet_value %Int.7ff11f.1, (%OrderedWith.impl_witness.350) [symbolic]
// CHECK:STDOUT:   %.3bd: type = fn_type_with_self_type %Less.type.016, %OrderedWith.facet [symbolic]
// CHECK:STDOUT:   %Less.type.4f5: type = fn_type @Less.2, @OrderedWith.impl.5b6(%N, %N) [symbolic]
// CHECK:STDOUT:   %Less.3b5: %Less.type.4f5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Less.specific_fn.5e2: <specific function> = specific_function %Less.3b5, @Less.2(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.7ff11f.1, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.7ff11f.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Op.type.e3a: type = fn_type @Op.97 [concrete]
// CHECK:STDOUT:   %.15d: type = fn_type_with_self_type %Op.type.e3a, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.181: %.15d = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.be7: <specific function> = specific_impl_function %impl.elem0.181, @Op.97(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Some.type.fe1: type = fn_type @Some, @Optional(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %Some.aae: %Some.type.fe1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Some.specific_fn: <specific function> = specific_function %Some.aae, @Some(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f0d: <witness> = lookup_impl_witness %Optional.e9f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.82c: %Destroy.type = facet_value %Optional.e9f, (%Destroy.lookup_impl_witness.f0d) [symbolic]
// CHECK:STDOUT:   %.485: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.82c [symbolic]
// CHECK:STDOUT:   %impl.elem0.c7c: %.485 = impl_witness_access %Destroy.lookup_impl_witness.f0d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ebe: <specific function> = specific_impl_function %impl.elem0.c7c, @Op.96(%Destroy.facet.82c) [symbolic]
// CHECK:STDOUT:   %ptr.d1d: type = ptr_type %Optional.e9f [symbolic]
// CHECK:STDOUT:   %require_complete.c68: <witness> = require_complete_type %ptr.d1d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f03: <witness> = lookup_impl_witness %Int.7ff11f.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.96a: %Destroy.type = facet_value %Int.7ff11f.1, (%Destroy.lookup_impl_witness.f03) [symbolic]
// CHECK:STDOUT:   %.f67: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.96a [symbolic]
// CHECK:STDOUT:   %impl.elem0.4dc: %.f67 = impl_witness_access %Destroy.lookup_impl_witness.f03, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f8f: <specific function> = specific_impl_function %impl.elem0.4dc, @Op.96(%Destroy.facet.96a) [symbolic]
// CHECK:STDOUT:   %None.type.8c4: type = fn_type @None, @Optional(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %None.7a7: %None.type.8c4 = struct_value () [symbolic]
// CHECK:STDOUT:   %None.specific_fn: <specific function> = specific_function %None.7a7, @None(%Int.7ff11f.1) [symbolic]
// CHECK:STDOUT:   %Op.type.2d1: type = fn_type @Op.1, @Destroy.impl(%IntRange.365) [concrete]
// CHECK:STDOUT:   %Op.fa8: %Op.type.2d1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.049: type = ptr_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.df8: <specific function> = specific_function %Op.fa8, @Op.1(%IntRange.365) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.d13 = import_ref Main//lib, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d98 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.e58 = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.261 = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.f1e294.2: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a86: @ImplicitAs.impl.c81.%Convert.type (%Convert.type.0b2) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @ImplicitAs.impl.c81.%Convert (constants.%Convert.6d7)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e36 = impl_witness_table (%Core.import_ref.a86), @ImplicitAs.impl.c81 [concrete]
// CHECK:STDOUT:   %Core.import_ref.e33 = import_ref Core//prelude/types/int, loc51_46, unloaded
// CHECK:STDOUT:   %Core.import_ref.a58 = import_ref Core//prelude/types/int, loc52_58, unloaded
// CHECK:STDOUT:   %Core.import_ref.a39 = import_ref Core//prelude/types/int, loc53_49, unloaded
// CHECK:STDOUT:   %Core.import_ref.c4c = import_ref Core//prelude/types/int, loc54_61, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.0e1 = impl_witness_table (%Core.import_ref.e33, %Core.import_ref.a58, %Core.import_ref.a39, %Core.import_ref.c4c), @OrderedWith.impl.5b6 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.0c8 = import_ref Main//lib, loc9_87, unloaded
// CHECK:STDOUT:   %Main.import_ref.f1e294.3: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.dae: type = import_ref Main//lib, loc9_8, loaded [symbolic = @Iterate.impl.%IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Main.import_ref.35b: type = import_ref Main//lib, loc9_24, loaded [symbolic = @Iterate.impl.%Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %Main.import_ref.e3faa9.1 = import_ref Main//lib, loc9_87, unloaded
// CHECK:STDOUT:   %Main.import_ref.e3faa9.2 = import_ref Main//lib, loc9_87, unloaded
// CHECK:STDOUT:   %Main.import_ref.11c = import_ref Main//lib, loc10_47, unloaded
// CHECK:STDOUT:   %Main.import_ref.f97 = import_ref Main//lib, loc11_77, unloaded
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.e3faa9.1, %Main.import_ref.e3faa9.2, %Main.import_ref.11c, %Main.import_ref.f97), @Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f1e294.4: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.f1e294.5: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.014 = import_ref Main//lib, inst519 [indirect], unloaded
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
// CHECK:STDOUT: generic impl @Iterate.impl(imports.%Main.import_ref.f1e294.3: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.7ff11f.1)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem0.a12 = %Int and constants.%impl.elem1 = %Int> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.6f5)]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness imports.%Iterate.impl_witness_table, @Iterate.impl(%N) [symbolic = %Iterate.impl_witness (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NewCursor.type: type = fn_type @NewCursor, @Iterate.impl(%N) [symbolic = %NewCursor.type (constants.%NewCursor.type)]
// CHECK:STDOUT:   %NewCursor: @Iterate.impl.%NewCursor.type (%NewCursor.type) = struct_value () [symbolic = %NewCursor (constants.%NewCursor)]
// CHECK:STDOUT:   %Next.type: type = fn_type @Next, @Iterate.impl(%N) [symbolic = %Next.type (constants.%Next.type)]
// CHECK:STDOUT:   %Next: @Iterate.impl.%Next.type (%Next.type) = struct_value () [symbolic = %Next (constants.%Next)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.dae as imports.%Main.import_ref.35b {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.0c8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.f1e294.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @IntRange(%N) [symbolic = %Make.type (constants.%Make.type.51f)]
// CHECK:STDOUT:   %Make: @IntRange.%Make.type (%Make.type.51f) = struct_value () [symbolic = %Make (constants.%Make.2ec)]
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
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core.ece [concrete = imports.%Core.ece]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %int_literal.make_type, %.loc5_27.2 [concrete = Core.IntLiteral]
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
// CHECK:STDOUT:   %impl.elem0: %.d6a = impl_witness_access constants.%ImplicitAs.impl_witness.e34, element0 [concrete = constants.%Convert.16d]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc6_31.2(%y.ref) [symbolic = constants.%int.convert_checked.58b]
// CHECK:STDOUT:   %.loc6_31.1: %i32 = value_of_initializer %int.convert_checked [symbolic = constants.%int.convert_checked.58b]
// CHECK:STDOUT:   %.loc6_31.2: %i32 = converted %y.ref, %.loc6_31.1 [symbolic = constants.%int.convert_checked.58b]
// CHECK:STDOUT:   %Range.call: init %IntRange.365 = call %Range.ref(%.loc6_31.2) to %.loc6_3
// CHECK:STDOUT:   assign %x.var, %Range.call
// CHECK:STDOUT:   %.loc6_21: type = splice_block %IntRange [concrete = constants.%IntRange.365] {
// CHECK:STDOUT:     %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %IntRange.365 = bind_name x, %x.var
// CHECK:STDOUT:   %Op.bound.loc6_3.1: <bound method> = bound_method %.loc6_3, constants.%Op.fa8
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.fa8, @Op.1(constants.%IntRange.365) [concrete = constants.%Op.specific_fn.df8]
// CHECK:STDOUT:   %bound_method.loc6_3.1: <bound method> = bound_method %.loc6_3, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc6_3.1: %ptr.049 = addr_of %.loc6_3
// CHECK:STDOUT:   %no_op.loc6_3.1: init %empty_tuple.type = call %bound_method.loc6_3.1(%addr.loc6_3.1)
// CHECK:STDOUT:   %Op.bound.loc6_3.2: <bound method> = bound_method %x.var, constants.%Op.fa8
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.fa8, @Op.1(constants.%IntRange.365) [concrete = constants.%Op.specific_fn.df8]
// CHECK:STDOUT:   %bound_method.loc6_3.2: <bound method> = bound_method %x.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc6_3.2: %ptr.049 = addr_of %x.var
// CHECK:STDOUT:   %no_op.loc6_3.2: init %empty_tuple.type = call %bound_method.loc6_3.2(%addr.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(imports.%Main.import_ref.f1e294.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.7ff11f.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.0ede7b.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Int [symbolic = %require_complete.2 (constants.%require_complete.ffde5f.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @Make.%Int (%Int.7ff11f.1), .end: @Make.%Int (%Int.7ff11f.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.434)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NewCursor(imports.%Main.import_ref.f1e294.4: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %IntRange [symbolic = %pattern_type.1 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.7ff11f.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %Int [symbolic = %pattern_type.2 (constants.%pattern_type.0ede7b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.524)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.ecb)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Int [symbolic = %require_complete.2 (constants.%require_complete.ffde5f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Next(imports.%Main.import_ref.f1e294.5: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.349)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %IntRange [symbolic = %pattern_type.1 (constants.%pattern_type.dcd)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.7ff11f.1)]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %Int [symbolic = %ptr.1 (constants.%ptr.58a)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %ptr.1 [symbolic = %pattern_type.2 (constants.%pattern_type.6d0)]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(%Int) [symbolic = %Optional (constants.%Optional.e9f)]
// CHECK:STDOUT:   %pattern_type.3: type = pattern_type %Optional [symbolic = %pattern_type.3 (constants.%pattern_type.fd8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %Optional [symbolic = %require_complete.1 (constants.%require_complete.5a1)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %IntRange [symbolic = %require_complete.2 (constants.%require_complete.524)]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %ptr.1 [symbolic = %require_complete.3 (constants.%require_complete.656)]
// CHECK:STDOUT:   %require_complete.4: <witness> = require_complete_type %Int [symbolic = %require_complete.4 (constants.%require_complete.ffde5f.1)]
// CHECK:STDOUT:   %pattern_type.4: type = pattern_type %Int [symbolic = %pattern_type.4 (constants.%pattern_type.0ede7b.1)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.ecb)]
// CHECK:STDOUT:   %OrderedWith.type: type = facet_type <@OrderedWith, @OrderedWith(%Int)> [symbolic = %OrderedWith.type (constants.%OrderedWith.type.256)]
// CHECK:STDOUT:   %require_complete.5: <witness> = require_complete_type %OrderedWith.type [symbolic = %require_complete.5 (constants.%require_complete.10b)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.e53)]
// CHECK:STDOUT:   %assoc0: @Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.e53) = assoc_entity element0, imports.%Main.import_ref.014 [symbolic = %assoc0 (constants.%assoc0.d1b)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.0e1, @OrderedWith.impl.5b6(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.350)]
// CHECK:STDOUT:   %Less.type.1: type = fn_type @Less.1, @OrderedWith(%Int) [symbolic = %Less.type.1 (constants.%Less.type.016)]
// CHECK:STDOUT:   %OrderedWith.facet: @Next.%OrderedWith.type (%OrderedWith.type.256) = facet_value %Int, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet)]
// CHECK:STDOUT:   %.1: type = fn_type_with_self_type %Less.type.1, %OrderedWith.facet [symbolic = %.1 (constants.%.3bd)]
// CHECK:STDOUT:   %Less.type.2: type = fn_type @Less.2, @OrderedWith.impl.5b6(%N, %N) [symbolic = %Less.type.2 (constants.%Less.type.4f5)]
// CHECK:STDOUT:   %Less: @Next.%Less.type.2 (%Less.type.4f5) = struct_value () [symbolic = %Less (constants.%Less.3b5)]
// CHECK:STDOUT:   %Less.specific_fn: <specific function> = specific_function %Less, @Less.2(%N, %N) [symbolic = %Less.specific_fn (constants.%Less.specific_fn.5e2)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %.2: type = fn_type_with_self_type constants.%Op.type.e3a, %Inc.facet [symbolic = %.2 (constants.%.15d)]
// CHECK:STDOUT:   %impl.elem0.1: @Next.%.2 (%.15d) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.1 (constants.%impl.elem0.181)]
// CHECK:STDOUT:   %specific_impl_fn.1: <specific function> = specific_impl_function %impl.elem0.1, @Op.97(%Inc.facet) [symbolic = %specific_impl_fn.1 (constants.%specific_impl_fn.be7)]
// CHECK:STDOUT:   %Some.type: type = fn_type @Some, @Optional(%Int) [symbolic = %Some.type (constants.%Some.type.fe1)]
// CHECK:STDOUT:   %Some: @Next.%Some.type (%Some.type.fe1) = struct_value () [symbolic = %Some (constants.%Some.aae)]
// CHECK:STDOUT:   %Some.specific_fn: <specific function> = specific_function %Some, @Some(%Int) [symbolic = %Some.specific_fn (constants.%Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.1: <witness> = lookup_impl_witness %Optional, @Destroy [symbolic = %Destroy.lookup_impl_witness.1 (constants.%Destroy.lookup_impl_witness.f0d)]
// CHECK:STDOUT:   %Destroy.facet.1: %Destroy.type = facet_value %Optional, (%Destroy.lookup_impl_witness.1) [symbolic = %Destroy.facet.1 (constants.%Destroy.facet.82c)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.1 [symbolic = %.3 (constants.%.485)]
// CHECK:STDOUT:   %impl.elem0.2: @Next.%.3 (%.485) = impl_witness_access %Destroy.lookup_impl_witness.1, element0 [symbolic = %impl.elem0.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:   %specific_impl_fn.2: <specific function> = specific_impl_function %impl.elem0.2, @Op.96(%Destroy.facet.1) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn.ebe)]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %Optional [symbolic = %ptr.2 (constants.%ptr.d1d)]
// CHECK:STDOUT:   %require_complete.6: <witness> = require_complete_type %ptr.2 [symbolic = %require_complete.6 (constants.%require_complete.c68)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.2: <witness> = lookup_impl_witness %Int, @Destroy [symbolic = %Destroy.lookup_impl_witness.2 (constants.%Destroy.lookup_impl_witness.f03)]
// CHECK:STDOUT:   %Destroy.facet.2: %Destroy.type = facet_value %Int, (%Destroy.lookup_impl_witness.2) [symbolic = %Destroy.facet.2 (constants.%Destroy.facet.96a)]
// CHECK:STDOUT:   %.4: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.2 [symbolic = %.4 (constants.%.f67)]
// CHECK:STDOUT:   %impl.elem0.3: @Next.%.4 (%.f67) = impl_witness_access %Destroy.lookup_impl_witness.2, element0 [symbolic = %impl.elem0.3 (constants.%impl.elem0.4dc)]
// CHECK:STDOUT:   %specific_impl_fn.3: <specific function> = specific_impl_function %impl.elem0.3, @Op.96(%Destroy.facet.2) [symbolic = %specific_impl_fn.3 (constants.%specific_impl_fn.f8f)]
// CHECK:STDOUT:   %None.type: type = fn_type @None, @Optional(%Int) [symbolic = %None.type (constants.%None.type.8c4)]
// CHECK:STDOUT:   %None: @Next.%None.type (%None.type.8c4) = struct_value () [symbolic = %None (constants.%None.7a7)]
// CHECK:STDOUT:   %None.specific_fn: <specific function> = specific_function %None, @None(%Int) [symbolic = %None.specific_fn (constants.%None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.51f
// CHECK:STDOUT:   %Make => constants.%Make.2ec
// CHECK:STDOUT:   %Int => constants.%Int.7ff11f.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ffde5f.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.ecb
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.434
// CHECK:STDOUT:   %complete_type => constants.%complete_type.c76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%N) {
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
// CHECK:STDOUT:   %Make.type => constants.%Make.type.cef
// CHECK:STDOUT:   %Make => constants.%Make.0dc
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.f21
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.0fe
// CHECK:STDOUT:   %complete_type => constants.%complete_type.a43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %Int => constants.%Int.7ff11f.1
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6f5
// CHECK:STDOUT:   %Iterate.impl_witness => constants.%Iterate.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NewCursor.type => constants.%NewCursor.type
// CHECK:STDOUT:   %NewCursor => constants.%NewCursor
// CHECK:STDOUT:   %Next.type => constants.%Next.type
// CHECK:STDOUT:   %Next => constants.%Next
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NewCursor(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.dcd
// CHECK:STDOUT:   %Int => constants.%Int.7ff11f.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.0ede7b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.349
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.dcd
// CHECK:STDOUT:   %Int => constants.%Int.7ff11f.1
// CHECK:STDOUT:   %ptr.1 => constants.%ptr.58a
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.6d0
// CHECK:STDOUT:   %Optional => constants.%Optional.e9f
// CHECK:STDOUT:   %pattern_type.3 => constants.%pattern_type.fd8
// CHECK:STDOUT: }
// CHECK:STDOUT:

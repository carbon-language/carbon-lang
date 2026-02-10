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
  var x: IntRange(32) = Range(y);
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.265: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.fc6021.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.764eab.1: type = pattern_type %Int.fc6021.1 [symbolic]
// CHECK:STDOUT:   %.5f2: form = init_form %IntRange.265, call_param2 [symbolic]
// CHECK:STDOUT:   %pattern_type.b16: type = pattern_type %IntRange.265 [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.1df: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.8a9: %IntRange.Make.type.1df = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %.Self.2b2: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.2b2 [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.5ce: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.c9c [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.53f: <witness> = lookup_impl_witness %.Self.2b2, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.49e: type = impl_witness_access %Iterate.lookup_impl_witness.53f, element1 [symbolic_self]
// CHECK:STDOUT:   %assoc0.16e: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.aa7 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %impl.elem0.3b1: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.53f, element0 [symbolic_self]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc9 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %custom_witness.8095d9.1: <witness> = custom_witness (%DestroyOp.b0ebf8.1), @Destroy [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %require_complete.9019d7.1: <witness> = require_complete_type %Int.fc6021.1 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7b282e.1: form = init_form %Int.fc6021.1, call_param1 [symbolic]
// CHECK:STDOUT:   %.4f8: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.7a8: <witness> = lookup_impl_witness %Int.fc6021.1, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Int.fc6021.1, (%custom_witness.8095d9.1, %Copy.lookup_impl_witness.7a8) [symbolic]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where %impl.elem1.49e = %Int.fc6021.1 and %impl.elem0.3b1 = %facet_value> [symbolic]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.eaa: type = fn_type @Optional.Some, @Optional(%T.542) [symbolic]
// CHECK:STDOUT:   %Optional.Some.6ca: %Optional.Some.type.eaa = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.None.type.fc5: type = fn_type @Optional.None, @Optional(%T.542) [symbolic]
// CHECK:STDOUT:   %Optional.None.fcb: %Optional.None.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.as.Iterate.impl.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.297: <witness> = require_complete_type %Iterate_where.type [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: %IntRange.as.Iterate.impl.NewCursor.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.c9c: type = ptr_type %Int.fc6021.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bda: type = pattern_type %ptr.c9c [symbolic]
// CHECK:STDOUT:   %.f57: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.b62: <witness> = lookup_impl_witness %Int.fc6021.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.01e: %OptionalStorage.type = facet_value %Int.fc6021.1, (%OptionalStorage.lookup_impl_witness.b62) [symbolic]
// CHECK:STDOUT:   %Optional.e48: type = class_type @Optional, @Optional(%OptionalStorage.facet.01e) [symbolic]
// CHECK:STDOUT:   %.6a7: form = init_form %Optional.e48, call_param2 [symbolic]
// CHECK:STDOUT:   %pattern_type.0c2: type = pattern_type %Optional.e48 [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: %IntRange.as.Iterate.impl.Next.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IntRange.elem.541: type = unbound_element_type %IntRange.265, %Int.fc6021.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.ff1: type = struct_type {.start: %Int.fc6021.1, .end: %Int.fc6021.1} [symbolic]
// CHECK:STDOUT:   %complete_type.427: <witness> = complete_type_witness %struct_type.start.end.ff1 [symbolic]
// CHECK:STDOUT:   %require_complete.8a1: <witness> = require_complete_type %IntRange.265 [symbolic]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %Int.fc6021.1, (%Copy.lookup_impl_witness.7a8) [symbolic]
// CHECK:STDOUT:   %.3676: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.3b9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.053: %.3676 = impl_witness_access %Copy.lookup_impl_witness.7a8, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.62b: <specific function> = specific_impl_function %impl.elem0.053, @Copy.Op(%Copy.facet.3b9) [symbolic]
// CHECK:STDOUT:   %require_complete.45c: <witness> = require_complete_type %ptr.c9c [symbolic]
// CHECK:STDOUT:   %Optional.None.type.d56: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.01e) [symbolic]
// CHECK:STDOUT:   %Optional.None.b26: %Optional.None.type.d56 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.f74: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.01e) [symbolic]
// CHECK:STDOUT:   %Optional.Some.076: %Optional.Some.type.f74 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.0bc: <witness> = require_complete_type %Optional.e48 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2b3: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.ca0: type = fn_type @OrderedWith.Less, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.c6e: %OrderedWith.Less.type.ca0 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.e44: type = facet_type <@OrderedWith, @OrderedWith(%Int.fc6021.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.566: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.fc6021.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.215: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.fc6021.1) [symbolic]
// CHECK:STDOUT:   %assoc0.aa8: %OrderedWith.assoc_type.215 = assoc_entity element0, imports.%Core.import_ref.aa1 [symbolic]
// CHECK:STDOUT:   %require_complete.dd3: <witness> = require_complete_type %OrderedWith.type.e44 [symbolic]
// CHECK:STDOUT:   %assoc0.3e3: %OrderedWith.assoc_type.2b3 = assoc_entity element0, imports.%Core.import_ref.780a28.1 [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.5f1: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.2e6(%N, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.a9b: %Int.as.OrderedWith.impl.Less.type.5f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.ea9: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.b93, @Int.as.OrderedWith.impl.2e6(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.3f1: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.2e6(%N, %N) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.4cf: %Int.as.OrderedWith.impl.Less.type.3f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4b5: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.2e6(%N, %N) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.e44 = facet_value %Int.fc6021.1, (%OrderedWith.impl_witness.ea9) [symbolic]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %OrderedWith.Less.type.566, %OrderedWith.facet [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.4c2: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.4cf, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.Op.type: type = fn_type @Inc.Op [concrete]
// CHECK:STDOUT:   %.53b: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.fc6021.1, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.fc6021.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.813: type = fn_type_with_self_type %Inc.Op.type, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1f5: %.813 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b9e: <specific function> = specific_impl_function %impl.elem0.1f5, @Inc.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.076, @Optional.Some(%OptionalStorage.facet.01e) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.3e3: %Destroy.type = facet_value %Int.fc6021.1, (%custom_witness.8095d9.1) [symbolic]
// CHECK:STDOUT:   %.623: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.3e3 [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.b26, @Optional.None(%OptionalStorage.facet.01e) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntRange.a89: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %.273: form = init_form %IntRange.a89, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.615: type = pattern_type %IntRange.a89 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.045: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.3e9: %IntRange.Make.type.045 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %.14e: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.elem.a58: type = unbound_element_type %IntRange.a89, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.d0a: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c45: <witness> = complete_type_witness %struct_type.start.end.d0a [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.e73: form = init_form %IntRange.a89, call_param2 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.3e9, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.facet.de4: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %.f79: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.de4 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d2a: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.16e]
// CHECK:STDOUT:   %Core.import_ref.66d: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1.5ce]
// CHECK:STDOUT:   %Core.import_ref.c9c: type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %CursorType: type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.aa7: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %facet_type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d20a: @Optional.%Optional.None.type (%Optional.None.type.fc5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.fcb)]
// CHECK:STDOUT:   %Core.import_ref.aeb: @Optional.%Optional.Some.type (%Optional.Some.type.eaa) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.6ca)]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.eeb: @OrderedWith.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b3) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.%assoc0 (constants.%assoc0.3e3)]
// CHECK:STDOUT:   %Core.import_ref.aa1: @OrderedWith.%OrderedWith.Less.type (%OrderedWith.Less.type.ca0) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.%OrderedWith.Less (constants.%OrderedWith.Less.c6e)]
// CHECK:STDOUT:   %Core.import_ref.780a28.1 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.12b: @Int.as.OrderedWith.impl.2e6.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.5f1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.2e6.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.a9b)]
// CHECK:STDOUT:   %Core.import_ref.d3d = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ef = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.806 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.b93 = impl_witness_table (%Core.import_ref.12b, %Core.import_ref.d3d, %Core.import_ref.5ef, %Core.import_ref.806), @Int.as.OrderedWith.impl.2e6 [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.615 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.615 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc26: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc26_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc26: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.a89]
// CHECK:STDOUT:     %.loc26_34.2: form = init_form %IntRange.loc26, call_param1 [concrete = constants.%.273]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc26_15: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc26_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %end: %i32 = value_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.a89 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.a89 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc9_54.1 [symbolic = %pattern_type (constants.%pattern_type.764eab.1)]
// CHECK:STDOUT:   %.loc9_85.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc9_85.1 (constants.%.4f8)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.7a8)]
// CHECK:STDOUT:   %facet_value.loc9_85.1: %facet_type = facet_value %Int.loc9_54.1, (constants.%custom_witness.8095d9.1, %Copy.lookup_impl_witness) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.49e = %Int.loc9_54.1 and constants.%impl.elem0.3b1 = %facet_value.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type)]
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.297)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type (constants.%IntRange.as.Iterate.impl.NewCursor.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)]
// CHECK:STDOUT:   %.loc11: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%facet_value.loc9_85.1) [symbolic = %.loc11 (constants.%.f57)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type (constants.%IntRange.as.Iterate.impl.Next.type)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %.loc9_24 {
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type) = fn_decl @IntRange.as.Iterate.impl.NewCursor [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.b16) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_18 (%pattern_type.b16) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_32 (%pattern_type.764eab.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.NewCursor.%pattern_type.loc10_32 (%pattern_type.764eab.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc10_45.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc10_45.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:       %.loc10_45.2: form = init_form %Int.loc10_45.2, call_param1 [symbolic = %.loc10_45.1 (constants.%.7b282e.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.265) = value_param call_param0
// CHECK:STDOUT:       %.loc10_24.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.265)] {
// CHECK:STDOUT:         %.loc10_24.2: type = specific_constant constants.%IntRange.265, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_24.2 [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.265) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.fc6021.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.fc6021.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type) = fn_decl @IntRange.as.Iterate.impl.Next [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.b16) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_13 (%pattern_type.b16) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %cursor.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.bda) = value_binding_pattern cursor [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_25 (%pattern_type.bda) = value_param_pattern %cursor.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_47 (%pattern_type.0c2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc11_47 (%pattern_type.0c2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc11_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc11: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc11_64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc11_68: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc11_73: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc11_74: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_75.2: %OptionalStorage.type = facet_value %Int.loc11_74, (constants.%OptionalStorage.lookup_impl_witness.b62) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:       %.loc11_75.5: %OptionalStorage.type = converted %Int.loc11_74, %OptionalStorage.facet.loc11_75.2 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:       %Optional.loc11_75.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.loc11_75.1 (constants.%Optional.e48)]
// CHECK:STDOUT:       %.loc11_75.6: form = init_form %Optional.loc11_75.2, call_param2 [symbolic = %.loc11_75.4 (constants.%.6a7)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.265) = value_param call_param0
// CHECK:STDOUT:       %.loc11_19.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.265)] {
// CHECK:STDOUT:         %.loc11_19.2: type = specific_constant constants.%IntRange.265, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_19.2 [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.265) = value_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.c9c) = value_param call_param1
// CHECK:STDOUT:       %.loc11_44: type = splice_block %ptr.loc11_44.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.c9c)] {
// CHECK:STDOUT:         %Core.ref.loc11_33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc11_37: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc11_42: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc11_43.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:         %ptr.loc11_44.2: type = ptr_type %Int.loc11_43.2 [symbolic = %ptr.loc11_44.1 (constants.%ptr.c9c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.c9c) = value_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, %IntRange.as.Iterate.impl.NewCursor.decl, %IntRange.as.Iterate.impl.Next.decl), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness.loc9_87.1: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N) [symbolic = %Iterate.impl_witness.loc9_87.2 (constants.%Iterate.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: type = impl_witness_assoc_constant constants.%Int.fc6021.1 [symbolic = %Int.loc9_54.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %facet_type = impl_witness_assoc_constant constants.%facet_value [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .NewCursor = %IntRange.as.Iterate.impl.NewCursor.decl
// CHECK:STDOUT:     .Next = %IntRange.as.Iterate.impl.Next.decl
// CHECK:STDOUT:     witness = %Iterate.impl_witness.loc9_87.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_16.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_16.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.1df)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.1df) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.8a9)]
// CHECK:STDOUT:   %.loc9: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N.loc4_16.1) [symbolic = %.loc9 (constants.%.4f8)]
// CHECK:STDOUT:   %Int.loc22_32.2: type = class_type @Int, @Int(%N.loc4_16.1) [symbolic = %Int.loc22_32.2 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc22_32.2 [symbolic = %require_complete (constants.%require_complete.9019d7.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_16.1) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc22_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.541)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int.loc22_32.2 (%Int.fc6021.1), .end: @IntRange.%Int.loc22_32.2 (%Int.fc6021.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.ff1)]
// CHECK:STDOUT:   %complete_type.loc24_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.427)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.1df) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.8a9)] {
// CHECK:STDOUT:       %start.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.764eab.1) = value_binding_pattern start [concrete]
// CHECK:STDOUT:       %start.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.764eab.1) = value_param_pattern %start.patt, call_param0 [concrete]
// CHECK:STDOUT:       %end.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.764eab.1) = value_binding_pattern end [concrete]
// CHECK:STDOUT:       %end.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.764eab.1) = value_param_pattern %end.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.b16) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.b16) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52.2: type = specific_constant constants.%IntRange.265, @IntRange(constants.%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52.2 [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:       %.loc5_52.3: form = init_form %Self.ref, call_param2 [symbolic = %.loc5_52.1 (constants.%.5f2)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.fc6021.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = value_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.fc6021.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc5_28.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = value_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.265) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.265) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.265 [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2b2]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2b2]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.66d [concrete = constants.%assoc1.5ce]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%Iterate.lookup_impl_witness.53f, element1 [symbolic_self = constants.%impl.elem1.49e]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2b2]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.d2a [concrete = constants.%assoc0.16e]
// CHECK:STDOUT:       %impl.elem0: %facet_type = impl_witness_access constants.%Iterate.lookup_impl_witness.53f, element0 [symbolic_self = constants.%impl.elem0.3b1]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_54.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:       %facet_value.loc9_85.2: %facet_type = facet_value %Int.loc9_85, (constants.%custom_witness.8095d9.1, constants.%Copy.lookup_impl_witness.7a8) [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_85.2: %facet_type = converted %Int.loc9_85, %facet_value.loc9_85.2 [symbolic = %facet_value.loc9_85.1 (constants.%facet_value)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr %.Self [symbolic = %Iterate_where.type (constants.%Iterate_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%Iterate.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %Int.loc9_54.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_85.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc22: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:     %Int.loc22_32.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:     %.loc22: @IntRange.%IntRange.elem (%IntRange.elem.541) = field_decl start, element0 [concrete]
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc23: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc23: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N)]
// CHECK:STDOUT:     %Int.loc23: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc22_32.2 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:     %.loc23: @IntRange.%IntRange.elem (%IntRange.elem.541) = field_decl end, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc24_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.ff1 [symbolic = %complete_type.loc24_1.2 (constants.%complete_type.427)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc24_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.265
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %.loc22
// CHECK:STDOUT:     .end [private] = %.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:   %pattern_type.loc5_11: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_11 (constants.%pattern_type.764eab.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %.loc5_52.1: form = init_form %IntRange, call_param2 [symbolic = %.loc5_52.1 (constants.%.5f2)]
// CHECK:STDOUT:   %pattern_type.loc5_49: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_49 (constants.%pattern_type.b16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.9019d7.1)]
// CHECK:STDOUT:   %require_complete.loc5_52: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_52 (constants.%require_complete.8a1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.ff1)]
// CHECK:STDOUT:   %.loc6_22.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc6_22.1 (constants.%.4f8)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.7a8)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.3b9)]
// CHECK:STDOUT:   %.loc6_22.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc6_22.2 (constants.%.3676)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.2 (%.3676) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1)) -> out %return.param: @IntRange.Make.%IntRange (%IntRange.265) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.ff1) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.2 (%.3676) = impl_witness_access constants.%Copy.lookup_impl_witness.7a8, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.Op(constants.%Copy.facet.3b9) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %Copy.Op.call.loc6_22: init @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = class_element_access %return.param, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) to %.loc6_39.2 = in_place_init %Copy.Op.call.loc6_22
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.2 (%.3676) = impl_witness_access constants.%Copy.lookup_impl_witness.7a8, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.Op(constants.%Copy.facet.3b9) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %Copy.Op.call.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) = class_element_access %return.param, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.fc6021.1) to %.loc6_39.4 = in_place_init %Copy.Op.call.loc6_36
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.265) to %return.param = class_init (%.loc6_39.3, %.loc6_39.5)
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.265) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc9(%self.param: @IntRange.as.Iterate.impl.%Int.loc9_54.1 (%Int.fc6021.1)) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %pattern_type.loc10_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc10_18 (constants.%pattern_type.b16)]
// CHECK:STDOUT:   %Int.loc10_45.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc10_45.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:   %.loc10_45.1: form = init_form %Int.loc10_45.1, call_param1 [symbolic = %.loc10_45.1 (constants.%.7b282e.1)]
// CHECK:STDOUT:   %pattern_type.loc10_32: type = pattern_type %Int.loc10_45.1 [symbolic = %pattern_type.loc10_32 (constants.%pattern_type.764eab.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_22: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc10_22 (constants.%require_complete.8a1)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc10_45.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.541)]
// CHECK:STDOUT:   %require_complete.loc10_60: <witness> = require_complete_type %Int.loc10_45.1 [symbolic = %require_complete.loc10_60 (constants.%require_complete.9019d7.1)]
// CHECK:STDOUT:   %.loc10_60.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc10_60.3 (constants.%.4f8)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc10_45.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.7a8)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc10_45.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.3b9)]
// CHECK:STDOUT:   %.loc10_60.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_60.4 (constants.%.3676)]
// CHECK:STDOUT:   %impl.elem0.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%.loc10_60.4 (%.3676) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_60.2: <specific function> = specific_impl_function %impl.elem0.loc10_60.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.265)) -> out %return.param: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.fc6021.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange (%IntRange.265) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.%IntRange.elem (%IntRange.elem.541) = name_ref start, @IntRange.%.loc22 [concrete = @IntRange.%.loc22]
// CHECK:STDOUT:     %.loc10_60.1: ref @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.fc6021.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_60.2: @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.fc6021.1) = acquire_value %.loc10_60.1
// CHECK:STDOUT:     %impl.elem0.loc10_60.1: @IntRange.as.Iterate.impl.NewCursor.%.loc10_60.4 (%.3676) = impl_witness_access constants.%Copy.lookup_impl_witness.7a8, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:     %bound_method.loc10_60.1: <bound method> = bound_method %.loc10_60.2, %impl.elem0.loc10_60.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_60.1: <specific function> = specific_impl_function %impl.elem0.loc10_60.1, @Copy.Op(constants.%Copy.facet.3b9) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:     %bound_method.loc10_60.2: <bound method> = bound_method %.loc10_60.2, %specific_impl_fn.loc10_60.1
// CHECK:STDOUT:     %Copy.Op.call: init @IntRange.as.Iterate.impl.NewCursor.%Int.loc10_45.1 (%Int.fc6021.1) = call %bound_method.loc10_60.2(%.loc10_60.2)
// CHECK:STDOUT:     return %Copy.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.b16)]
// CHECK:STDOUT:   %Int.loc11_43.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:   %ptr.loc11_44.1: type = ptr_type %Int.loc11_43.1 [symbolic = %ptr.loc11_44.1 (constants.%ptr.c9c)]
// CHECK:STDOUT:   %pattern_type.loc11_25: type = pattern_type %ptr.loc11_44.1 [symbolic = %pattern_type.loc11_25 (constants.%pattern_type.bda)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.7a8)]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Int.loc11_43.1, (constants.%custom_witness.8095d9.1, %Copy.lookup_impl_witness) [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %.loc11_75.3: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%facet_value) [symbolic = %.loc11_75.3 (constants.%.f57)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.b62)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_75.1: %OptionalStorage.type = facet_value %Int.loc11_43.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:   %Optional.loc11_75.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.loc11_75.1 (constants.%Optional.e48)]
// CHECK:STDOUT:   %.loc11_75.4: form = init_form %Optional.loc11_75.1, call_param2 [symbolic = %.loc11_75.4 (constants.%.6a7)]
// CHECK:STDOUT:   %pattern_type.loc11_47: type = pattern_type %Optional.loc11_75.1 [symbolic = %pattern_type.loc11_47 (constants.%pattern_type.0c2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_17: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc11_17 (constants.%require_complete.8a1)]
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %ptr.loc11_44.1 [symbolic = %require_complete.loc11_31 (constants.%require_complete.45c)]
// CHECK:STDOUT:   %require_complete.loc11_75: <witness> = require_complete_type %Optional.loc11_75.1 [symbolic = %require_complete.loc11_75 (constants.%require_complete.0bc)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %Int.loc11_43.1 [symbolic = %require_complete.loc12 (constants.%require_complete.9019d7.1)]
// CHECK:STDOUT:   %pattern_type.loc12: type = pattern_type %Int.loc11_43.1 [symbolic = %pattern_type.loc12 (constants.%pattern_type.764eab.1)]
// CHECK:STDOUT:   %.loc12_32.3: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc12_32.3 (constants.%.4f8)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc11_43.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.3b9)]
// CHECK:STDOUT:   %.loc12_32.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc12_32.4 (constants.%.3676)]
// CHECK:STDOUT:   %impl.elem0.loc12_32.2: @IntRange.as.Iterate.impl.Next.%.loc12_32.4 (%.3676) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_32.2: <specific function> = specific_impl_function %impl.elem0.loc12_32.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc11_43.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.541)]
// CHECK:STDOUT:   %OrderedWith.type.loc13_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc11_43.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.e44)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %OrderedWith.type.loc13_17.2 [symbolic = %require_complete.loc13 (constants.%require_complete.dd3)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.215)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.215) = assoc_entity element0, imports.%Core.import_ref.aa1 [symbolic = %assoc0 (constants.%assoc0.aa8)]
// CHECK:STDOUT:   %.loc13_17.2: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.2e6(%N, %N) [symbolic = %.loc13_17.2 (constants.%.4b5)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.b93, @Int.as.OrderedWith.impl.2e6(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.ea9)]
// CHECK:STDOUT:   %OrderedWith.Less.type: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.loc11_43.1) [symbolic = %OrderedWith.Less.type (constants.%OrderedWith.Less.type.566)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.%OrderedWith.type.loc13_17.2 (%OrderedWith.type.e44) = facet_value %Int.loc11_43.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet)]
// CHECK:STDOUT:   %.loc13_17.3: type = fn_type_with_self_type %OrderedWith.Less.type, %OrderedWith.facet [symbolic = %.loc13_17.3 (constants.%.2ac)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.2e6(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.3f1)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.3f1) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.4cf)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.4c2)]
// CHECK:STDOUT:   %.loc14_9.1: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic = %.loc14_9.1 (constants.%.53b)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc11_43.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.loc11_43.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %.loc14_9.2: type = fn_type_with_self_type constants.%Inc.Op.type, %Inc.facet [symbolic = %.loc14_9.2 (constants.%.813)]
// CHECK:STDOUT:   %impl.elem0.loc14_9.2: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.813) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.1f5)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_9.2: <specific function> = specific_impl_function %impl.elem0.loc14_9.2, @Inc.Op(%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.b9e)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.f74)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.f74) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.076)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc15_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc11_43.1, (constants.%custom_witness.8095d9.1) [symbolic = %Destroy.facet (constants.%Destroy.facet.3e3)]
// CHECK:STDOUT:   %.loc12_7: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc12_7 (constants.%.623)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.d56)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.d56) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.b26)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc17_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc11_75.1) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.265), %cursor.param: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.c9c)) -> out %return.param: @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.764eab.1) = ref_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.var_patt: @IntRange.as.Iterate.impl.Next.%pattern_type.loc12 (%pattern_type.764eab.1) = var_pattern %value.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = var %value.var_patt
// CHECK:STDOUT:     %cursor.ref.loc12: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.c9c) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc12_32.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = deref %cursor.ref.loc12
// CHECK:STDOUT:     %.loc12_32.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = acquire_value %.loc12_32.1
// CHECK:STDOUT:     %impl.elem0.loc12_32.1: @IntRange.as.Iterate.impl.Next.%.loc12_32.4 (%.3676) = impl_witness_access constants.%Copy.lookup_impl_witness.7a8, element0 [symbolic = %impl.elem0.loc12_32.2 (constants.%impl.elem0.053)]
// CHECK:STDOUT:     %bound_method.loc12_32.1: <bound method> = bound_method %.loc12_32.2, %impl.elem0.loc12_32.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_32.1: <specific function> = specific_impl_function %impl.elem0.loc12_32.1, @Copy.Op(constants.%Copy.facet.3b9) [symbolic = %specific_impl_fn.loc12_32.2 (constants.%specific_impl_fn.62b)]
// CHECK:STDOUT:     %bound_method.loc12_32.2: <bound method> = bound_method %.loc12_32.2, %specific_impl_fn.loc12_32.1
// CHECK:STDOUT:     %Copy.Op.call: init @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = call %bound_method.loc12_32.2(%.loc12_32.2)
// CHECK:STDOUT:     assign %value.var, %Copy.Op.call
// CHECK:STDOUT:     %.loc12_28: type = splice_block %Int.loc12 [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc12: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc12: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc12: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = ref_binding value, %value.var
// CHECK:STDOUT:     %value.ref.loc13: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.%IntRange (%IntRange.265) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.%IntRange.elem (%IntRange.elem.541) = name_ref end, @IntRange.%.loc23 [concrete = @IntRange.%.loc23]
// CHECK:STDOUT:     %.loc13_23.1: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc13_23.2: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = acquire_value %.loc13_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc13_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.fc6021.1)> [symbolic = %OrderedWith.type.loc13_17.2 (constants.%OrderedWith.type.e44)]
// CHECK:STDOUT:     %.loc13_17.1: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.215) = specific_constant imports.%Core.import_ref.eeb, @OrderedWith(constants.%Int.fc6021.1) [symbolic = %assoc0 (constants.%assoc0.aa8)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.%OrderedWith.assoc_type (%OrderedWith.assoc_type.215) = name_ref Less, %.loc13_17.1 [symbolic = %assoc0 (constants.%assoc0.aa8)]
// CHECK:STDOUT:     %impl.elem0.loc13: @IntRange.as.Iterate.impl.Next.%.loc13_17.3 (%.2ac) = impl_witness_access constants.%OrderedWith.impl_witness.ea9, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.4cf)]
// CHECK:STDOUT:     %bound_method.loc13_17.1: <bound method> = bound_method %value.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc13, @Int.as.OrderedWith.impl.Less.1(constants.%N, constants.%N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.4c2)]
// CHECK:STDOUT:     %bound_method.loc13_17.2: <bound method> = bound_method %value.ref.loc13, %specific_fn
// CHECK:STDOUT:     %.loc13_11: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = acquire_value %value.ref.loc13
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13_17.2(%.loc13_11, %.loc13_23.2)
// CHECK:STDOUT:     %.loc13_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc13_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc13_27.1
// CHECK:STDOUT:     if %.loc13_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc14: @IntRange.as.Iterate.impl.Next.%ptr.loc11_44.1 (%ptr.c9c) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc14_11: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = deref %cursor.ref.loc14
// CHECK:STDOUT:     %impl.elem0.loc14_9.1: @IntRange.as.Iterate.impl.Next.%.loc14_9.2 (%.813) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_9.2 (constants.%impl.elem0.1f5)]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %.loc14_11, %impl.elem0.loc14_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_9.1: <specific function> = specific_impl_function %impl.elem0.loc14_9.1, @Inc.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc14_9.2 (constants.%specific_impl_fn.b9e)]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %.loc14_11, %specific_impl_fn.loc14_9.1
// CHECK:STDOUT:     %Inc.Op.call: init %empty_tuple.type = call %bound_method.loc14_9.2(%.loc14_11)
// CHECK:STDOUT:     %Core.ref.loc15_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc15: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc15_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc15: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc15: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc15: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_41: %OptionalStorage.type = facet_value %Int.loc15, (constants.%OptionalStorage.lookup_impl_witness.b62) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %.loc15_41: %OptionalStorage.type = converted %Int.loc15, %OptionalStorage.facet.loc15_41 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %Optional.loc15: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.loc11_75.1 (constants.%Optional.e48)]
// CHECK:STDOUT:     %.loc15_42: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.f74) = specific_constant imports.%Core.import_ref.aeb, @Optional(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.Some (constants.%Optional.Some.076)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.%Optional.Some.type (%Optional.Some.type.f74) = name_ref Some, %.loc15_42 [symbolic = %Optional.Some (constants.%Optional.Some.076)]
// CHECK:STDOUT:     %value.ref.loc15: ref @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.1: %OptionalStorage.type = facet_value constants.%Int.fc6021.1, (constants.%OptionalStorage.lookup_impl_witness.b62) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %.loc15_53.1: %OptionalStorage.type = converted constants.%Int.fc6021.1, %OptionalStorage.facet.loc15_53.1 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc15_53.2: %OptionalStorage.type = facet_value constants.%Int.fc6021.1, (constants.%OptionalStorage.lookup_impl_witness.b62) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %.loc15_53.2: %OptionalStorage.type = converted constants.%Int.fc6021.1, %OptionalStorage.facet.loc15_53.2 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc15_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.Some.specific_fn.loc15_42.2 (constants.%Optional.Some.specific_fn)]
// CHECK:STDOUT:     %.loc11_75.1: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc15_48: @IntRange.as.Iterate.impl.Next.%Int.loc11_43.1 (%Int.fc6021.1) = acquire_value %value.ref.loc15
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) to %.loc11_75.1 = call %Optional.Some.specific_fn.loc15_42.1(%.loc15_48)
// CHECK:STDOUT:     %DestroyOp.bound.loc12_7.1: <bound method> = bound_method %value.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:     %DestroyOp.call.loc12_7.1: init %empty_tuple.type = call %DestroyOp.bound.loc12_7.1(%value.var)
// CHECK:STDOUT:     return %Optional.Some.call to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc17_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc17: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc17_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc17: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc17: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:     %Int.loc17: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc11_43.1 (constants.%Int.fc6021.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc17: %OptionalStorage.type = facet_value %Int.loc17, (constants.%OptionalStorage.lookup_impl_witness.b62) [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %.loc17_41: %OptionalStorage.type = converted %Int.loc17, %OptionalStorage.facet.loc17 [symbolic = %OptionalStorage.facet.loc11_75.1 (constants.%OptionalStorage.facet.01e)]
// CHECK:STDOUT:     %Optional.loc17: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.loc11_75.1 (constants.%Optional.e48)]
// CHECK:STDOUT:     %.loc17_42: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.d56) = specific_constant imports.%Core.import_ref.d20a, @Optional(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.None (constants.%Optional.None.b26)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.%Optional.None.type (%Optional.None.type.d56) = name_ref None, %.loc17_42 [symbolic = %Optional.None (constants.%Optional.None.b26)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc17_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.01e) [symbolic = %Optional.None.specific_fn.loc17_42.2 (constants.%Optional.None.specific_fn)]
// CHECK:STDOUT:     %.loc11_75.2: ref @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) = splice_block %return.param {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.%Optional.loc11_75.1 (%Optional.e48) to %.loc11_75.2 = call %Optional.None.specific_fn.loc17_42.1()
// CHECK:STDOUT:     %DestroyOp.bound.loc12_7.2: <bound method> = bound_method %value.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:     %DestroyOp.call.loc12_7.2: init %empty_tuple.type = call %DestroyOp.bound.loc12_7.2(%value.var)
// CHECK:STDOUT:     return %Optional.None.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> out %return.param: %IntRange.a89 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc27: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc27: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.a89]
// CHECK:STDOUT:   %.loc27_22: %IntRange.Make.type.045 = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.3e9]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.045 = name_ref Make, %.loc27_22 [concrete = constants.%IntRange.Make.3e9]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc26_34.1: ref %IntRange.a89 = splice_block %return.param {}
// CHECK:STDOUT:   %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_28.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0, %.loc27_28.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.a89 to %.loc26_34.1 = call %IntRange.Make.specific_fn(%.loc27_28.2, %end.ref)
// CHECK:STDOUT:   return %IntRange.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.1df
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.8a9
// CHECK:STDOUT:   %.loc9 => constants.%.4f8
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9019d7.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.541
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.ff1
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.764eab.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.5f2
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.b16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.764eab.1
// CHECK:STDOUT:   %.loc9_85.1 => constants.%.4f8
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.7a8
// CHECK:STDOUT:   %facet_value.loc9_85.1 => constants.%facet_value
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type
// CHECK:STDOUT:   %Iterate.impl_witness.loc9_87.2 => constants.%Iterate.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.297
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type => constants.%IntRange.as.Iterate.impl.NewCursor.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor => constants.%IntRange.as.Iterate.impl.NewCursor
// CHECK:STDOUT:   %.loc11 => constants.%.f57
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type => constants.%IntRange.as.Iterate.impl.Next.type
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next => constants.%IntRange.as.Iterate.impl.Next
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.NewCursor(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %pattern_type.loc10_18 => constants.%pattern_type.b16
// CHECK:STDOUT:   %Int.loc10_45.1 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %.loc10_45.1 => constants.%.7b282e.1
// CHECK:STDOUT:   %pattern_type.loc10_32 => constants.%pattern_type.764eab.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.b16
// CHECK:STDOUT:   %Int.loc11_43.1 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %ptr.loc11_44.1 => constants.%ptr.c9c
// CHECK:STDOUT:   %pattern_type.loc11_25 => constants.%pattern_type.bda
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.7a8
// CHECK:STDOUT:   %facet_value => constants.%facet_value
// CHECK:STDOUT:   %.loc11_75.3 => constants.%.f57
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.b62
// CHECK:STDOUT:   %OptionalStorage.facet.loc11_75.1 => constants.%OptionalStorage.facet.01e
// CHECK:STDOUT:   %Optional.loc11_75.1 => constants.%Optional.e48
// CHECK:STDOUT:   %.loc11_75.4 => constants.%.6a7
// CHECK:STDOUT:   %pattern_type.loc11_47 => constants.%pattern_type.0c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.045
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.3e9
// CHECK:STDOUT:   %.loc9 => constants.%.14e
// CHECK:STDOUT:   %Int.loc22_32.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.a89
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.a58
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %complete_type.loc24_1.2 => constants.%complete_type.c45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %IntRange => constants.%IntRange.a89
// CHECK:STDOUT:   %.loc5_52.1 => constants.%.e73
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.615
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc5_52 => constants.%complete_type.c45
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %.loc6_22.1 => constants.%.14e
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f17
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.de4
// CHECK:STDOUT:   %.loc6_22.2 => constants.%.f79
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.664
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.b6d943.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.79a: type = struct_type {.start: %Int.b6d943.1, .end: %Int.b6d943.1} [symbolic]
// CHECK:STDOUT:   %complete_type.fb4: <witness> = complete_type_witness %struct_type.start.end.79a [symbolic]
// CHECK:STDOUT:   %IntRange.265: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.elem.302: type = unbound_element_type %IntRange.265, %Int.b6d943.1 [symbolic]
// CHECK:STDOUT:   %require_complete.2ded7d.1: <witness> = require_complete_type %Int.b6d943.1 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.bb68b6.1: type = pattern_type %Int.b6d943.1 [symbolic]
// CHECK:STDOUT:   %.b26: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.1df: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.8a9: %IntRange.Make.type.1df = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.b16: type = pattern_type %IntRange.265 [symbolic]
// CHECK:STDOUT:   %.5f2: form = init_form %IntRange.265, call_param2 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.4c7: <witness> = lookup_impl_witness %Int.b6d943.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.b6d943.1, (%Copy.lookup_impl_witness.4c7) [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.72c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.82f: %.72c = impl_witness_access %Copy.lookup_impl_witness.4c7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7d1: <specific function> = specific_impl_function %impl.elem0.82f, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %require_complete.8a1: <witness> = require_complete_type %IntRange.265 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.a89: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.045: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.3e9: %IntRange.Make.type.045 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %.fc8: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.elem.d76: type = unbound_element_type %IntRange.a89, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.0fe: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a43: <witness> = complete_type_witness %struct_type.start.end.0fe [concrete]
// CHECK:STDOUT:   %pattern_type.615: type = pattern_type %IntRange.a89 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cf3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.255: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.58d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e45, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cf3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.58d) [concrete]
// CHECK:STDOUT:   %.952: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.6b552a.1: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.4b6: <witness> = import_ref Main//lib, loc24_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.fb4)]
// CHECK:STDOUT:   %Main.import_ref.011 = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f89 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.e7c = import_ref Main//lib, loc22_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.8ef = import_ref Main//lib, loc23_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.6b552a.2: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b25: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.255)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e45 = impl_witness_table (%Core.import_ref.b25), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %Read.decl: %Read.type = fn_decl @Read [concrete = constants.%Read] {
// CHECK:STDOUT:     %y.patt: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_29.1: type = splice_block %.loc4_29.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core.ece [concrete = imports.%Core.ece]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_29.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_29.3: type = converted %IntLiteral.call, %.loc4_29.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y.loc4_9.2: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_9.1 (constants.%y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.6b552a.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.1df)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.1df) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.8a9)]
// CHECK:STDOUT:   %.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.1 (constants.%.b26)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.b6d943.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.2ded7d.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.302)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.b6d943.1), .end: @IntRange.%Int (%Int.b6d943.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.79a)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.fb4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.4b6
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.011
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.f89
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.e7c
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.8ef
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Read(%y.loc4_9.2: Core.IntLiteral) {
// CHECK:STDOUT:   %y.loc4_9.1: Core.IntLiteral = symbolic_binding y, 0 [symbolic = %y.loc4_9.1 (constants.%y)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y.loc4_9.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_31.3: <bound method> = bound_method %y.loc4_9.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_31.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.2: init %i32 = call %bound_method.loc5_31.3(%y.loc4_9.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.615 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.615 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref %IntRange.a89 = var %x.var_patt
// CHECK:STDOUT:     %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:     %y.ref: Core.IntLiteral = name_ref y, %y.loc4_9.2 [symbolic = %y.loc4_9.1 (constants.%y)]
// CHECK:STDOUT:     %.loc5_3: ref %IntRange.a89 = splice_block %x.var {}
// CHECK:STDOUT:     %impl.elem0: %.952 = impl_witness_access constants.%ImplicitAs.impl_witness.58d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4]
// CHECK:STDOUT:     %bound_method.loc5_31.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_31.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = %bound_method.loc5_31.3 (constants.%bound_method)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.1: init %i32 = call %bound_method.loc5_31.2(%y.ref) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_31.2: %i32 = converted %y.ref, %.loc5_31.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_31.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Range.call: init %IntRange.a89 to %.loc5_3 = call %Range.ref(%.loc5_31.2)
// CHECK:STDOUT:     assign %x.var, %Range.call
// CHECK:STDOUT:     %.loc5_21: type = splice_block %IntRange [concrete = constants.%IntRange.a89] {
// CHECK:STDOUT:       %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.a89]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %IntRange.a89 = ref_binding x, %x.var
// CHECK:STDOUT:     %DestroyOp.bound: <bound method> = bound_method %x.var, constants.%DestroyOp
// CHECK:STDOUT:     %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%x.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.6b552a.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.b6d943.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.bb68b6.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.265)]
// CHECK:STDOUT:   %.1: form = init_form %IntRange, call_param2 [symbolic = %.1 (constants.%.5f2)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.b16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %Int [symbolic = %require_complete.1 (constants.%require_complete.2ded7d.1)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %IntRange [symbolic = %require_complete.2 (constants.%require_complete.8a1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.b6d943.1), .end: @IntRange.Make.%Int (%Int.b6d943.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.79a)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.2 (constants.%.b26)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.4c7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.3 (constants.%.72c)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.3 (%.72c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.82f)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.7d1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %IntRange.a89) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Read(constants.%y) {
// CHECK:STDOUT:   %y.loc4_9.1 => constants.%y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.1df
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.8a9
// CHECK:STDOUT:   %.1 => constants.%.b26
// CHECK:STDOUT:   %Int => constants.%Int.b6d943.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2ded7d.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.302
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.79a
// CHECK:STDOUT:   %complete_type => constants.%complete_type.fb4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.b6d943.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.bb68b6.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.265
// CHECK:STDOUT:   %.1 => constants.%.5f2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.b16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.045
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.3e9
// CHECK:STDOUT:   %.1 => constants.%.fc8
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.a89
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.d76
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.0fe
// CHECK:STDOUT:   %complete_type => constants.%complete_type.a43
// CHECK:STDOUT: }
// CHECK:STDOUT:

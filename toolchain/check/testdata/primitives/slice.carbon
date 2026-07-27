// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/slice.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/slice.carbon

// --- test_slice.carbon

library "[[@TEST_NAME]]";

fn TestMake(p: i32*, n: u64) -> Core.Slice(i32) {
  //@dump-sem-ir-begin
  return Core.Slice(i32).Make(p, n);
  //@dump-sem-ir-end
}

fn TestSize(s: Core.Slice(i32)) -> u64 {
  //@dump-sem-ir-begin
  return s.Size();
  //@dump-sem-ir-end
}

fn TestCopy(s: Core.Slice(i32)) -> Core.Slice(i32) {
  //@dump-sem-ir-begin
  let t: Core.Slice(i32) = s;
  return t;
  //@dump-sem-ir-end
}

fn TestIterate(s: Core.Slice(i32)) -> i32 {
  var sum: i32 = 0;
  //@dump-sem-ir-begin
  for (x: i32 in s) {
    sum += x;
  }
  //@dump-sem-ir-end
  return sum;
}


fn TestUnformed(p: f64*, n: u64) -> Core.Slice(f64) {
  //@dump-sem-ir-begin
  var s: Core.Slice(f64);
  s = Core.Slice(f64).Make(p, n);
  return s;
  //@dump-sem-ir-end
}

fn TestRange(s: Core.Slice(i32), start: u64, count: u64) -> Core.Slice(i32) {
  //@dump-sem-ir-begin
  return s.Range(start, count);
  //@dump-sem-ir-end
}

fn TestKeyword(s: slice(i32)) -> slice(i32) {
  //@dump-sem-ir-begin
  return s;
  //@dump-sem-ir-end
}



// --- fail_todo_index.carbon

// According to p002274, this should work via a blanket impl of `IndexWith`
// for all types implementing `IndirectIndexWith`
fn TestIndex(s: Core.Slice(i32), i: u64) -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_index.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.IndexWith(u64)` in type `Core.Slice(i32)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return s[i];
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return s[i];
  //@dump-sem-ir-end
}







// CHECK:STDOUT: --- test_slice.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %Slice.type: type = generic_class_type @Slice [concrete]
// CHECK:STDOUT:   %Slice.generic: %Slice.type = struct_value () [concrete]
// CHECK:STDOUT:   %u64.builtin: type = int_type unsigned, %int_64 [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Slice.Range.type.cc1: type = fn_type @Slice.Range, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.Range.d3d: %Slice.Range.type.cc1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.Size.type.927: type = fn_type @Slice.Size, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.Size.e96: %Slice.Size.type.927 = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.Make.type.768: type = fn_type @Slice.Make, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.Make.ee0: %Slice.Make.type.768 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Slice.c12: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.162: type = pattern_type %Slice.c12 [concrete]
// CHECK:STDOUT:   %Slice.Make.type.3ec: type = fn_type @Slice.Make, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Make.34f: %Slice.Make.type.3ec = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Size.type.250: type = fn_type @Slice.Size, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Size.4fc: %Slice.Size.type.250 = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Range.type.0bd: type = fn_type @Slice.Range, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Range.a65: %Slice.Range.type.0bd = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Make.specific_fn.6dc: <specific function> = specific_function %Slice.Make.34f, @Slice.Make(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Size.specific_fn.03b: <specific function> = specific_function %Slice.Size.4fc, @Slice.Size(%i32) [concrete]
// CHECK:STDOUT:   %t.patt: %pattern_type.162 = value_binding_pattern t [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.8ae: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.c5b: %Slice.as.Copy.impl.Op.type.8ae = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.137: <witness> = impl_witness imports.%Copy.impl_witness_table.8b8, @Slice.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.4dc: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.789: %Slice.as.Copy.impl.Op.type.4dc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.848: %Copy.type = facet_value %Slice.c12, (%Copy.impl_witness.137) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f11: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.848) [concrete]
// CHECK:STDOUT:   %.606: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f11, %Copy.facet.848 [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.specific_fn.e5e: <specific function> = specific_function %Slice.as.Copy.impl.Op.789, @Slice.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.a1a: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.dc4: %facet_type.a1a = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.5e0: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.5db: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.985: %T.as_type.as.OptionalStorage.impl.Get.type.5db = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.7df: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.ade: %T.as_type.as.OptionalStorage.impl.Has.type.7df = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.fcb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.6ea: %T.as_type.as.OptionalStorage.impl.Some.type.fcb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.62f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.6bc3: %T.as_type.as.OptionalStorage.impl.None.type.62f = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.type.b97: type = fn_type @Slice.as.Iterate.impl.Next, @Slice.as.Iterate.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.728: %Slice.as.Iterate.impl.Next.type.b97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.type.33e: type = fn_type @Slice.as.Iterate.impl.NewCursor, @Slice.as.Iterate.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.a4e: %Slice.as.Iterate.impl.NewCursor.type.33e = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.c17: type = ptr_type %u64 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc26_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.4: <witness> = custom_witness (%Destroy.Op.1a2547.4), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.1fc: %facet_type.a1a = facet_value %i32, (%Copy.impl_witness.0e0, %custom_witness.df9cc1.4) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.817: <witness> = impl_witness imports.%Iterate.impl_witness_table.08d, @Slice.as.Iterate.impl(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.type.608: type = fn_type @Slice.as.Iterate.impl.NewCursor, @Slice.as.Iterate.impl(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.705: %Slice.as.Iterate.impl.NewCursor.type.608 = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.type.d50: type = fn_type @Slice.as.Iterate.impl.Next, @Slice.as.Iterate.impl(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.0de: %Slice.as.Iterate.impl.Next.type.d50 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.4a1: %Iterate.type = facet_value %Slice.c12, (%Iterate.impl_witness.817) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bf3: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.4a1) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b45: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.4a1) [concrete]
// CHECK:STDOUT:   %.5dc: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bf3, %Iterate.facet.4a1 [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %Slice.as.Iterate.impl.NewCursor.705, @Slice.as.Iterate.impl.NewCursor(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %.eba: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b45, %Iterate.facet.4a1 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.e35: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ca7: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.a87, @T.as_type.as.OptionalStorage.impl(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9bb: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.ca7) [concrete]
// CHECK:STDOUT:   %Optional.224: type = class_type @Optional, @Optional(%OptionalStorage.facet.9bb) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %Slice.as.Iterate.impl.Next.0de, @Slice.as.Iterate.impl.Next(%facet_value.1fc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.17d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.9bb) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.fb6: %Optional.HasValue.type.17d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.ea4: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.9bb) [concrete]
// CHECK:STDOUT:   %Optional.Get.9ad: %Optional.Get.type.ea4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d07318.2: %Destroy.type = facet_value %i32, (%custom_witness.df9cc1.4) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.023: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d07318.2) [concrete]
// CHECK:STDOUT:   %.1d6: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad0: type = struct_type {.value: %MaybeUnformed.023, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.fb6, @Optional.HasValue(%OptionalStorage.facet.9bb) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.9ad, @Optional.Get(%OptionalStorage.facet.9bb) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.2fb: type = generic_interface_type @AddAssignWith [concrete]
// CHECK:STDOUT:   %AddAssignWith.generic: %AddAssignWith.type.2fb = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.941: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.1: %Int.as.AddAssignWith.impl.Op.type.a58fbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.2: %Int.as.AddAssignWith.impl.Op.type.a58fbe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.138: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.36a: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.80b: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.23f: %Int.as.AnyInt.impl.AsInt.type.80b = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.76c: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.6b8: %facet_type.138 = facet_value %i32, (%custom_witness.36a, %AnyInt.impl_witness.76c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.03f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3b9, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.6b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.640: %ImplicitAs.type.914 = facet_value %i32, (%ImplicitAs.impl_witness.03f) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.ccf: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.640) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a94901.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.640) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.83e130.1: %Int.as.AddAssignWith.impl.Op.type.a94901.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a94901.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.640) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.83e130.2: %Int.as.AddAssignWith.impl.Op.type.a94901.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.b9a: %AddAssignWith.type.941 = facet_value %i32, (%AddAssignWith.impl_witness.ccf) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.51d: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.b9a) [concrete]
// CHECK:STDOUT:   %.98a: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.51d, %AddAssignWith.facet.b9a [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.80c75c.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.83e130.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.640) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.80c75c.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.83e130.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.640) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.loc26_19.8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.12: type = fn_type @Destroy.Op.loc26_19.10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.12: %Destroy.Op.type.1d8f74.12 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %ptr.749: type = ptr_type %f64 [concrete]
// CHECK:STDOUT:   %Slice.d46: type = class_type @Slice, @Slice(%f64) [concrete]
// CHECK:STDOUT:   %pattern_type.c4b: type = pattern_type %Slice.d46 [concrete]
// CHECK:STDOUT:   %Slice.Make.type.eac: type = fn_type @Slice.Make, @Slice(%f64) [concrete]
// CHECK:STDOUT:   %Slice.Make.2cb: %Slice.Make.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.ptr.size.b4d: type = struct_type {.ptr: %ptr.749, .size: %u64} [concrete]
// CHECK:STDOUT:   %s.patt.482: %pattern_type.c4b = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.c4b = var_pattern %s.patt.482 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.fef: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.215: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.52f(%T.fef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.f65: %T.as_type.as.DefaultOrUnformed.impl.Op.type.215 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.c4d: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.ed5, @Slice.as.UnformedInit.impl(%f64) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet: %UnformedInit.type = facet_value %Slice.d46, (%UnformedInit.impl_witness.c4d) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.f9c: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.972, @T.as_type.as.DefaultOrUnformed.impl.52f(%UnformedInit.facet) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Slice.d46, (%DefaultOrUnformed.impl_witness.f9c) [concrete]
// CHECK:STDOUT:   %Slice.Make.specific_fn.ffc: <specific function> = specific_function %Slice.Make.2cb, @Slice.Make(%f64) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.47e: <witness> = impl_witness imports.%Copy.impl_witness_table.8b8, @Slice.as.Copy.impl(%f64) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.032: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%f64) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.a11: %Slice.as.Copy.impl.Op.type.032 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.13f: %Copy.type = facet_value %Slice.d46, (%Copy.impl_witness.47e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a20: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.13f) [concrete]
// CHECK:STDOUT:   %.a42: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a20, %Copy.facet.13f [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.specific_fn.303: <specific function> = specific_function %Slice.as.Copy.impl.Op.a11, @Slice.as.Copy.impl.Op(%f64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.15: type = fn_type @Destroy.Op.loc36_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.15: %Destroy.Op.type.1d8f74.15 = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Range.specific_fn: <specific function> = specific_function %Slice.Range.a65, @Slice.Range(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Slice = %Core.Slice
// CHECK:STDOUT:     .Copy = %Core.Copy.e5d
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .AddAssignWith = %Core.AddAssignWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.Slice: %Slice.type = import_ref Core//prelude/types/slice, Slice, loaded [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %Core.import_ref.07e: @Slice.%Slice.Make.type (%Slice.Make.type.768) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Make (constants.%Slice.Make.ee0)]
// CHECK:STDOUT:   %Core.import_ref.0c9: @Slice.%Slice.Size.type (%Slice.Size.type.927) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Size (constants.%Slice.Size.e96)]
// CHECK:STDOUT:   %Core.import_ref.e9f: @Slice.%Slice.Range.type (%Slice.Range.type.cc1) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Range (constants.%Slice.Range.d3d)]
// CHECK:STDOUT:   %Core.Copy.e5d: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.516e: @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op.type (%Slice.as.Copy.impl.Op.type.8ae) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op (constants.%Slice.as.Copy.impl.Op.c5b)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8b8 = impl_witness_table (%Core.import_ref.516e), @Slice.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.2a4: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.7fa) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.643)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3b9 = impl_witness_table (%Core.import_ref.2a4), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Iterate: type = import_ref Core//prelude/iterate, Iterate, loaded [concrete = constants.%Iterate.type]
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.5e0)]
// CHECK:STDOUT:   %Core.import_ref.726: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.62f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.6bc3)]
// CHECK:STDOUT:   %Core.import_ref.8d6: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.fcb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.6ea)]
// CHECK:STDOUT:   %Core.import_ref.fdc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.7df) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.ade)]
// CHECK:STDOUT:   %Core.import_ref.9ff: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.5db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.985)]
// CHECK:STDOUT:   %Core.import_ref.cea = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.a87 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.726, %Core.import_ref.8d6, %Core.import_ref.fdc, %Core.import_ref.9ff, %Core.import_ref.cea), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9a513d.2 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fb56 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d670: @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.NewCursor.type (%Slice.as.Iterate.impl.NewCursor.type.33e) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.NewCursor (constants.%Slice.as.Iterate.impl.NewCursor.a4e)]
// CHECK:STDOUT:   %Core.import_ref.fca: @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.Next.type (%Slice.as.Iterate.impl.Next.type.b97) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.Next (constants.%Slice.as.Iterate.impl.Next.728)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.08d = impl_witness_table (%Core.import_ref.9a513d.2, %Core.import_ref.fb56, %Core.import_ref.d670, %Core.import_ref.fca), @Slice.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.AddAssignWith: %AddAssignWith.type.2fb = import_ref Core//prelude/operators/arithmetic, AddAssignWith, loaded [concrete = constants.%AddAssignWith.generic]
// CHECK:STDOUT:   %Core.import_ref.098: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.45b360.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.098), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.45b360.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.fd3: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.80b) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.23f)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.fd3), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.083f: @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.215) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.f65)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.972 = impl_witness_table (%Core.import_ref.083f), @T.as_type.as.DefaultOrUnformed.impl.52f [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.ed5 = impl_witness_table (), @Slice.as.UnformedInit.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestMake(%p.param: %ptr.d08, %n.param: %u64) -> out %return.param: %Slice.c12 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref.loc6: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Slice.ref.loc6: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Slice.loc6: type = class_type @Slice, @Slice(constants.%i32) [concrete = constants.%Slice.c12]
// CHECK:STDOUT:   %.loc6: %Slice.Make.type.3ec = specific_constant imports.%Core.import_ref.07e, @Slice(constants.%i32) [concrete = constants.%Slice.Make.34f]
// CHECK:STDOUT:   %Make.ref: %Slice.Make.type.3ec = name_ref Make, %.loc6 [concrete = constants.%Slice.Make.34f]
// CHECK:STDOUT:   %p.ref: %ptr.d08 = name_ref p, %p
// CHECK:STDOUT:   %n.ref: %u64 = name_ref n, %n
// CHECK:STDOUT:   %Slice.Make.specific_fn: <specific function> = specific_function %Make.ref, @Slice.Make(constants.%i32) [concrete = constants.%Slice.Make.specific_fn.6dc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.Make.call: init %Slice.c12 to %.loc4_47.1 = call %Slice.Make.specific_fn(%p.ref, %n.ref)
// CHECK:STDOUT:   return %Slice.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSize(%s.param: %Slice.c12) -> out %return.param: %u64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.c12 = name_ref s, %s
// CHECK:STDOUT:   %.loc12: %Slice.Size.type.250 = specific_constant imports.%Core.import_ref.0c9, @Slice(constants.%i32) [concrete = constants.%Slice.Size.4fc]
// CHECK:STDOUT:   %Size.ref: %Slice.Size.type.250 = name_ref Size, %.loc12 [concrete = constants.%Slice.Size.4fc]
// CHECK:STDOUT:   %Slice.Size.bound: <bound method> = bound_method %s.ref, %Size.ref
// CHECK:STDOUT:   %Slice.Size.specific_fn: <specific function> = specific_function %Size.ref, @Slice.Size(constants.%i32) [concrete = constants.%Slice.Size.specific_fn.03b]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.ref, %Slice.Size.specific_fn
// CHECK:STDOUT:   %Slice.Size.call: init %u64 = call %bound_method(%s.ref)
// CHECK:STDOUT:   return %Slice.Size.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCopy(%s.param: %Slice.c12) -> out %return.param: %Slice.c12 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.c12 = name_ref s, %s
// CHECK:STDOUT:   %.loc18: type = splice_block %Slice.loc18 [concrete = constants.%Slice.c12] {
// CHECK:STDOUT:     %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Slice.ref.loc18: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:     %i32.loc18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Slice.loc18: type = class_type @Slice, @Slice(constants.%i32) [concrete = constants.%Slice.c12]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: %Slice.c12 = wrapper_binding t, %s.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.162 = value_binding_pattern t [concrete = constants.%t.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.ref: %Slice.c12 = name_ref t, %t
// CHECK:STDOUT:   %impl.elem0: %.606 = impl_witness_access constants.%Copy.impl_witness.137, element0 [concrete = constants.%Slice.as.Copy.impl.Op.789]
// CHECK:STDOUT:   %bound_method.loc19_10.1: <bound method> = bound_method %t.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Slice.as.Copy.impl.Op(constants.%i32) [concrete = constants.%Slice.as.Copy.impl.Op.specific_fn.e5e]
// CHECK:STDOUT:   %bound_method.loc19_10.2: <bound method> = bound_method %t.ref, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.call: init %Slice.c12 to %.loc16_50.1 = call %bound_method.loc19_10.2(%t.ref)
// CHECK:STDOUT:   return %Slice.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIterate(%s.param: %Slice.c12) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Slice.c12 = name_ref s, %s
// CHECK:STDOUT:   %impl.elem2: %.5dc = impl_witness_access constants.%Iterate.impl_witness.817, element2 [concrete = constants.%Slice.as.Iterate.impl.NewCursor.705]
// CHECK:STDOUT:   %bound_method.loc26_19.1: <bound method> = bound_method %s.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc26_19.1: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.0e0, constants.%custom_witness.df9cc1.4) [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %.loc26_19.1: %facet_type.a1a = converted constants.%i32, %facet_value.loc26_19.1 [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %facet_value.loc26_19.2: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.0e0, constants.%custom_witness.df9cc1.4) [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %.loc26_19.2: %facet_type.a1a = converted constants.%i32, %facet_value.loc26_19.2 [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %specific_fn.loc26_19.1: <specific function> = specific_function %impl.elem2, @Slice.as.Iterate.impl.NewCursor(constants.%facet_value.1fc) [concrete = constants.%Slice.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_19.2: <bound method> = bound_method %s.ref, %specific_fn.loc26_19.1
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.call: init %u64 = call %bound_method.loc26_19.2(%s.ref)
// CHECK:STDOUT:   %var: ref %u64 = var_storage invalid
// CHECK:STDOUT:   assign %var, %Slice.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c17 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.eba = impl_witness_access constants.%Iterate.impl_witness.817, element3 [concrete = constants.%Slice.as.Iterate.impl.Next.0de]
// CHECK:STDOUT:   %bound_method.loc26_19.3: <bound method> = bound_method %s.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc26_19.3: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.0e0, constants.%custom_witness.df9cc1.4) [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %.loc26_19.3: %facet_type.a1a = converted constants.%i32, %facet_value.loc26_19.3 [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %facet_value.loc26_19.4: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.0e0, constants.%custom_witness.df9cc1.4) [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %.loc26_19.4: %facet_type.a1a = converted constants.%i32, %facet_value.loc26_19.4 [concrete = constants.%facet_value.1fc]
// CHECK:STDOUT:   %specific_fn.loc26_19.2: <specific function> = specific_function %impl.elem3, @Slice.as.Iterate.impl.Next(constants.%facet_value.1fc) [concrete = constants.%Slice.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_19.4: <bound method> = bound_method %s.ref, %specific_fn.loc26_19.2
// CHECK:STDOUT:   %.loc26_19.5: ref %Optional.224 = temporary_storage
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.call: init %Optional.224 to %.loc26_19.5 = call %bound_method.loc26_19.4(%s.ref, %addr)
// CHECK:STDOUT:   %.loc26_19.6: ref %Optional.224 = temporary %.loc26_19.5, %Slice.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc26_19.7: %Optional.HasValue.type.17d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.9bb) [concrete = constants.%Optional.HasValue.fb6]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.17d = name_ref HasValue, %.loc26_19.7 [concrete = constants.%Optional.HasValue.fb6]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.9bb) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc26_19.8: %Optional.224 = acquire_value %.loc26_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.8)
// CHECK:STDOUT:   %.loc26_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc26_19.10: bool = converted %Optional.HasValue.call, %.loc26_19.9
// CHECK:STDOUT:   if %.loc26_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc26_19.11: %Optional.Get.type.ea4 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.9bb) [concrete = constants.%Optional.Get.9ad]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.ea4 = name_ref Get, %.loc26_19.11 [concrete = constants.%Optional.Get.9ad]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc26_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.9bb) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc26_19.12: %Optional.224 = acquire_value %.loc26_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %i32 = call %bound_method.loc26_19.6(%.loc26_19.12)
// CHECK:STDOUT:   %.loc26_19.13: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc26_19.14: %i32 = converted %Optional.Get.call, %.loc26_19.13
// CHECK:STDOUT:   %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %.loc26_19.14
// CHECK:STDOUT:   %sum.ref.loc27: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0.loc27: %.98a = impl_witness_access constants.%AddAssignWith.impl_witness.ccf, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.83e130.2]
// CHECK:STDOUT:   %bound_method.loc27_9.1: <bound method> = bound_method %sum.ref.loc27, %impl.elem0.loc27
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.640) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.80c75c.1]
// CHECK:STDOUT:   %bound_method.loc27_9.2: <bound method> = bound_method %sum.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27: %Int.as.AddAssignWith.impl.Op.type.a94901.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.640) [concrete = constants.%Int.as.AddAssignWith.impl.Op.83e130.1]
// CHECK:STDOUT:   %Op.ref: %Int.as.AddAssignWith.impl.Op.type.a94901.1 = name_ref Op, %.loc27 [concrete = constants.%Int.as.AddAssignWith.impl.Op.83e130.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref.loc27, %Op.ref
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.640) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.80c75c.2]
// CHECK:STDOUT:   %bound_method.loc27_9.3: <bound method> = bound_method %sum.ref.loc27, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc27_9.3(%sum.ref.loc27, %x.ref)
// CHECK:STDOUT:   %Destroy.Op.bound.loc26_19.1: <bound method> = bound_method %.loc26_19.6, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc26_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.1(%.loc26_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc26_19.2: <bound method> = bound_method %.loc26_19.6, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc26_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.2(%.loc26_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc26_19.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.12
// CHECK:STDOUT:   %Destroy.Op.call.loc26_19.3: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.3(%self.param: ref %.1d6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.4(%self.param: ref %MaybeUnformed.023) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.5(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.6(%self.param: ref %struct_type.value.has_value.ad0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.7(%self.param: ref %DefaultOptionalStorage.e35) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.8(%self.param: ref %Optional.224) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.9(%self.param: ref %u64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.10(%self.param: ref %u64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnformed(%p.param: %ptr.749, %n.param: %u64) -> out %return.param: %Slice.d46 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %Slice.d46 = var_storage %s.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Slice.d46, (constants.%DefaultOrUnformed.impl_witness.f9c) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc36_25.1: %DefaultOrUnformed.type = converted constants.%Slice.d46, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc36_25.1 [concrete = constants.%Slice.d46]
// CHECK:STDOUT:   %.loc36_25.2: type = converted %.loc36_25.1, %as_type [concrete = constants.%Slice.d46]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc36_3: ref %Slice.d46 = splice_block %s.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %Slice.d46 to %.loc36_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %s.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc36_24: type = splice_block %Slice.loc36 [concrete = constants.%Slice.d46] {
// CHECK:STDOUT:     %Core.ref.loc36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Slice.ref.loc36: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:     %f64.loc36: type = type_literal constants.%f64 [concrete = constants.%f64]
// CHECK:STDOUT:     %Slice.loc36: type = class_type @Slice, @Slice(constants.%f64) [concrete = constants.%Slice.d46]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %Slice.d46 = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.c4b = ref_binding_pattern s [concrete = constants.%s.patt.482]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.c4b = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc37: ref %Slice.d46 = name_ref s, %s
// CHECK:STDOUT:   %Core.ref.loc37: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Slice.ref.loc37: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %f64.loc37: type = type_literal constants.%f64 [concrete = constants.%f64]
// CHECK:STDOUT:   %Slice.loc37: type = class_type @Slice, @Slice(constants.%f64) [concrete = constants.%Slice.d46]
// CHECK:STDOUT:   %.loc37_22: %Slice.Make.type.eac = specific_constant imports.%Core.import_ref.07e, @Slice(constants.%f64) [concrete = constants.%Slice.Make.2cb]
// CHECK:STDOUT:   %Make.ref: %Slice.Make.type.eac = name_ref Make, %.loc37_22 [concrete = constants.%Slice.Make.2cb]
// CHECK:STDOUT:   %p.ref: %ptr.749 = name_ref p, %p
// CHECK:STDOUT:   %n.ref: %u64 = name_ref n, %n
// CHECK:STDOUT:   %Slice.Make.specific_fn: <specific function> = specific_function %Make.ref, @Slice.Make(constants.%f64) [concrete = constants.%Slice.Make.specific_fn.ffc]
// CHECK:STDOUT:   %.loc37_3: ref %Slice.d46 = splice_block %s.ref.loc37 {}
// CHECK:STDOUT:   %Slice.Make.call: init %Slice.d46 to %.loc37_3 = call %Slice.Make.specific_fn(%p.ref, %n.ref)
// CHECK:STDOUT:   assign %s.ref.loc37, %Slice.Make.call
// CHECK:STDOUT:   %s.ref.loc38: ref %Slice.d46 = name_ref s, %s
// CHECK:STDOUT:   %.loc38: %Slice.d46 = acquire_value %s.ref.loc38
// CHECK:STDOUT:   %impl.elem0: %.a42 = impl_witness_access constants.%Copy.impl_witness.47e, element0 [concrete = constants.%Slice.as.Copy.impl.Op.a11]
// CHECK:STDOUT:   %bound_method.loc38_10.1: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Slice.as.Copy.impl.Op(constants.%f64) [concrete = constants.%Slice.as.Copy.impl.Op.specific_fn.303]
// CHECK:STDOUT:   %bound_method.loc38_10.2: <bound method> = bound_method %.loc38, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.call: init %Slice.d46 to %.loc34_51.1 = call %bound_method.loc38_10.2(%.loc38)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %s.var, constants.%Destroy.Op.1a2547.15
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%s.var)
// CHECK:STDOUT:   return %Slice.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc36_3.1(%self.param: ref %ptr.749) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc36_3.2(%self.param: ref %struct_type.ptr.size.b4d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc36_3.3(%self.param: ref %Slice.d46) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRange(%s.param: %Slice.c12, %start.param: %u64, %count.param: %u64) -> out %return.param: %Slice.c12 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.c12 = name_ref s, %s
// CHECK:STDOUT:   %.loc44: %Slice.Range.type.0bd = specific_constant imports.%Core.import_ref.e9f, @Slice(constants.%i32) [concrete = constants.%Slice.Range.a65]
// CHECK:STDOUT:   %Range.ref: %Slice.Range.type.0bd = name_ref Range, %.loc44 [concrete = constants.%Slice.Range.a65]
// CHECK:STDOUT:   %Slice.Range.bound: <bound method> = bound_method %s.ref, %Range.ref
// CHECK:STDOUT:   %start.ref: %u64 = name_ref start, %start
// CHECK:STDOUT:   %count.ref: %u64 = name_ref count, %count
// CHECK:STDOUT:   %Slice.Range.specific_fn: <specific function> = specific_function %Range.ref, @Slice.Range(constants.%i32) [concrete = constants.%Slice.Range.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.ref, %Slice.Range.specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.Range.call: init %Slice.c12 to %.loc42_75.1 = call %bound_method(%s.ref, %start.ref, %count.ref)
// CHECK:STDOUT:   return %Slice.Range.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestKeyword(%s.param: %Slice.c12) -> out %return.param: %Slice.c12 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.c12 = name_ref s, %s
// CHECK:STDOUT:   %impl.elem0: %.606 = impl_witness_access constants.%Copy.impl_witness.137, element0 [concrete = constants.%Slice.as.Copy.impl.Op.789]
// CHECK:STDOUT:   %bound_method.loc50_10.1: <bound method> = bound_method %s.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Slice.as.Copy.impl.Op(constants.%i32) [concrete = constants.%Slice.as.Copy.impl.Op.specific_fn.e5e]
// CHECK:STDOUT:   %bound_method.loc50_10.2: <bound method> = bound_method %s.ref, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.call: init %Slice.c12 to %.loc48_43.1 = call %bound_method.loc50_10.2(%s.ref)
// CHECK:STDOUT:   return %Slice.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.c12: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIndex(%s.param: %Slice.c12, %i.param: %u64) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.c12 = name_ref s, %s
// CHECK:STDOUT:   %i.ref: %u64 = name_ref i, %i
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

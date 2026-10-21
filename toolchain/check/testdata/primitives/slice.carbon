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

// --- make.carbon

library "[[@TEST_NAME]]";

fn TestMake(p: i32*, n: u64) -> Core.Slice(i32) {
  //@dump-sem-ir-begin
  return Core.Slice(i32).Make(p, n);
  //@dump-sem-ir-end
}

// --- size.carbon

library "[[@TEST_NAME]]";

fn TestSize(s: Core.Slice(i32)) -> u64 {
  //@dump-sem-ir-begin
  return s.Size();
  //@dump-sem-ir-end
}

// --- copy.carbon

library "[[@TEST_NAME]]";

fn TestCopy(s: Core.Slice(i32)) -> Core.Slice(i32) {
  //@dump-sem-ir-begin
  return s;
  //@dump-sem-ir-end
}

// --- iterate.carbon

library "[[@TEST_NAME]]";

fn TestIterate(s: Core.Slice(i32)) -> i32 {
  var sum: i32 = 0;
  //@dump-sem-ir-begin
  for (x: i32 in s) {
    sum += x;
  }
  //@dump-sem-ir-end
  return sum;
}

// --- unformed.carbon

library "[[@TEST_NAME]]";

fn TestUnformed(p: f64*, n: u64) -> Core.Slice(f64) {
  //@dump-sem-ir-begin
  var s: Core.Slice(f64);
  s = Core.Slice(f64).Make(p, n);
  return s;
  //@dump-sem-ir-end
}

// --- range.carbon

library "[[@TEST_NAME]]";

fn TestRange(s: Core.Slice(i32), start: u64, count: u64) -> Core.Slice(i32) {
  //@dump-sem-ir-begin
  return s.Range(start, count);
  //@dump-sem-ir-end
}

// --- keyword.carbon

library "[[@TEST_NAME]]";

fn TestKeyword(s: slice(i32)) -> slice(i32) {
  //@dump-sem-ir-begin
  return s;
  //@dump-sem-ir-end
}

// --- index.carbon

library "[[@TEST_NAME]]";

fn TestIndex(s: Core.Slice(i32), i: u64) -> i32 {
  //@dump-sem-ir-begin
  return s[i];
  //@dump-sem-ir-end
}

// --- array_to_slice_pointer.carbon

library "[[@TEST_NAME]]";

fn TakesSlice(s: Core.Slice(i32)) -> i32;

fn CallWithArrayPointer(a: array(i32, 5)*) -> i32 {
  //@dump-sem-ir-begin
  return TakesSlice(a);
  //@dump-sem-ir-end
}

// --- array_to_slice_var.carbon

library "[[@TEST_NAME]]";

fn TakesSlice(s: Core.Slice(i32)) -> i32;

fn CallWithAddressOfLocalVariable() -> i32 {
  var b: array(i32, 5) = (1, 2, 3, 4, 5);
  //@dump-sem-ir-begin
  return TakesSlice(&b);
  //@dump-sem-ir-end
}

// --- fail_array_to_slice_param.carbon

library "[[@TEST_NAME]]";

fn TakesSlice(s: Core.Slice(i32)) -> i32;

fn CallWithAddressOfParameter(a: array(i32, 5)) -> i32 {
  // NOTE: This is expected to fail because parameters are non-references in Carbon.
  // CHECK:STDERR: fail_array_to_slice_param.carbon:[[@LINE+2]]:21: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:
  return TakesSlice(&a);
}

// CHECK:STDOUT: --- make.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %UInt.type: type = parameterized_class_name_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %Slice.type: type = parameterized_class_name_type @Slice [concrete]
// CHECK:STDOUT:   %Slice.generic: %Slice.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Slice.Make.type.a71: type = fn_type @Slice.Make, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.Make.9a2: %Slice.Make.type.a71 = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Make.type.d45: type = fn_type @Slice.Make, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Make.326: %Slice.Make.type.d45 = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Make.specific_fn.bea: <specific function> = specific_function %Slice.Make.326, @Slice.Make(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Slice = %Core.Slice
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.Slice: %Slice.type = import_ref Core//prelude/types/slice, Slice, loaded [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %Core.import_ref.a05: @Slice.%Slice.Make.type (%Slice.Make.type.a71) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Make (constants.%Slice.Make.9a2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestMake(%p.param: %ptr.a17, %n.param: %u64) -> out %return.param: %Slice.5af {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref.loc6: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Slice.ref.loc6: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Slice.loc6: type = class_type @Slice, @Slice(constants.%i32) [concrete = constants.%Slice.5af]
// CHECK:STDOUT:   %.loc6: %Slice.Make.type.d45 = specific_constant imports.%Core.import_ref.a05, @Slice(constants.%i32) [concrete = constants.%Slice.Make.326]
// CHECK:STDOUT:   %Make.ref: %Slice.Make.type.d45 = name_ref Make, %.loc6 [concrete = constants.%Slice.Make.326]
// CHECK:STDOUT:   %p.ref: %ptr.a17 = name_ref p, %p
// CHECK:STDOUT:   %n.ref: %u64 = name_ref n, %n
// CHECK:STDOUT:   %Slice.Make.specific_fn: <specific function> = specific_function %Make.ref, @Slice.Make(constants.%i32) [concrete = constants.%Slice.Make.specific_fn.bea]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.Make.call: init %Slice.5af to %.loc4_47.1 = call %Slice.Make.specific_fn(%p.ref, %n.ref)
// CHECK:STDOUT:   return %Slice.Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Slice.Size.type.458: type = fn_type @Slice.Size, @Slice(%T) [symbolic]
// CHECK:STDOUT:   %Slice.Size.5ca: %Slice.Size.type.458 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Size.type.05f: type = fn_type @Slice.Size, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Size.edb: %Slice.Size.type.05f = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Size.specific_fn: <specific function> = specific_function %Slice.Size.edb, @Slice.Size(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e71: @Slice.%Slice.Size.type (%Slice.Size.type.458) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Size (constants.%Slice.Size.5ca)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSize(%s.param: %Slice.5af) -> out %return.param: %u64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.5af = name_ref s, %s
// CHECK:STDOUT:   %.loc6: %Slice.Size.type.05f = specific_constant imports.%Core.import_ref.e71, @Slice(constants.%i32) [concrete = constants.%Slice.Size.edb]
// CHECK:STDOUT:   %Size.ref: %Slice.Size.type.05f = name_ref Size, %.loc6 [concrete = constants.%Slice.Size.edb]
// CHECK:STDOUT:   %Slice.Size.bound: <bound method> = bound_method %s.ref, %Size.ref
// CHECK:STDOUT:   %Slice.Size.specific_fn: <specific function> = specific_function %Size.ref, @Slice.Size(constants.%i32) [concrete = constants.%Slice.Size.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.ref, %Slice.Size.specific_fn
// CHECK:STDOUT:   %Slice.Size.call: init %u64 = call %bound_method(%s.ref)
// CHECK:STDOUT:   return %Slice.Size.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.253: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.7e2: %Slice.as.Copy.impl.Op.type.253 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.714: <witness> = impl_witness imports.%Copy.impl_witness_table.22d, @Slice.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.344: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.246: %Slice.as.Copy.impl.Op.type.344 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b52: %Copy.type = facet_value %Slice.5af, (%Copy.impl_witness.714) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.950: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b52) [concrete]
// CHECK:STDOUT:   %.d5b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.950, %Copy.facet.b52 [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Slice.as.Copy.impl.Op.246, @Slice.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e2: @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op.type (%Slice.as.Copy.impl.Op.type.253) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op (constants.%Slice.as.Copy.impl.Op.7e2)]
// CHECK:STDOUT:   %Copy.impl_witness_table.22d = impl_witness_table (%Core.import_ref.7e2), @Slice.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCopy(%s.param: %Slice.5af) -> out %return.param: %Slice.5af {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.5af = name_ref s, %s
// CHECK:STDOUT:   %impl.elem0: %.d5b = impl_witness_access constants.%Copy.impl_witness.714, element0 [concrete = constants.%Slice.as.Copy.impl.Op.246]
// CHECK:STDOUT:   %bound_method.loc6_10.1: <bound method> = bound_method %s.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Slice.as.Copy.impl.Op(constants.%i32) [concrete = constants.%Slice.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_10.2: <bound method> = bound_method %s.ref, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.call: init %Slice.5af to %.loc4_50.1 = call %bound_method.loc6_10.2(%s.ref)
// CHECK:STDOUT:   return %Slice.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- iterate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %facet_type.08a: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.d4bbe0.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.170: %facet_type.08a = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.386: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.170) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.85a: %From.as_type.as.ImplicitAs.impl.Convert.type.386 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.0ee: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.497: %Int.as.AnyInt.impl.AsInt.type.0ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.a1a: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.dc4: %facet_type.a1a = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.5d6: type = facet_access_type %T.dc4 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.491: <witness> = lookup_impl_witness %T.dc4, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.589: %Destroy.type = facet_value %T.as_type.5d6, (%Destroy.lookup_impl_witness.491) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.f70: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.589) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.b57: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.dc4) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.527: type = unbound_element_type %DefaultOptionalStorage.b57, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.08b: type = unbound_element_type %DefaultOptionalStorage.b57, %MaybeUnformed.f70 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.428: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.736: %T.as_type.as.OptionalStorage.impl.Get.type.428 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d98: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.b3f: %T.as_type.as.OptionalStorage.impl.Has.type.d98 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.91d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.a45: %T.as_type.as.OptionalStorage.impl.Some.type.91d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.6a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.87e: %T.as_type.as.OptionalStorage.impl.None.type.6a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.1, %Destroy.WithSelf.SubobjectDestroy.d01daf.1, %Destroy.WithSelf.SelfDestruct.db3fdb.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.ed0: %Destroy.type = facet_value %i32, (%custom_witness.f8f19d.1) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.type.15f: type = fn_type @Slice.as.Iterate.impl.Next, @Slice.as.Iterate.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.876: %Slice.as.Iterate.impl.Next.type.15f = struct_value () [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.type.136: type = fn_type @Slice.as.Iterate.impl.NewCursor, @Slice.as.Iterate.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.57e: %Slice.as.Iterate.impl.NewCursor.type.136 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.090: %facet_type.a1a = facet_value %i32, (%Copy.impl_witness.8af, %custom_witness.f8f19d.1) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.a42: <witness> = impl_witness imports.%Iterate.impl_witness_table.5a1, @Slice.as.Iterate.impl(%facet_value.090) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.type.5be: type = fn_type @Slice.as.Iterate.impl.NewCursor, @Slice.as.Iterate.impl(%facet_value.090) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.f43: %Slice.as.Iterate.impl.NewCursor.type.5be = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.type.cf9: type = fn_type @Slice.as.Iterate.impl.Next, @Slice.as.Iterate.impl(%facet_value.090) [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.c79: %Slice.as.Iterate.impl.Next.type.cf9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7ca: %Iterate.type = facet_value %Slice.5af, (%Iterate.impl_witness.a42) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.6e1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7ca) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.eee: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7ca) [concrete]
// CHECK:STDOUT:   %.fdd: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.6e1, %Iterate.facet.7ca [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %Slice.as.Iterate.impl.NewCursor.f43, @Slice.as.Iterate.impl.NewCursor(%facet_value.090) [concrete]
// CHECK:STDOUT:   %.231: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.eee, %Iterate.facet.7ca [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.d2f: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.090) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.c95: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.0d6, @T.as_type.as.OptionalStorage.impl(%facet_value.090) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.ae9: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.c95) [concrete]
// CHECK:STDOUT:   %Optional.bf9: type = class_type @Optional, @Optional(%OptionalStorage.facet.ae9) [concrete]
// CHECK:STDOUT:   %pattern_type.c16: type = pattern_type %Optional.bf9 [concrete]
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %Slice.as.Iterate.impl.Next.c79, @Slice.as.Iterate.impl.Next(%facet_value.090) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.6ac: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.ae9) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.e41: %Optional.HasValue.type.6ac = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.4fb: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.ae9) [concrete]
// CHECK:STDOUT:   %Optional.Get.65d: %Optional.Get.type.4fb = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.993: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.ed0) [concrete]
// CHECK:STDOUT:   %.b31: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.079: type = unbound_element_type %DefaultOptionalStorage.d2f, %MaybeUnformed.993 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.93e: type = unbound_element_type %DefaultOptionalStorage.d2f, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.2e0: type = struct_type {.value: %MaybeUnformed.993, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.e41, @Optional.HasValue(%OptionalStorage.facet.ae9) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.65d, @Optional.Get(%OptionalStorage.facet.ae9) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.1ad: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.1: %Int.as.AddAssignWith.impl.Op.type.93140c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.2: %Int.as.AddAssignWith.impl.Op.type.93140c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.0f0: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.633: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %AnyInt.impl_witness.83b: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.709: %facet_type.0f0 = facet_value %i32, (%custom_witness.633, %AnyInt.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.943: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8f1, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.709) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c18: %ImplicitAs.type.6b0 = facet_value %i32, (%ImplicitAs.impl_witness.943) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.95a: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.47f4e6.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bff4f9.1: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.47f4e6.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bff4f9.2: %Int.as.AddAssignWith.impl.Op.type.47f4e6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d53: %AddAssignWith.type.1ad = facet_value %i32, (%AddAssignWith.impl_witness.95a) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.313: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.d53) [concrete]
// CHECK:STDOUT:   %.316: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.313, %AddAssignWith.facet.d53 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bff4f9.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bff4f9.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %pattern_type.16e: type = pattern_type %.b31 [concrete]
// CHECK:STDOUT:   %self.param_patt.214: %pattern_type.16e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a42: %pattern_type.16e = wrapper_binding_pattern self, %self.param_patt.214 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc7_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.59f: type = pattern_type %MaybeUnformed.993 [concrete]
// CHECK:STDOUT:   %self.param_patt.f37: %pattern_type.59f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c6e: %pattern_type.59f = wrapper_binding_pattern self, %self.param_patt.f37 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc7_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc7_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %struct_type.value.has_value.2e0 [concrete]
// CHECK:STDOUT:   %self.param_patt.4dc: %pattern_type.0fa = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0fc: %pattern_type.0fa = wrapper_binding_pattern self, %self.param_patt.4dc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc7_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f68: type = pattern_type %DefaultOptionalStorage.d2f [concrete]
// CHECK:STDOUT:   %self.param_patt.98b: %pattern_type.f68 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.868: %pattern_type.f68 = wrapper_binding_pattern self, %self.param_patt.98b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc7_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.336: %pattern_type.c16 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7ed: %pattern_type.c16 = wrapper_binding_pattern self, %self.param_patt.336 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc7_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.39b: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.386) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.85a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8f1 = impl_witness_table (%Core.import_ref.39b), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.323: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.0ee) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.497)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.323), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.02f: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.b57)]
// CHECK:STDOUT:   %Core.import_ref.bd5: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.6a1) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.87e)]
// CHECK:STDOUT:   %Core.import_ref.3c9: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.91d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.a45)]
// CHECK:STDOUT:   %Core.import_ref.d9b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d98) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.b3f)]
// CHECK:STDOUT:   %Core.import_ref.c0b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.428) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.736)]
// CHECK:STDOUT:   %Core.import_ref.af7 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.0d6 = impl_witness_table (%Core.import_ref.02f, %Core.import_ref.bd5, %Core.import_ref.3c9, %Core.import_ref.d9b, %Core.import_ref.c0b, %Core.import_ref.af7), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c2a: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.08b) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.7cb]
// CHECK:STDOUT:   %Core.import_ref.41b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.527) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.06c]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e9: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a513d.2 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.18e = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.628: @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.NewCursor.type (%Slice.as.Iterate.impl.NewCursor.type.136) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.NewCursor (constants.%Slice.as.Iterate.impl.NewCursor.57e)]
// CHECK:STDOUT:   %Core.import_ref.7619: @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.Next.type (%Slice.as.Iterate.impl.Next.type.15f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Iterate.impl.%Slice.as.Iterate.impl.Next (constants.%Slice.as.Iterate.impl.Next.876)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.5a1 = impl_witness_table (%Core.import_ref.9a513d.2, %Core.import_ref.18e, %Core.import_ref.628, %Core.import_ref.7619), @Slice.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b25: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.93140c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.b25), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.da0: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.93140c.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.2)]
// CHECK:STDOUT:   %Core.import_ref.b0d0 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.06c: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.527) = field_decl has_value, element1, %Core.import_ref.b0d0 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.ab7 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.7cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.08b) = field_decl value, element0, %Core.import_ref.ab7 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.16e = ref_param_pattern [concrete = constants.%self.param_patt.214]
// CHECK:STDOUT:     %self.patt: %pattern_type.16e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a42]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.b31 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.b31 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc7_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.16e = ref_param_pattern [concrete = constants.%self.param_patt.214]
// CHECK:STDOUT:     %self.patt: %pattern_type.16e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a42]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.b31 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.b31 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.59f = ref_param_pattern [concrete = constants.%self.param_patt.f37]
// CHECK:STDOUT:     %self.patt: %pattern_type.59f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c6e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.993 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.993 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc7_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.59f = ref_param_pattern [concrete = constants.%self.param_patt.f37]
// CHECK:STDOUT:     %self.patt: %pattern_type.59f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c6e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.993 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.993 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc7_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0fa = ref_param_pattern [concrete = constants.%self.param_patt.4dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.0fa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0fc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.2e0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.2e0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc7_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0fa = ref_param_pattern [concrete = constants.%self.param_patt.4dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.0fa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0fc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.2e0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.2e0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f68 = ref_param_pattern [concrete = constants.%self.param_patt.98b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f68 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.868]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.d2f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.d2f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc7_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f68 = ref_param_pattern [concrete = constants.%self.param_patt.98b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f68 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.868]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.d2f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.d2f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c16 = ref_param_pattern [concrete = constants.%self.param_patt.336]
// CHECK:STDOUT:     %self.patt: %pattern_type.c16 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ed]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.bf9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.bf9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc7_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c16 = ref_param_pattern [concrete = constants.%self.param_patt.336]
// CHECK:STDOUT:     %self.patt: %pattern_type.c16 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ed]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.bf9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.bf9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIterate(%s.param: %Slice.5af) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Slice.5af = name_ref s, %s
// CHECK:STDOUT:   %impl.elem2: %.fdd = impl_witness_access constants.%Iterate.impl_witness.a42, element2 [concrete = constants.%Slice.as.Iterate.impl.NewCursor.f43]
// CHECK:STDOUT:   %bound_method.loc7_19.1: <bound method> = bound_method %s.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc7_19.1: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.8af, constants.%custom_witness.f8f19d.1) [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %.loc7_19.1: %facet_type.a1a = converted constants.%i32, %facet_value.loc7_19.1 [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %facet_value.loc7_19.2: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.8af, constants.%custom_witness.f8f19d.1) [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %.loc7_19.2: %facet_type.a1a = converted constants.%i32, %facet_value.loc7_19.2 [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %specific_fn.loc7_19.1: <specific function> = specific_function %impl.elem2, @Slice.as.Iterate.impl.NewCursor(constants.%facet_value.090) [concrete = constants.%Slice.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_19.2: <bound method> = bound_method %s.ref, %specific_fn.loc7_19.1
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.call: init %u64 = call %bound_method.loc7_19.2(%s.ref)
// CHECK:STDOUT:   %var: ref %u64 = var_storage invalid
// CHECK:STDOUT:   assign %var, %Slice.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.231 = impl_witness_access constants.%Iterate.impl_witness.a42, element3 [concrete = constants.%Slice.as.Iterate.impl.Next.c79]
// CHECK:STDOUT:   %bound_method.loc7_19.3: <bound method> = bound_method %s.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc7_19.3: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.8af, constants.%custom_witness.f8f19d.1) [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %.loc7_19.3: %facet_type.a1a = converted constants.%i32, %facet_value.loc7_19.3 [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %facet_value.loc7_19.4: %facet_type.a1a = facet_value constants.%i32, (constants.%Copy.impl_witness.8af, constants.%custom_witness.f8f19d.1) [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %.loc7_19.4: %facet_type.a1a = converted constants.%i32, %facet_value.loc7_19.4 [concrete = constants.%facet_value.090]
// CHECK:STDOUT:   %specific_fn.loc7_19.2: <specific function> = specific_function %impl.elem3, @Slice.as.Iterate.impl.Next(constants.%facet_value.090) [concrete = constants.%Slice.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_19.4: <bound method> = bound_method %s.ref, %specific_fn.loc7_19.2
// CHECK:STDOUT:   %.loc7_19.5: ref %Optional.bf9 = temporary_storage
// CHECK:STDOUT:   %Slice.as.Iterate.impl.Next.call: init %Optional.bf9 to %.loc7_19.5 = call %bound_method.loc7_19.4(%s.ref, %var)
// CHECK:STDOUT:   %.loc7_19.6: ref %Optional.bf9 = temporary %.loc7_19.5, %Slice.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc7_19.7: %Optional.HasValue.type.6ac = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.ae9) [concrete = constants.%Optional.HasValue.e41]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.6ac = name_ref HasValue, %.loc7_19.7 [concrete = constants.%Optional.HasValue.e41]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc7_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.ae9) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_19.5: <bound method> = bound_method %.loc7_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc7_19.8: %Optional.bf9 = acquire_value %.loc7_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc7_19.5(%.loc7_19.8)
// CHECK:STDOUT:   %.loc7_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc7_19.10: bool = converted %Optional.HasValue.call, %.loc7_19.9
// CHECK:STDOUT:   if %.loc7_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc7_19.11: %Optional.Get.type.4fb = specific_constant imports.%Core.import_ref.29e9, @Optional(constants.%OptionalStorage.facet.ae9) [concrete = constants.%Optional.Get.65d]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.4fb = name_ref Get, %.loc7_19.11 [concrete = constants.%Optional.Get.65d]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc7_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.ae9) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_19.6: <bound method> = bound_method %.loc7_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc7_19.12: %Optional.bf9 = acquire_value %.loc7_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %i32 = call %bound_method.loc7_19.6(%.loc7_19.12)
// CHECK:STDOUT:   %.loc7_19.13: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc7_19.14: %i32 = converted %Optional.Get.call, %.loc7_19.13
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %.loc7_19.14
// CHECK:STDOUT:   %sum.ref.loc8: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0.loc8: %.316 = impl_witness_access constants.%AddAssignWith.impl_witness.95a, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.2]
// CHECK:STDOUT:   %bound_method.loc8_9.1: <bound method> = bound_method %sum.ref.loc8, %impl.elem0.loc8
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1]
// CHECK:STDOUT:   %bound_method.loc8_9.2: <bound method> = bound_method %sum.ref.loc8, %specific_fn.loc8
// CHECK:STDOUT:   %.loc8: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = specific_constant imports.%Core.Op.da0, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:   %Op.ref: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = name_ref Op, %.loc8 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref.loc8, %Op.ref
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.2]
// CHECK:STDOUT:   %bound_method.loc8_9.3: <bound method> = bound_method %sum.ref.loc8, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_9.3(%sum.ref.loc8, %x.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.1: <bound method> = bound_method %.loc7_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.1(%.loc7_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.2: <bound method> = bound_method %.loc7_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.2(%.loc7_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_19.1(%self.param: ref %.b31) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_19.1(%self.param: ref %.b31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_19.1(%self.param: ref %.b31) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_19.2(%self.param: ref %MaybeUnformed.993) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_19.2(%self.param: ref %MaybeUnformed.993) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_19.2(%self.param: ref %MaybeUnformed.993) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_19.4(%self.param: ref %struct_type.value.has_value.2e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.1: <bound method> = bound_method %.loc7_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.1(%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_19.2: ref %MaybeUnformed.993 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.2: <bound method> = bound_method %.loc7_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.2(%.loc7_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_19.4(%self.param: ref %struct_type.value.has_value.2e0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_19.4(%self.param: ref %struct_type.value.has_value.2e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_19.5(%self.param: ref %DefaultOptionalStorage.d2f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.93e = name_ref has_value, imports.%Core.import_ref.41b [concrete = imports.%field_decl.06c]
// CHECK:STDOUT:   %.loc7_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.1: <bound method> = bound_method %.loc7_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.1(%.loc7_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.079 = name_ref value, imports.%Core.import_ref.c2a [concrete = imports.%field_decl.7cb]
// CHECK:STDOUT:   %.loc7_19.2: ref %MaybeUnformed.993 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_19.2: <bound method> = bound_method %.loc7_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc7_19.2(%.loc7_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_19.5(%self.param: ref %DefaultOptionalStorage.d2f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_19.5(%self.param: ref %DefaultOptionalStorage.d2f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_19.6(%self.param: ref %Optional.bf9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_19.6(%self.param: ref %Optional.bf9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_19.6(%self.param: ref %Optional.bf9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = parameterized_class_name_type @Float [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %ptr.6fa: type = ptr_type %f64 [concrete]
// CHECK:STDOUT:   %pattern_type.9b4: type = pattern_type %ptr.6fa [concrete]
// CHECK:STDOUT:   %UInt.type: type = parameterized_class_name_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.dd7: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %Slice.type: type = parameterized_class_name_type @Slice [concrete]
// CHECK:STDOUT:   %Slice.generic: %Slice.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Slice.990: type = class_type @Slice, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.elem.0b4: type = unbound_element_type %Slice.990, %u64 [symbolic]
// CHECK:STDOUT:   %Slice.elem.2fb: type = unbound_element_type %Slice.990, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Slice.Make.type.a71: type = fn_type @Slice.Make, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.Make.9a2: %Slice.Make.type.a71 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Slice.71d: type = class_type @Slice, @Slice(%f64) [concrete]
// CHECK:STDOUT:   %pattern_type.608: type = pattern_type %Slice.71d [concrete]
// CHECK:STDOUT:   %Slice.Make.type.970: type = fn_type @Slice.Make, @Slice(%f64) [concrete]
// CHECK:STDOUT:   %Slice.Make.d29: %Slice.Make.type.970 = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.elem.fec: type = unbound_element_type %Slice.71d, %ptr.6fa [concrete]
// CHECK:STDOUT:   %Slice.elem.42a: type = unbound_element_type %Slice.71d, %u64 [concrete]
// CHECK:STDOUT:   %struct_type.ptr.size.8e0: type = struct_type {.ptr: %ptr.6fa, .size: %u64} [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.608 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.608 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.fef: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.d59: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.52f(%T.fef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.324: %T.as_type.as.DefaultOrUnformed.impl.Op.type.d59 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.141: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.132, @Slice.as.UnformedInit.impl(%f64) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet: %UnformedInit.type = facet_value %Slice.71d, (%UnformedInit.impl_witness.141) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.fb0: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.8f5, @T.as_type.as.DefaultOrUnformed.impl.52f(%UnformedInit.facet) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Slice.71d, (%DefaultOrUnformed.impl_witness.fb0) [concrete]
// CHECK:STDOUT:   %Slice.Make.specific_fn.072: <specific function> = specific_function %Slice.Make.d29, @Slice.Make(%f64) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.253: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.7e2: %Slice.as.Copy.impl.Op.type.253 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b0b: <witness> = impl_witness imports.%Copy.impl_witness_table.22d, @Slice.as.Copy.impl(%f64) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.cd9: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%f64) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.091: %Slice.as.Copy.impl.Op.type.cd9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.04c: %Copy.type = facet_value %Slice.71d, (%Copy.impl_witness.b0b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.04f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.04c) [concrete]
// CHECK:STDOUT:   %.eb5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.04f, %Copy.facet.04c [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Slice.as.Copy.impl.Op.091, @Slice.as.Copy.impl.Op(%f64) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.e4d: %pattern_type.9b4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6d0: %pattern_type.9b4 = wrapper_binding_pattern self, %self.param_patt.e4d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.e89: %pattern_type.dd7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.aab: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt.e89 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %struct_type.ptr.size.8e0 [concrete]
// CHECK:STDOUT:   %self.param_patt.d0d: %pattern_type.97e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.df4: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt.d0d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc6_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.346: %pattern_type.608 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d45: %pattern_type.608 = wrapper_binding_pattern self, %self.param_patt.346 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc6_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Slice = %Core.Slice
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.Slice: %Slice.type = import_ref Core//prelude/types/slice, Slice, loaded [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %Core.import_ref.a05: @Slice.%Slice.Make.type (%Slice.Make.type.a71) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Make (constants.%Slice.Make.9a2)]
// CHECK:STDOUT:   %Core.import_ref.78f: @Slice.%Slice.elem.1 (%Slice.elem.2fb) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [concrete = %field_decl.17c]
// CHECK:STDOUT:   %Core.import_ref.ff4: @Slice.%Slice.elem.2 (%Slice.elem.0b4) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [concrete = %field_decl.7e4]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.116: @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.d59) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.324)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.8f5 = impl_witness_table (%Core.import_ref.116), @T.as_type.as.DefaultOrUnformed.impl.52f [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.132 = impl_witness_table (), @Slice.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.7e2: @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op.type (%Slice.as.Copy.impl.Op.type.253) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op (constants.%Slice.as.Copy.impl.Op.7e2)]
// CHECK:STDOUT:   %Copy.impl_witness_table.22d = impl_witness_table (%Core.import_ref.7e2), @Slice.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.f7f = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.7e4: @Slice.%Slice.elem.2 (%Slice.elem.0b4) = field_decl size, element1, %Core.import_ref.f7f in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.c61 = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.17c: @Slice.%Slice.elem.1 (%Slice.elem.2fb) = field_decl ptr, element0, %Core.import_ref.c61 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9b4 = ref_param_pattern [concrete = constants.%self.param_patt.e4d]
// CHECK:STDOUT:     %self.patt: %pattern_type.9b4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6d0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.6fa = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.6fa = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc6_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9b4 = ref_param_pattern [concrete = constants.%self.param_patt.e4d]
// CHECK:STDOUT:     %self.patt: %pattern_type.9b4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6d0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.6fa = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.6fa = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dd7 = ref_param_pattern [concrete = constants.%self.param_patt.e89]
// CHECK:STDOUT:     %self.patt: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.aab]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc6_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dd7 = ref_param_pattern [concrete = constants.%self.param_patt.e89]
// CHECK:STDOUT:     %self.patt: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.aab]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97e = ref_param_pattern [concrete = constants.%self.param_patt.d0d]
// CHECK:STDOUT:     %self.patt: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.df4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.ptr.size.8e0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.ptr.size.8e0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc6_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97e = ref_param_pattern [concrete = constants.%self.param_patt.d0d]
// CHECK:STDOUT:     %self.patt: %pattern_type.97e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.df4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.ptr.size.8e0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.ptr.size.8e0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.608 = ref_param_pattern [concrete = constants.%self.param_patt.346]
// CHECK:STDOUT:     %self.patt: %pattern_type.608 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d45]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Slice.71d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Slice.71d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc6_3.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.608 = ref_param_pattern [concrete = constants.%self.param_patt.346]
// CHECK:STDOUT:     %self.patt: %pattern_type.608 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d45]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Slice.71d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Slice.71d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnformed(%p.param: %ptr.6fa, %n.param: %u64) -> out %return.param: %Slice.71d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %Slice.71d = var_storage %s.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Slice.71d, (constants.%DefaultOrUnformed.impl_witness.fb0) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc6_25.1: %DefaultOrUnformed.type = converted constants.%Slice.71d, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc6_25.1 [concrete = constants.%Slice.71d]
// CHECK:STDOUT:   %.loc6_25.2: type = converted %.loc6_25.1, %as_type [concrete = constants.%Slice.71d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc6_3: ref %Slice.71d = splice_block %s.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %Slice.71d to %.loc6_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %s.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc6_24: type = splice_block %Slice.loc6 [concrete = constants.%Slice.71d] {
// CHECK:STDOUT:     %Core.ref.loc6: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Slice.ref.loc6: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:     %f64.loc6: type = type_literal constants.%f64 [concrete = constants.%f64]
// CHECK:STDOUT:     %Slice.loc6: type = class_type @Slice, @Slice(constants.%f64) [concrete = constants.%Slice.71d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %Slice.71d = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.608 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.608 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc7: ref %Slice.71d = name_ref s, %s
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Slice.ref.loc7: %Slice.type = name_ref Slice, imports.%Core.Slice [concrete = constants.%Slice.generic]
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64 [concrete = constants.%f64]
// CHECK:STDOUT:   %Slice.loc7: type = class_type @Slice, @Slice(constants.%f64) [concrete = constants.%Slice.71d]
// CHECK:STDOUT:   %.loc7_22: %Slice.Make.type.970 = specific_constant imports.%Core.import_ref.a05, @Slice(constants.%f64) [concrete = constants.%Slice.Make.d29]
// CHECK:STDOUT:   %Make.ref: %Slice.Make.type.970 = name_ref Make, %.loc7_22 [concrete = constants.%Slice.Make.d29]
// CHECK:STDOUT:   %p.ref: %ptr.6fa = name_ref p, %p
// CHECK:STDOUT:   %n.ref: %u64 = name_ref n, %n
// CHECK:STDOUT:   %Slice.Make.specific_fn: <specific function> = specific_function %Make.ref, @Slice.Make(constants.%f64) [concrete = constants.%Slice.Make.specific_fn.072]
// CHECK:STDOUT:   %.loc7_3: ref %Slice.71d = splice_block %s.ref.loc7 {}
// CHECK:STDOUT:   %Slice.Make.call: init %Slice.71d to %.loc7_3 = call %Slice.Make.specific_fn(%p.ref, %n.ref)
// CHECK:STDOUT:   assign %s.ref.loc7, %Slice.Make.call
// CHECK:STDOUT:   %s.ref.loc8: ref %Slice.71d = name_ref s, %s
// CHECK:STDOUT:   %.loc8: %Slice.71d = acquire_value %s.ref.loc8
// CHECK:STDOUT:   %impl.elem0: %.eb5 = impl_witness_access constants.%Copy.impl_witness.b0b, element0 [concrete = constants.%Slice.as.Copy.impl.Op.091]
// CHECK:STDOUT:   %bound_method.loc8_10.1: <bound method> = bound_method %.loc8, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Slice.as.Copy.impl.Op(constants.%f64) [concrete = constants.%Slice.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_10.2: <bound method> = bound_method %.loc8, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.call: init %Slice.71d to %.loc4_51.1 = call %bound_method.loc8_10.2(%.loc8)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %s.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%s.var)
// CHECK:STDOUT:   return %Slice.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.1(%self.param: ref %ptr.6fa) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.1(%self.param: ref %ptr.6fa) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.1(%self.param: ref %ptr.6fa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.2(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.2(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.2(%self.param: ref %u64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.3(%self.param: ref %struct_type.ptr.size.8e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_3.1: ref %u64 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.1: <bound method> = bound_method %.loc6_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.1(%.loc6_3.1)
// CHECK:STDOUT:   %.loc6_3.2: ref %ptr.6fa = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.2: <bound method> = bound_method %.loc6_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.2(%.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.3(%self.param: ref %struct_type.ptr.size.8e0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.3(%self.param: ref %struct_type.ptr.size.8e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.4(%self.param: ref %Slice.71d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %size.ref: %Slice.elem.42a = name_ref size, imports.%Core.import_ref.ff4 [concrete = imports.%field_decl.7e4]
// CHECK:STDOUT:   %.loc6_3.1: ref %u64 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.1: <bound method> = bound_method %.loc6_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.1(%.loc6_3.1)
// CHECK:STDOUT:   %ptr.ref: %Slice.elem.fec = name_ref ptr, imports.%Core.import_ref.78f [concrete = imports.%field_decl.17c]
// CHECK:STDOUT:   %.loc6_3.2: ref %ptr.6fa = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.2: <bound method> = bound_method %.loc6_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.2(%.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.4(%self.param: ref %Slice.71d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.4(%self.param: ref %Slice.71d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- range.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Slice.Range.type.e8a: type = fn_type @Slice.Range, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.Range.49d: %Slice.Range.type.e8a = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Range.type.02f: type = fn_type @Slice.Range, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.Range.8ae: %Slice.Range.type.02f = struct_value () [concrete]
// CHECK:STDOUT:   %Slice.Range.specific_fn: <specific function> = specific_function %Slice.Range.8ae, @Slice.Range(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.235: @Slice.%Slice.Range.type (%Slice.Range.type.e8a) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.%Slice.Range (constants.%Slice.Range.49d)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRange(%s.param: %Slice.5af, %start.param: %u64, %count.param: %u64) -> out %return.param: %Slice.5af {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.5af = name_ref s, %s
// CHECK:STDOUT:   %.loc6: %Slice.Range.type.02f = specific_constant imports.%Core.import_ref.235, @Slice(constants.%i32) [concrete = constants.%Slice.Range.8ae]
// CHECK:STDOUT:   %Range.ref: %Slice.Range.type.02f = name_ref Range, %.loc6 [concrete = constants.%Slice.Range.8ae]
// CHECK:STDOUT:   %Slice.Range.bound: <bound method> = bound_method %s.ref, %Range.ref
// CHECK:STDOUT:   %start.ref: %u64 = name_ref start, %start
// CHECK:STDOUT:   %count.ref: %u64 = name_ref count, %count
// CHECK:STDOUT:   %Slice.Range.specific_fn: <specific function> = specific_function %Range.ref, @Slice.Range(constants.%i32) [concrete = constants.%Slice.Range.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.ref, %Slice.Range.specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.Range.call: init %Slice.5af to %.loc4_75.1 = call %bound_method(%s.ref, %start.ref, %count.ref)
// CHECK:STDOUT:   return %Slice.Range.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- keyword.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.253: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.7e2: %Slice.as.Copy.impl.Op.type.253 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.714: <witness> = impl_witness imports.%Copy.impl_witness_table.22d, @Slice.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.type.344: type = fn_type @Slice.as.Copy.impl.Op, @Slice.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.246: %Slice.as.Copy.impl.Op.type.344 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b52: %Copy.type = facet_value %Slice.5af, (%Copy.impl_witness.714) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.950: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b52) [concrete]
// CHECK:STDOUT:   %.d5b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.950, %Copy.facet.b52 [concrete]
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Slice.as.Copy.impl.Op.246, @Slice.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e2: @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op.type (%Slice.as.Copy.impl.Op.type.253) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.Copy.impl.%Slice.as.Copy.impl.Op (constants.%Slice.as.Copy.impl.Op.7e2)]
// CHECK:STDOUT:   %Copy.impl_witness_table.22d = impl_witness_table (%Core.import_ref.7e2), @Slice.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestKeyword(%s.param: %Slice.5af) -> out %return.param: %Slice.5af {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.5af = name_ref s, %s
// CHECK:STDOUT:   %impl.elem0: %.d5b = impl_witness_access constants.%Copy.impl_witness.714, element0 [concrete = constants.%Slice.as.Copy.impl.Op.246]
// CHECK:STDOUT:   %bound_method.loc6_10.1: <bound method> = bound_method %s.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Slice.as.Copy.impl.Op(constants.%i32) [concrete = constants.%Slice.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_10.2: <bound method> = bound_method %s.ref, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Slice.as.Copy.impl.Op.call: init %Slice.5af to %.loc4_43.1 = call %bound_method.loc6_10.2(%s.ref)
// CHECK:STDOUT:   return %Slice.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.373: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1b9: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %Subscript: %ImplicitAs.type.1b9 = symbolic_binding Subscript, 1 [symbolic]
// CHECK:STDOUT:   %Slice.as.IndirectIndexWith.impl.Ref.type.870: type = fn_type @Slice.as.IndirectIndexWith.impl.Ref, @Slice.as.IndirectIndexWith.impl(%T.67d, %Subscript) [symbolic]
// CHECK:STDOUT:   %Slice.as.IndirectIndexWith.impl.Ref.5f9: %Slice.as.IndirectIndexWith.impl.Ref.type.870 = struct_value () [symbolic]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %UInt.f423d4.2: type = class_type @UInt, @UInt(%To.fe9) [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.f423d4.3: type = class_type @UInt, @UInt(%From.fe9) [symbolic]
// CHECK:STDOUT:   %FromUInt.type.f23: type = facet_type <@FromUInt, @FromUInt(%UInt.f423d4.3)> [symbolic]
// CHECK:STDOUT:   %To.e19: %FromUInt.type.f23 = symbolic_binding To, 1 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.2cf: type = fn_type @UInt.as.ImplicitAs.impl.Convert.2, @UInt.as.ImplicitAs.impl.544(%From.fe9, %To.e19) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.807: %UInt.as.ImplicitAs.impl.Convert.type.2cf = struct_value () [symbolic]
// CHECK:STDOUT:   %FromUInt.type.a89: type = facet_type <@FromUInt, @FromUInt(%u64)> [concrete]
// CHECK:STDOUT:   %facet_type.990: type = facet_type <@IntFitsIn, @IntFitsIn(%UInt.f423d4.2) & @AnyUInt> [symbolic]
// CHECK:STDOUT:   %From.26a: %facet_type.990 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %UInt.as.FromUInt.impl.Convert.type.89b: type = fn_type @UInt.as.FromUInt.impl.Convert, @UInt.as.FromUInt.impl(%To.fe9, %From.26a) [symbolic]
// CHECK:STDOUT:   %UInt.as.FromUInt.impl.Convert.e69: %UInt.as.FromUInt.impl.Convert.type.89b = struct_value () [symbolic]
// CHECK:STDOUT:   %UInt.as.AnyUInt.impl.AsUInt.type.32d: type = fn_type @UInt.as.AnyUInt.impl.AsUInt, @UInt.as.AnyUInt.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.AnyUInt.impl.AsUInt.523: %UInt.as.AnyUInt.impl.AsUInt.type.32d = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.568: type = facet_type <@IntFitsIn, @IntFitsIn(%u64) & @AnyUInt> [concrete]
// CHECK:STDOUT:   %custom_witness: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%u64) [concrete]
// CHECK:STDOUT:   %AnyUInt.impl_witness.2b1: <witness> = impl_witness imports.%AnyUInt.impl_witness_table, @UInt.as.AnyUInt.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.568 = facet_value %u64, (%custom_witness, %AnyUInt.impl_witness.2b1) [concrete]
// CHECK:STDOUT:   %FromUInt.impl_witness.431: <witness> = impl_witness imports.%FromUInt.impl_witness_table.fa5, @UInt.as.FromUInt.impl(%int_64, %facet_value) [concrete]
// CHECK:STDOUT:   %FromUInt.facet: %FromUInt.type.a89 = facet_value %u64, (%FromUInt.impl_witness.431) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.859: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.803, @UInt.as.ImplicitAs.impl.544(%int_64, %FromUInt.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.1b9 = facet_value %u64, (%ImplicitAs.impl_witness.859) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness.306: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table, @Slice.as.IndirectIndexWith.impl(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Slice.as.IndirectIndexWith.impl.Ref.type.977: type = fn_type @Slice.as.IndirectIndexWith.impl.Ref, @Slice.as.IndirectIndexWith.impl(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Slice.as.IndirectIndexWith.impl.Ref.9a7: %Slice.as.IndirectIndexWith.impl.Ref.type.977 = struct_value () [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.373 = facet_value %Slice.5af, (%IndirectIndexWith.impl_witness.306) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.d29: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%u64, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.2da: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.d29, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %Slice.as.IndirectIndexWith.impl.Ref.specific_fn: <specific function> = specific_function %Slice.as.IndirectIndexWith.impl.Ref.9a7, @Slice.as.IndirectIndexWith.impl.Ref(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b3 = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3f1: @Slice.as.IndirectIndexWith.impl.%Slice.as.IndirectIndexWith.impl.Ref.type (%Slice.as.IndirectIndexWith.impl.Ref.type.870) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @Slice.as.IndirectIndexWith.impl.%Slice.as.IndirectIndexWith.impl.Ref (constants.%Slice.as.IndirectIndexWith.impl.Ref.5f9)]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.0b3, %Core.import_ref.3f1), @Slice.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b70: @UInt.as.ImplicitAs.impl.544.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.2cf) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.544.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.807)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.803 = impl_witness_table (%Core.import_ref.b70), @UInt.as.ImplicitAs.impl.544 [concrete]
// CHECK:STDOUT:   %Core.import_ref.dad: @UInt.as.FromUInt.impl.%UInt.as.FromUInt.impl.Convert.type (%UInt.as.FromUInt.impl.Convert.type.89b) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.FromUInt.impl.%UInt.as.FromUInt.impl.Convert (constants.%UInt.as.FromUInt.impl.Convert.e69)]
// CHECK:STDOUT:   %FromUInt.impl_witness_table.fa5 = impl_witness_table (%Core.import_ref.dad), @UInt.as.FromUInt.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.AnyUInt.impl.%N (constants.%N)]
// CHECK:STDOUT:   %Core.import_ref.76e: @UInt.as.AnyUInt.impl.%UInt.as.AnyUInt.impl.AsUInt.type (%UInt.as.AnyUInt.impl.AsUInt.type.32d) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.AnyUInt.impl.%UInt.as.AnyUInt.impl.AsUInt (constants.%UInt.as.AnyUInt.impl.AsUInt.523)]
// CHECK:STDOUT:   %AnyUInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.76e), @UInt.as.AnyUInt.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIndex(%s.param: %Slice.5af, %i.param: %u64) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Slice.5af = name_ref s, %s
// CHECK:STDOUT:   %i.ref: %u64 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem1: %.2da = impl_witness_access constants.%IndirectIndexWith.impl_witness.306, element1 [concrete = constants.%Slice.as.IndirectIndexWith.impl.Ref.9a7]
// CHECK:STDOUT:   %bound_method.loc6_13.1: <bound method> = bound_method %s.ref, %impl.elem1
// CHECK:STDOUT:   %ImplicitAs.facet.loc6_13.1: %ImplicitAs.type.1b9 = facet_value constants.%u64, (constants.%ImplicitAs.impl_witness.859) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc6_13.1: %ImplicitAs.type.1b9 = converted constants.%u64, %ImplicitAs.facet.loc6_13.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc6_13.2: %ImplicitAs.type.1b9 = facet_value constants.%u64, (constants.%ImplicitAs.impl_witness.859) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc6_13.2: %ImplicitAs.type.1b9 = converted constants.%u64, %ImplicitAs.facet.loc6_13.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn.loc6_13.1: <specific function> = specific_function %impl.elem1, @Slice.as.IndirectIndexWith.impl.Ref(constants.%i32, constants.%ImplicitAs.facet) [concrete = constants.%Slice.as.IndirectIndexWith.impl.Ref.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_13.2: <bound method> = bound_method %s.ref, %specific_fn.loc6_13.1
// CHECK:STDOUT:   %Slice.as.IndirectIndexWith.impl.Ref.call: ref %i32 = call %bound_method.loc6_13.2(%s.ref, %i.ref)
// CHECK:STDOUT:   %.loc6_13.3: %i32 = acquire_value %Slice.as.IndirectIndexWith.impl.Ref.call
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc6_13.3: <bound method> = bound_method %.loc6_13.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn.loc6_13.2: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_13.4: <bound method> = bound_method %.loc6_13.3, %specific_fn.loc6_13.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_13.4(%.loc6_13.3)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- array_to_slice_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Slice.990: type = class_type @Slice, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.elem.0b4: type = unbound_element_type %Slice.990, %u64 [symbolic]
// CHECK:STDOUT:   %Slice.elem.2fb: type = unbound_element_type %Slice.990, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.dd7: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.fcd: type = pattern_type %Slice.5af [concrete]
// CHECK:STDOUT:   %TakesSlice.type: type = fn_type @TakesSlice [concrete]
// CHECK:STDOUT:   %TakesSlice: %TakesSlice.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %N.d23: Core.IntLiteral = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Array.91b: type = class_type @Array, @Array(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %ptr.673: type = ptr_type %Array.91b [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Slice.elem.232: type = unbound_element_type %Slice.5af, %ptr.a17 [concrete]
// CHECK:STDOUT:   %Slice.elem.f61: type = unbound_element_type %Slice.5af, %u64 [concrete]
// CHECK:STDOUT:   %struct_type.ptr.size.e24: type = struct_type {.ptr: %ptr.a17, .size: %u64} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2f3: type = facet_type <@ImplicitAs, @ImplicitAs(%Slice.5af)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.31b: type = fn_type @ptr.as.ImplicitAs.impl.Convert.2, @ptr.as.ImplicitAs.impl.c61(%T.67d, %N.d23) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.8f8: %ptr.as.ImplicitAs.impl.Convert.type.31b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.026, @ptr.as.ImplicitAs.impl.c61(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.f77: type = fn_type @ptr.as.ImplicitAs.impl.Convert.2, @ptr.as.ImplicitAs.impl.c61(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.e8d: %ptr.as.ImplicitAs.impl.Convert.type.f77 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2f3 = facet_value %ptr.673, (%ImplicitAs.impl_witness.a39) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.140: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Slice.5af, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.e4c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.140, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.e8d, @ptr.as.ImplicitAs.impl.Convert.2(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %self.param_patt.6cc: %pattern_type.7dc = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a12: %pattern_type.7dc = wrapper_binding_pattern self, %self.param_patt.6cc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_21.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc8_21.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_21.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.e89: %pattern_type.dd7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.aab: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt.e89 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.71f: type = pattern_type %struct_type.ptr.size.e24 [concrete]
// CHECK:STDOUT:   %self.param_patt.fd9: %pattern_type.71f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.280: %pattern_type.71f = wrapper_binding_pattern self, %self.param_patt.fd9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_21.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc8_21.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.924: %pattern_type.fcd = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.049: %pattern_type.fcd = wrapper_binding_pattern self, %self.param_patt.924 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_21.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc8_21.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_21.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.78f: @Slice.%Slice.elem.1 (%Slice.elem.2fb) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [concrete = %field_decl.17c]
// CHECK:STDOUT:   %Core.import_ref.ff4: @Slice.%Slice.elem.2 (%Slice.elem.0b4) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [concrete = %field_decl.7e4]
// CHECK:STDOUT:   %Core.import_ref.f60: @ptr.as.ImplicitAs.impl.c61.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.31b) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.c61.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.8f8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.026 = impl_witness_table (%Core.import_ref.f60), @ptr.as.ImplicitAs.impl.c61 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f7f = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.7e4: @Slice.%Slice.elem.2 (%Slice.elem.0b4) = field_decl size, element1, %Core.import_ref.f7f in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.c61 = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.17c: @Slice.%Slice.elem.1 (%Slice.elem.2fb) = field_decl ptr, element0, %Core.import_ref.c61 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_21.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7dc = ref_param_pattern [concrete = constants.%self.param_patt.6cc]
// CHECK:STDOUT:     %self.patt: %pattern_type.7dc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a12]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.a17 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.a17 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_21.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7dc = ref_param_pattern [concrete = constants.%self.param_patt.6cc]
// CHECK:STDOUT:     %self.patt: %pattern_type.7dc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a12]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.a17 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.a17 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_21.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dd7 = ref_param_pattern [concrete = constants.%self.param_patt.e89]
// CHECK:STDOUT:     %self.patt: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.aab]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_21.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dd7 = ref_param_pattern [concrete = constants.%self.param_patt.e89]
// CHECK:STDOUT:     %self.patt: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.aab]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_21.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.71f = ref_param_pattern [concrete = constants.%self.param_patt.fd9]
// CHECK:STDOUT:     %self.patt: %pattern_type.71f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.280]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.ptr.size.e24 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.ptr.size.e24 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc8_21.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.71f = ref_param_pattern [concrete = constants.%self.param_patt.fd9]
// CHECK:STDOUT:     %self.patt: %pattern_type.71f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.280]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.ptr.size.e24 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.ptr.size.e24 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_21.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fcd = ref_param_pattern [concrete = constants.%self.param_patt.924]
// CHECK:STDOUT:     %self.patt: %pattern_type.fcd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.049]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Slice.5af = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Slice.5af = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc8_21.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fcd = ref_param_pattern [concrete = constants.%self.param_patt.924]
// CHECK:STDOUT:     %self.patt: %pattern_type.fcd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.049]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Slice.5af = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Slice.5af = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithArrayPointer(%a.param: %ptr.673) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakesSlice.ref: %TakesSlice.type = name_ref TakesSlice, file.%TakesSlice.decl [concrete = constants.%TakesSlice]
// CHECK:STDOUT:   %a.ref: %ptr.673 = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0: %.e4c = impl_witness_access constants.%ImplicitAs.impl_witness.a39, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.e8d]
// CHECK:STDOUT:   %bound_method.loc8_21.1: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.2(constants.%i32, constants.%int_5.64b) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_21.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc8_21.1: ref %Slice.5af = temporary_storage
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %Slice.5af to %.loc8_21.1 = call %bound_method.loc8_21.2(%a.ref)
// CHECK:STDOUT:   %.loc8_21.2: init %Slice.5af = converted %a.ref, %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc8_21.3: ref %Slice.5af = temporary %.loc8_21.1, %.loc8_21.2
// CHECK:STDOUT:   %.loc8_21.4: %Slice.5af = acquire_value %.loc8_21.3
// CHECK:STDOUT:   %TakesSlice.call: init %i32 = call %TakesSlice.ref(%.loc8_21.4)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc8_21.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc8_21.3)
// CHECK:STDOUT:   return %TakesSlice.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_21.1(%self.param: ref %ptr.a17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_21.1(%self.param: ref %ptr.a17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_21.1(%self.param: ref %ptr.a17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_21.2(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_21.2(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_21.2(%self.param: ref %u64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_21.3(%self.param: ref %struct_type.ptr.size.e24) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_21.1: ref %u64 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8_21.1: <bound method> = bound_method %.loc8_21.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8_21.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8_21.1(%.loc8_21.1)
// CHECK:STDOUT:   %.loc8_21.2: ref %ptr.a17 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8_21.2: <bound method> = bound_method %.loc8_21.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8_21.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8_21.2(%.loc8_21.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_21.3(%self.param: ref %struct_type.ptr.size.e24) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_21.3(%self.param: ref %struct_type.ptr.size.e24) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_21.4(%self.param: ref %Slice.5af) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %size.ref: %Slice.elem.f61 = name_ref size, imports.%Core.import_ref.ff4 [concrete = imports.%field_decl.7e4]
// CHECK:STDOUT:   %.loc8_21.1: ref %u64 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8_21.1: <bound method> = bound_method %.loc8_21.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8_21.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8_21.1(%.loc8_21.1)
// CHECK:STDOUT:   %ptr.ref: %Slice.elem.232 = name_ref ptr, imports.%Core.import_ref.78f [concrete = imports.%field_decl.17c]
// CHECK:STDOUT:   %.loc8_21.2: ref %ptr.a17 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8_21.2: <bound method> = bound_method %.loc8_21.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8_21.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8_21.2(%.loc8_21.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_21.4(%self.param: ref %Slice.5af) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_21.4(%self.param: ref %Slice.5af) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- array_to_slice_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Slice.990: type = class_type @Slice, @Slice(%T.67d) [symbolic]
// CHECK:STDOUT:   %Slice.elem.0b4: type = unbound_element_type %Slice.990, %u64 [symbolic]
// CHECK:STDOUT:   %Slice.elem.2fb: type = unbound_element_type %Slice.990, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.dd7: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Slice.5af: type = class_type @Slice, @Slice(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.fcd: type = pattern_type %Slice.5af [concrete]
// CHECK:STDOUT:   %TakesSlice.type: type = fn_type @TakesSlice [concrete]
// CHECK:STDOUT:   %TakesSlice: %TakesSlice.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %N.d23: Core.IntLiteral = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Array.91b: type = class_type @Array, @Array(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %ptr.673: type = ptr_type %Array.91b [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Slice.elem.232: type = unbound_element_type %Slice.5af, %ptr.a17 [concrete]
// CHECK:STDOUT:   %Slice.elem.f61: type = unbound_element_type %Slice.5af, %u64 [concrete]
// CHECK:STDOUT:   %struct_type.ptr.size.e24: type = struct_type {.ptr: %ptr.a17, .size: %u64} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2f3: type = facet_type <@ImplicitAs, @ImplicitAs(%Slice.5af)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.31b: type = fn_type @ptr.as.ImplicitAs.impl.Convert.2, @ptr.as.ImplicitAs.impl.c61(%T.67d, %N.d23) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.8f8: %ptr.as.ImplicitAs.impl.Convert.type.31b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.026, @ptr.as.ImplicitAs.impl.c61(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.f77: type = fn_type @ptr.as.ImplicitAs.impl.Convert.2, @ptr.as.ImplicitAs.impl.c61(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.e8d: %ptr.as.ImplicitAs.impl.Convert.type.f77 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ad8: %ImplicitAs.type.2f3 = facet_value %ptr.673, (%ImplicitAs.impl_witness.a39) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.140: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Slice.5af, %ImplicitAs.facet.ad8) [concrete]
// CHECK:STDOUT:   %.e4c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.140, %ImplicitAs.facet.ad8 [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.e8d, @ptr.as.ImplicitAs.impl.Convert.2(%i32, %int_5.64b) [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %self.param_patt.6cc: %pattern_type.7dc = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a12: %pattern_type.7dc = wrapper_binding_pattern self, %self.param_patt.6cc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_21.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_21.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_21.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.e89: %pattern_type.dd7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.aab: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt.e89 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.71f: type = pattern_type %struct_type.ptr.size.e24 [concrete]
// CHECK:STDOUT:   %self.param_patt.fd9: %pattern_type.71f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.280: %pattern_type.71f = wrapper_binding_pattern self, %self.param_patt.fd9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_21.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_21.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.924: %pattern_type.fcd = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.049: %pattern_type.fcd = wrapper_binding_pattern self, %self.param_patt.924 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_21.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9_21.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_21.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.78f: @Slice.%Slice.elem.1 (%Slice.elem.2fb) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [concrete = %field_decl.17c]
// CHECK:STDOUT:   %Core.import_ref.ff4: @Slice.%Slice.elem.2 (%Slice.elem.0b4) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [concrete = %field_decl.7e4]
// CHECK:STDOUT:   %Core.import_ref.f60: @ptr.as.ImplicitAs.impl.c61.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.31b) = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.c61.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.8f8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.026 = impl_witness_table (%Core.import_ref.f60), @ptr.as.ImplicitAs.impl.c61 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f7f = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.7e4: @Slice.%Slice.elem.2 (%Slice.elem.0b4) = field_decl size, element1, %Core.import_ref.f7f in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.c61 = import_ref Core//prelude/types/slice, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.17c: @Slice.%Slice.elem.1 (%Slice.elem.2fb) = field_decl ptr, element0, %Core.import_ref.c61 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_21.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7dc = ref_param_pattern [concrete = constants.%self.param_patt.6cc]
// CHECK:STDOUT:     %self.patt: %pattern_type.7dc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a12]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.a17 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.a17 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_21.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7dc = ref_param_pattern [concrete = constants.%self.param_patt.6cc]
// CHECK:STDOUT:     %self.patt: %pattern_type.7dc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a12]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.a17 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.a17 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_21.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dd7 = ref_param_pattern [concrete = constants.%self.param_patt.e89]
// CHECK:STDOUT:     %self.patt: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.aab]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_21.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dd7 = ref_param_pattern [concrete = constants.%self.param_patt.e89]
// CHECK:STDOUT:     %self.patt: %pattern_type.dd7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.aab]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u64 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u64 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_21.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.71f = ref_param_pattern [concrete = constants.%self.param_patt.fd9]
// CHECK:STDOUT:     %self.patt: %pattern_type.71f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.280]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.ptr.size.e24 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.ptr.size.e24 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_21.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.71f = ref_param_pattern [concrete = constants.%self.param_patt.fd9]
// CHECK:STDOUT:     %self.patt: %pattern_type.71f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.280]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.ptr.size.e24 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.ptr.size.e24 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_21.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fcd = ref_param_pattern [concrete = constants.%self.param_patt.924]
// CHECK:STDOUT:     %self.patt: %pattern_type.fcd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.049]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Slice.5af = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Slice.5af = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc9_21.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fcd = ref_param_pattern [concrete = constants.%self.param_patt.924]
// CHECK:STDOUT:     %self.patt: %pattern_type.fcd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.049]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Slice.5af = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Slice.5af = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddressOfLocalVariable() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %TakesSlice.ref: %TakesSlice.type = name_ref TakesSlice, file.%TakesSlice.decl [concrete = constants.%TakesSlice]
// CHECK:STDOUT:   %b.ref: ref %Array.91b = name_ref b, %b
// CHECK:STDOUT:   %addr: %ptr.673 = addr_of %b.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.e4c = impl_witness_access constants.%ImplicitAs.impl_witness.a39, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.e8d]
// CHECK:STDOUT:   %bound_method.loc9_21.1: <bound method> = bound_method %addr, %impl.elem0.loc9
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @ptr.as.ImplicitAs.impl.Convert.2(constants.%i32, constants.%int_5.64b) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_21.2: <bound method> = bound_method %addr, %specific_fn.loc9
// CHECK:STDOUT:   %.loc9_21.1: ref %Slice.5af = temporary_storage
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %Slice.5af to %.loc9_21.1 = call %bound_method.loc9_21.2(%addr)
// CHECK:STDOUT:   %.loc9_21.2: init %Slice.5af = converted %addr, %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc9_21.3: ref %Slice.5af = temporary %.loc9_21.1, %.loc9_21.2
// CHECK:STDOUT:   %.loc9_21.4: %Slice.5af = acquire_value %.loc9_21.3
// CHECK:STDOUT:   %TakesSlice.call: init %i32 = call %TakesSlice.ref(%.loc9_21.4)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9: <bound method> = bound_method %.loc9_21.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9(%.loc9_21.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return %TakesSlice.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_21.1(%self.param: ref %ptr.a17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_21.1(%self.param: ref %ptr.a17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_21.1(%self.param: ref %ptr.a17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_21.2(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_21.2(%self.param: ref %u64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_21.2(%self.param: ref %u64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_21.3(%self.param: ref %struct_type.ptr.size.e24) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_21.1: ref %u64 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_21.1: <bound method> = bound_method %.loc9_21.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_21.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_21.1(%.loc9_21.1)
// CHECK:STDOUT:   %.loc9_21.2: ref %ptr.a17 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_21.2: <bound method> = bound_method %.loc9_21.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_21.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_21.2(%.loc9_21.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_21.3(%self.param: ref %struct_type.ptr.size.e24) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_21.3(%self.param: ref %struct_type.ptr.size.e24) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_21.4(%self.param: ref %Slice.5af) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %size.ref: %Slice.elem.f61 = name_ref size, imports.%Core.import_ref.ff4 [concrete = imports.%field_decl.7e4]
// CHECK:STDOUT:   %.loc9_21.1: ref %u64 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_21.1: <bound method> = bound_method %.loc9_21.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_21.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_21.1(%.loc9_21.1)
// CHECK:STDOUT:   %ptr.ref: %Slice.elem.232 = name_ref ptr, imports.%Core.import_ref.78f [concrete = imports.%field_decl.17c]
// CHECK:STDOUT:   %.loc9_21.2: ref %ptr.a17 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_21.2: <bound method> = bound_method %.loc9_21.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_21.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_21.2(%.loc9_21.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_21.4(%self.param: ref %Slice.5af) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_21.4(%self.param: ref %Slice.5af) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

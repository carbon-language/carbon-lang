// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/for.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/for/pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/for/pattern.carbon

// --- empty_range.carbon

library "[[@TEST_NAME]]";

class EmptyRange(T:! Core.Copy) {
  fn Make() -> Self { return {}; }

  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor[self: Self]() -> {} {
      return {};
    }
    fn Next[self: Self](cursor: {}*) -> Core.Optional(T) {
      return Core.Optional(T).None();
    }
  }
}

class C {
  impl as Core.Copy {
    fn Op[self: Self]() -> Self { return {}; }
  }
}

// --- value.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(c: C);

fn Run() {
  //@dump-sem-ir-begin
  for (c: C in EmptyRange(C).Make()) {
    Body(c);
  }
  //@dump-sem-ir-end
}

// --- var.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(c: C*);

fn Run() {
  //@dump-sem-ir-begin
  for (var c: C in EmptyRange(C).Make()) {
    Body(&c);
  }
  //@dump-sem-ir-end
}

// --- tuple.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(a: bool, b: bool);

fn Run() {
  //@dump-sem-ir-begin
  for ((a: bool, b: bool) in EmptyRange((bool, bool)).Make()) {
    Body(a, b);
  }
  //@dump-sem-ir-end
}

// --- tuple_class.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(a: C, b: C);

fn Run() {
  //@dump-sem-ir-begin
  for ((a: C, b: C) in EmptyRange((C, C)).Make()) {
    Body(a, b);
  }
  //@dump-sem-ir-end
}

// --- fail_bad_pattern.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

class X {}

fn Run() {
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+7]]:7: error: cannot implicitly convert expression of type `C` to `X` [ConversionFailure]
  // CHECK:STDERR:   for (x: X in EmptyRange(C).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+4]]:7: note: type `C` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   for (x: X in EmptyRange(C).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (x: X in EmptyRange(C).Make()) {
  }
}

// CHECK:STDOUT: --- value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8cf: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.261: %EmptyRange.Make.type.8cf = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f5b: <witness> = impl_witness imports.%Copy.impl_witness_table.d1a [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.f5b) [concrete]
// CHECK:STDOUT:   %EmptyRange.ef7: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.5b8: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.3f3: %EmptyRange.Make.type.5b8 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.3f3, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.80d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.5d9: %EmptyRange.as.Iterate.impl.Next.type.80d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a35: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.73f: %EmptyRange.as.Iterate.impl.NewCursor.type.a35 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.108: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.622: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.a31: %EmptyRange.as.Iterate.impl.NewCursor.type.622 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.01b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.e9e: %EmptyRange.as.Iterate.impl.Next.type.01b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.ef7, (%Iterate.impl_witness.108) [concrete]
// CHECK:STDOUT:   %.1b8: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.a31, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.953: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.6b9: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.e9e, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.4f0: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.46b: %Optional.HasValue.type.4f0 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.09f: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.901: %Optional.Get.type.09f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.46b, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.901, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.150: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.727: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.150) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.8fa: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.727 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.d5b: %type_where = facet_value %Optional.6b9, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.2c4: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d5b) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.e53: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.2c4 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.431: %type_where = facet_value %EmptyRange.ef7, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.842: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.431) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.751: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.842 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.6c6: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8cf) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.261)]
// CHECK:STDOUT:   %Main.import_ref.ba9: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d1a = impl_witness_table (%Main.import_ref.ba9), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e02: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.a35) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.73f)]
// CHECK:STDOUT:   %Main.import_ref.4c1: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.80d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.5d9)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.e02, %Main.import_ref.4c1), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.81c: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.5fa: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.f5b) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.ef7]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.5b8 = specific_constant imports.%Main.import_ref.6c6, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.3f3]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.5b8 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.3f3]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.ef7 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.ef7 = call %EmptyRange.Make.specific_fn() to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.ef7 = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.1b8 = impl_witness_access constants.%Iterate.impl_witness.108, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.a31]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.ef7 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.953 = impl_witness_access constants.%Iterate.impl_witness.108, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.e9e]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.6b9 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.ef7 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.6b9 = call %bound_method.loc10_36.4(%.loc10_35.4, %addr) to %.loc10_36.1
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.6b9 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.4f0 = specific_constant imports.%Main.import_ref.81c, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.46b]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.4f0 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.46b]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.6b9 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.09f = specific_constant imports.%Main.import_ref.5fa, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.901]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.09f = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.901]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.6b9 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc10_36.6(%.loc10_36.9) to %.loc10_36.8
// CHECK:STDOUT:   %C.ref.loc10_11: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_36.10: ref %C = temporary %.loc10_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc10_36.11: %C = acquire_value %.loc10_36.10
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc10_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.8fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.7: <bound method> = bound_method %.loc10_36.10, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_36.1: init %empty_tuple.type = call %bound_method.loc10_36.7(%.loc10_36.10)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.e53
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.8: <bound method> = bound_method %.loc10_36.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_36.2: init %empty_tuple.type = call %bound_method.loc10_36.8(%.loc10_36.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_36.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.9: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_36.3: init %empty_tuple.type = call %bound_method.loc10_36.9(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.751
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_35.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_35: init %empty_tuple.type = call %bound_method.loc10_35(%.loc10_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8cf: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.261: %EmptyRange.Make.type.8cf = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f5b: <witness> = impl_witness imports.%Copy.impl_witness_table.d1a [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.f5b) [concrete]
// CHECK:STDOUT:   %EmptyRange.ef7: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.5b8: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.3f3: %EmptyRange.Make.type.5b8 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.3f3, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.80d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.5d9: %EmptyRange.as.Iterate.impl.Next.type.80d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a35: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.73f: %EmptyRange.as.Iterate.impl.NewCursor.type.a35 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.108: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.622: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.a31: %EmptyRange.as.Iterate.impl.NewCursor.type.622 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.01b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.e9e: %EmptyRange.as.Iterate.impl.Next.type.01b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.ef7, (%Iterate.impl_witness.108) [concrete]
// CHECK:STDOUT:   %.1b8: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.a31, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.953: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.6b9: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.e9e, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.4f0: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.46b: %Optional.HasValue.type.4f0 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.09f: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.901: %Optional.Get.type.09f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.46b, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.901, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.150: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.727: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.150) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.8fa: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.727 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.d5b: %type_where = facet_value %Optional.6b9, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.2c4: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d5b) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.e53: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.2c4 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.431: %type_where = facet_value %EmptyRange.ef7, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.842: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.431) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.751: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.842 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.6c6: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8cf) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.261)]
// CHECK:STDOUT:   %Main.import_ref.ba9: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d1a = impl_witness_table (%Main.import_ref.ba9), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e02: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.a35) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.73f)]
// CHECK:STDOUT:   %Main.import_ref.4c1: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.80d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.5d9)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.e02, %Main.import_ref.4c1), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.81c: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.5fa: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.f5b) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.ef7]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.5b8 = specific_constant imports.%Main.import_ref.6c6, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.3f3]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.5b8 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.3f3]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.ef7 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.ef7 = call %EmptyRange.Make.specific_fn() to %.loc10_39.1
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.ef7 = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.1b8 = impl_witness_access constants.%Iterate.impl_witness.108, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.a31]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.ef7 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.953 = impl_witness_access constants.%Iterate.impl_witness.108, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.e9e]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.6b9 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.ef7 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.6b9 = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10) to %.loc10_40.1
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.6b9 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.4f0 = specific_constant imports.%Main.import_ref.81c, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.46b]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.4f0 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.46b]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.6b9 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.09f = specific_constant imports.%Main.import_ref.5fa, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.901]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.09f = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.901]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.6b9 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc10_40.6(%.loc10_40.8) to %.loc10_8
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc11: %ptr.31e = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc11)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_8: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.8fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_8: init %empty_tuple.type = call %bound_method.loc10_8(%c.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.e53
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_40.7: <bound method> = bound_method %.loc10_40.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_40.1: init %empty_tuple.type = call %bound_method.loc10_40.7(%.loc10_40.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_40.2: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_40.8: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_40.2: init %empty_tuple.type = call %bound_method.loc10_40.8(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.751
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_39: <bound method> = bound_method %.loc10_39.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_39: init %empty_tuple.type = call %bound_method.loc10_39(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.784: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %tuple.type.784 [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8cf: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.261: %EmptyRange.Make.type.8cf = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.9fa: %tuple.type.24b = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.165: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.66d(%T.04d, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.529: %tuple.type.as.Copy.impl.Op.type.165 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.509: <witness> = impl_witness imports.%Copy.impl_witness_table.35d [concrete]
// CHECK:STDOUT:   %Copy.facet.c82: %Copy.type = facet_value bool, (%Copy.impl_witness.509) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.52e: <witness> = impl_witness imports.%Copy.impl_witness_table.5ef, @tuple.type.as.Copy.impl.66d(%Copy.facet.c82, %Copy.facet.c82) [concrete]
// CHECK:STDOUT:   %Copy.facet.f7a: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.52e) [concrete]
// CHECK:STDOUT:   %EmptyRange.bbc: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.d21: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.84f: %EmptyRange.Make.type.d21 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.84f, @EmptyRange.Make(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.80d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.5d9: %EmptyRange.as.Iterate.impl.Next.type.80d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a35: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.73f: %EmptyRange.as.Iterate.impl.NewCursor.type.a35 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.6e5: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.ba1: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.8a2: %EmptyRange.as.Iterate.impl.NewCursor.type.ba1 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.27a: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f70: %EmptyRange.as.Iterate.impl.Next.type.27a = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.bbc, (%Iterate.impl_witness.6e5) [concrete]
// CHECK:STDOUT:   %.e0b: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.8a2, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.7a3: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.119: type = class_type @Optional, @Optional(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.f70, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.fd0: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c98: %Optional.HasValue.type.fd0 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.798: type = fn_type @Optional.Get, @Optional(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %Optional.Get.8d1: %Optional.Get.type.798 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.c98, @Optional.HasValue(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.8d1, @Optional.Get(%Copy.facet.f7a) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.d7c: %type_where = facet_value %tuple.type.784, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5f5: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d7c) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.149: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5f5 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.037: %type_where = facet_value %Optional.119, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.818: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.037) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.6b4: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.818 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.d13: %type_where = facet_value %EmptyRange.bbc, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.829: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d13) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.6c8: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.829 = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6c6: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8cf) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.261)]
// CHECK:STDOUT:   %Core.import_ref.055: @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.165) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.529)]
// CHECK:STDOUT:   %Copy.impl_witness_table.5ef = impl_witness_table (%Core.import_ref.055), @tuple.type.as.Copy.impl.66d [concrete]
// CHECK:STDOUT:   %Core.import_ref.a53: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.35d = impl_witness_table (%Core.import_ref.a53), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e02: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.a35) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.73f)]
// CHECK:STDOUT:   %Main.import_ref.4c1: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.80d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.5d9)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.e02, %Main.import_ref.4c1), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.81c: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.5fa: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_25: %pattern_type.860 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Bool.call.loc10_42: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %Bool.call.loc10_48: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc10_52: %tuple.type.24b = tuple_literal (%Bool.call.loc10_42, %Bool.call.loc10_48) [concrete = constants.%tuple.9fa]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.52e) [concrete = constants.%Copy.facet.f7a]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.f7a]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.f7a) [concrete = constants.%EmptyRange.bbc]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.d21 = specific_constant imports.%Main.import_ref.6c6, @EmptyRange(constants.%Copy.facet.f7a) [concrete = constants.%EmptyRange.Make.84f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.d21 = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.84f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.f7a) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.bbc = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.bbc = call %EmptyRange.Make.specific_fn() to %.loc10_60.1
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.bbc = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.e0b = impl_witness_access constants.%Iterate.impl_witness.6e5, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.8a2]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.f7a) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.bbc = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.7a3 = impl_witness_access constants.%Iterate.impl_witness.6e5, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.f70]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.f7a) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.119 = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.bbc = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.119 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr) to %.loc10_61.1
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.119 = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.fd0 = specific_constant imports.%Main.import_ref.81c, @Optional(constants.%Copy.facet.f7a) [concrete = constants.%Optional.HasValue.c98]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.fd0 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.c98]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.f7a) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.119 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.798 = specific_constant imports.%Main.import_ref.5fa, @Optional(constants.%Copy.facet.f7a) [concrete = constants.%Optional.Get.8d1]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.798 = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.8d1]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.f7a) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.119 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.784 = call %bound_method.loc10_61.6(%.loc10_61.9) to %.loc10_61.8
// CHECK:STDOUT:   %.loc10_61.10: ref %tuple.type.784 = temporary %.loc10_61.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %.loc10_61.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %.loc10_61.10, element1
// CHECK:STDOUT:   %.loc10_12.1: type = splice_block %.loc10_12.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc10_12: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_12.2: type = value_of_initializer %Bool.call.loc10_12 [concrete = bool]
// CHECK:STDOUT:     %.loc10_12.3: type = converted %Bool.call.loc10_12, %.loc10_12.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_61.11: bool = acquire_value %tuple.elem0
// CHECK:STDOUT:   %a: bool = value_binding a, %.loc10_61.11
// CHECK:STDOUT:   %.loc10_21.1: type = splice_block %.loc10_21.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc10_21: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_21.2: type = value_of_initializer %Bool.call.loc10_21 [concrete = bool]
// CHECK:STDOUT:     %.loc10_21.3: type = converted %Bool.call.loc10_21, %.loc10_21.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_61.12: bool = acquire_value %tuple.elem1
// CHECK:STDOUT:   %b: bool = value_binding b, %.loc10_61.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.149
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.7: <bound method> = bound_method %.loc10_61.10, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_61.1: init %empty_tuple.type = call %bound_method.loc10_61.7(%.loc10_61.10)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.6b4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.8: <bound method> = bound_method %.loc10_61.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_61.2: init %empty_tuple.type = call %bound_method.loc10_61.8(%.loc10_61.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_61.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.9: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_61.3: init %empty_tuple.type = call %bound_method.loc10_61.9(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.6c8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_60: <bound method> = bound_method %.loc10_60.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_60: init %empty_tuple.type = call %bound_method.loc10_60(%.loc10_60.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.d23: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.c3b: type = pattern_type %tuple.type.d23 [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8cf: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.261: %EmptyRange.Make.type.8cf = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.6f4: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.165: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.66d(%T.04d, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.529: %tuple.type.as.Copy.impl.Op.type.165 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f5b: <witness> = impl_witness imports.%Copy.impl_witness_table.d1a [concrete]
// CHECK:STDOUT:   %Copy.facet.ed5: %Copy.type = facet_value %C, (%Copy.impl_witness.f5b) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.7b0: <witness> = impl_witness imports.%Copy.impl_witness_table.5ef, @tuple.type.as.Copy.impl.66d(%Copy.facet.ed5, %Copy.facet.ed5) [concrete]
// CHECK:STDOUT:   %Copy.facet.459: %Copy.type = facet_value %tuple.type.d23, (%Copy.impl_witness.7b0) [concrete]
// CHECK:STDOUT:   %EmptyRange.c54: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.ec9: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.87f: %EmptyRange.Make.type.ec9 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.87f, @EmptyRange.Make(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.80d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.5d9: %EmptyRange.as.Iterate.impl.Next.type.80d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a35: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.73f: %EmptyRange.as.Iterate.impl.NewCursor.type.a35 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.e9f: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.592: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.c94: %EmptyRange.as.Iterate.impl.NewCursor.type.592 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.95f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.a8c: %EmptyRange.as.Iterate.impl.Next.type.95f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.c54, (%Iterate.impl_witness.e9f) [concrete]
// CHECK:STDOUT:   %.dc3: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.c94, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.054: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.266: type = class_type @Optional, @Optional(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.a8c, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.2aa: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.3a8: %Optional.HasValue.type.2aa = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.922: type = fn_type @Optional.Get, @Optional(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %Optional.Get.5c2: %Optional.Get.type.922 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.3a8, @Optional.HasValue(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.5c2, @Optional.Get(%Copy.facet.459) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.10d: %type_where = facet_value %tuple.type.d23, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.709: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.10d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.c61: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.709 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.796: %type_where = facet_value %Optional.266, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.7c7: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.796) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.d95: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.7c7 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b98: %type_where = facet_value %EmptyRange.c54, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.766: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b98) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.51d: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.6c6: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8cf) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.261)]
// CHECK:STDOUT:   %Core.import_ref.055: @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.165) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.529)]
// CHECK:STDOUT:   %Copy.impl_witness_table.5ef = impl_witness_table (%Core.import_ref.055), @tuple.type.as.Copy.impl.66d [concrete]
// CHECK:STDOUT:   %Main.import_ref.ba9: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d1a = impl_witness_table (%Main.import_ref.ba9), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e02: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.a35) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.73f)]
// CHECK:STDOUT:   %Main.import_ref.4c1: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.80d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.5d9)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.e02, %Main.import_ref.4c1), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.81c: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.5fa: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7c7 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7c7 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.c3b = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.24b = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39) [concrete = constants.%tuple.6f4]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.d23, (constants.%Copy.impl_witness.7b0) [concrete = constants.%Copy.facet.459]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.459]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.459) [concrete = constants.%EmptyRange.c54]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.ec9 = specific_constant imports.%Main.import_ref.6c6, @EmptyRange(constants.%Copy.facet.459) [concrete = constants.%EmptyRange.Make.87f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.ec9 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.87f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.459) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.c54 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.c54 = call %EmptyRange.Make.specific_fn() to %.loc10_48.1
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.c54 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.dc3 = impl_witness_access constants.%Iterate.impl_witness.e9f, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.c94]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.459) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.c54 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.054 = impl_witness_access constants.%Iterate.impl_witness.e9f, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.a8c]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.459) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.266 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.c54 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.266 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr) to %.loc10_49.1
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.266 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.2aa = specific_constant imports.%Main.import_ref.81c, @Optional(constants.%Copy.facet.459) [concrete = constants.%Optional.HasValue.3a8]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.2aa = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.3a8]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.459) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.266 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.922 = specific_constant imports.%Main.import_ref.5fa, @Optional(constants.%Copy.facet.459) [concrete = constants.%Optional.Get.5c2]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.922 = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.5c2]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.459) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.d23 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.266 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.d23 = call %bound_method.loc10_49.6(%.loc10_49.9) to %.loc10_49.8
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.d23 = temporary %.loc10_49.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %.loc10_49.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %.loc10_49.10, element1
// CHECK:STDOUT:   %C.ref.loc10_12: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_49.11: %C = acquire_value %tuple.elem0
// CHECK:STDOUT:   %a: %C = value_binding a, %.loc10_49.11
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_49.12: %C = acquire_value %tuple.elem1
// CHECK:STDOUT:   %b: %C = value_binding b, %.loc10_49.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.c61
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.7: <bound method> = bound_method %.loc10_49.10, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_49.1: init %empty_tuple.type = call %bound_method.loc10_49.7(%.loc10_49.10)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d95
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.8: <bound method> = bound_method %.loc10_49.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_49.2: init %empty_tuple.type = call %bound_method.loc10_49.8(%.loc10_49.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_49.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.9: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_49.3: init %empty_tuple.type = call %bound_method.loc10_49.9(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.51d
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_48: <bound method> = bound_method %.loc10_48.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_48: init %empty_tuple.type = call %bound_method.loc10_48(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

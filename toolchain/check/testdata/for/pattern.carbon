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
    fn NewCursor[unused self: Self]() -> {} {
      return {};
    }
    fn Next[unused self: Self](unused cursor: {}*) -> Core.Optional(T) {
      return Core.Optional(T).None();
    }
  }
}

class C {
  impl as Core.Copy {
    fn Op[unused self: Self]() -> Self { return {}; }
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
  // CHECK:STDERR:   for (unused x: X in EmptyRange(C).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+4]]:7: note: type `C` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   for (unused x: X in EmptyRange(C).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (unused x: X in EmptyRange(C).Make()) {
  }
}

// CHECK:STDOUT: --- value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.648: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.317: %EmptyRange.Make.type.648 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b02: <witness> = impl_witness imports.%Copy.impl_witness_table.f49 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.b02) [concrete]
// CHECK:STDOUT:   %EmptyRange.63c: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.c46: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.93a: %EmptyRange.Make.type.c46 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.93a, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.eeb: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.82b: %Optional.Get.type.eeb = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f4b: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.40a: %Optional.HasValue.type.f4b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7ff: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.4c5: %EmptyRange.as.Iterate.impl.Next.type.7ff = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.9ee: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.966: %EmptyRange.as.Iterate.impl.NewCursor.type.9ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.0f2: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.97d: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.bb5: %EmptyRange.as.Iterate.impl.NewCursor.type.97d = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.927: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.28e: %EmptyRange.as.Iterate.impl.Next.type.927 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.63c, (%Iterate.impl_witness.0f2) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5c2: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.313: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.df4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5c2, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.bb5, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.eb4: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.313, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.c31: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.28e, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.9f9: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.9ac: %Optional.HasValue.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b89: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.b6e: %Optional.Get.type.b89 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.4de: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.9ac, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.b6e, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.1: type = fn_type @Destroy.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.1: %Destroy.Op.type.790b3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.5: type = fn_type @Destroy.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.5: %Destroy.Op.type.790b3b.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a8b: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.648) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.317)]
// CHECK:STDOUT:   %Main.import_ref.4f1: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_40, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.f49 = impl_witness_table (%Main.import_ref.4f1), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f0a: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f4b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.40a)]
// CHECK:STDOUT:   %Core.import_ref.0ac: @Optional.%Optional.Get.type (%Optional.Get.type.eeb) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.82b)]
// CHECK:STDOUT:   %Main.import_ref.e5a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e66: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.9ee) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.966)]
// CHECK:STDOUT:   %Main.import_ref.1f9: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.7ff) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.4c5)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.e5a, %Main.import_ref.999, %Main.import_ref.e66, %Main.import_ref.1f9), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0d5 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.b02) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.63c]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.c46 = specific_constant imports.%Main.import_ref.a8b, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.93a]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.c46 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.93a]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.63c = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.63c to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.63c = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.df4 = impl_witness_access constants.%Iterate.impl_witness.0f2, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.bb5]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.63c = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.eb4 = impl_witness_access constants.%Iterate.impl_witness.0f2, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.28e]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.c31 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.63c = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.c31 to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %addr)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.c31 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.9f9 = specific_constant imports.%Core.import_ref.f0a, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.9ac]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.9f9 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.9ac]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.c31 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.b89 = specific_constant imports.%Core.import_ref.0ac, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.b6e]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.b89 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.b6e]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.c31 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_36.8 = call %bound_method.loc10_36.6(%.loc10_36.9)
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
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.1(%.loc10_36.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.3: <bound method> = bound_method %var, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_35: init %empty_tuple.type = call %Destroy.Op.bound.loc10_35(%.loc10_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.4de) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.5(%self.param: ref %Optional.c31) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_35(%self.param: ref %EmptyRange.63c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.6be: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.648: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.317: %EmptyRange.Make.type.648 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b02: <witness> = impl_witness imports.%Copy.impl_witness_table.f49 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.b02) [concrete]
// CHECK:STDOUT:   %EmptyRange.63c: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.c46: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.93a: %EmptyRange.Make.type.c46 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.93a, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.eeb: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.82b: %Optional.Get.type.eeb = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f4b: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.40a: %Optional.HasValue.type.f4b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7ff: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.4c5: %EmptyRange.as.Iterate.impl.Next.type.7ff = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.9ee: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.966: %EmptyRange.as.Iterate.impl.NewCursor.type.9ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.0f2: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.97d: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.bb5: %EmptyRange.as.Iterate.impl.NewCursor.type.97d = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.927: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.28e: %EmptyRange.as.Iterate.impl.Next.type.927 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.63c, (%Iterate.impl_witness.0f2) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5c2: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.313: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.df4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5c2, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.bb5, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.eb4: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.313, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.c31: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.28e, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.9f9: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.9ac: %Optional.HasValue.type.9f9 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b89: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.b6e: %Optional.Get.type.b89 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.4de: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.9ac, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.b6e, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.1: type = fn_type @Destroy.Op.loc10_8.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.1: %Destroy.Op.type.790b3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc10_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.5: type = fn_type @Destroy.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.5: %Destroy.Op.type.790b3b.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a8b: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.648) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.317)]
// CHECK:STDOUT:   %Main.import_ref.4f1: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_40, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.f49 = impl_witness_table (%Main.import_ref.4f1), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f0a: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f4b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.40a)]
// CHECK:STDOUT:   %Core.import_ref.0ac: @Optional.%Optional.Get.type (%Optional.Get.type.eeb) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.82b)]
// CHECK:STDOUT:   %Main.import_ref.e5a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e66: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.9ee) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.966)]
// CHECK:STDOUT:   %Main.import_ref.1f9: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.7ff) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.4c5)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.e5a, %Main.import_ref.999, %Main.import_ref.e66, %Main.import_ref.1f9), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0d5 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.0d5 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.b02) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.63c]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.c46 = specific_constant imports.%Main.import_ref.a8b, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.93a]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.c46 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.93a]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.63c = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.63c to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.63c = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.df4 = impl_witness_access constants.%Iterate.impl_witness.0f2, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.bb5]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.63c = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.eb4 = impl_witness_access constants.%Iterate.impl_witness.0f2, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.28e]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.c31 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.63c = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.c31 to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.c31 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.9f9 = specific_constant imports.%Core.import_ref.f0a, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.9ac]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.9f9 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.9ac]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.c31 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.b89 = specific_constant imports.%Core.import_ref.0ac, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.b6e]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.b89 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.b6e]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.c31 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8 = call %bound_method.loc10_40.6(%.loc10_40.8)
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc11: %ptr.6be = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc11)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_8: <bound method> = bound_method %c.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_8: init %empty_tuple.type = call %Destroy.Op.bound.loc10_8(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.1(%.loc10_40.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.2: <bound method> = bound_method %var, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.2(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_39: init %empty_tuple.type = call %Destroy.Op.bound.loc10_39(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_8.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.2(%self.param: ref %struct_type.has_value.value.4de) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.3(%self.param: ref %Optional.c31) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_39(%self.param: ref %EmptyRange.63c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.648: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.317: %EmptyRange.Make.type.648 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.9fa: %tuple.type.24b = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.688: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.27a(%T.baf, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0cf: %tuple.type.as.Copy.impl.Op.type.688 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.9d2: <witness> = impl_witness imports.%Copy.impl_witness_table.82b [concrete]
// CHECK:STDOUT:   %Copy.facet.e28: %Copy.type = facet_value bool, (%Copy.impl_witness.9d2) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.590: <witness> = impl_witness imports.%Copy.impl_witness_table.65e, @tuple.type.as.Copy.impl.27a(%Copy.facet.e28, %Copy.facet.e28) [concrete]
// CHECK:STDOUT:   %Copy.facet.498: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.590) [concrete]
// CHECK:STDOUT:   %EmptyRange.d6b: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.5dd: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.268: %EmptyRange.Make.type.5dd = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.268, @EmptyRange.Make(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.eeb: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.82b: %Optional.Get.type.eeb = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f4b: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.40a: %Optional.HasValue.type.f4b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7ff: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.4c5: %EmptyRange.as.Iterate.impl.Next.type.7ff = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.9ee: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.966: %EmptyRange.as.Iterate.impl.NewCursor.type.9ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.5f9: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.324: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.769: %EmptyRange.as.Iterate.impl.NewCursor.type.324 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.63d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.74e: %EmptyRange.as.Iterate.impl.Next.type.63d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.d6b, (%Iterate.impl_witness.5f9) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.7f4: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d05: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.277: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.7f4, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.769, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %.e77: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d05, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.62d: type = class_type @Optional, @Optional(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.74e, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.2b7: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.a1d: %Optional.HasValue.type.2b7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.cf8: type = fn_type @Optional.Get, @Optional(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %Optional.Get.904: %Optional.Get.type.cf8 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.6f9: type = struct_type {.has_value: bool, .value: %tuple.type.784} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.a1d, @Optional.HasValue(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.904, @Optional.Get(%Copy.facet.498) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.5: type = fn_type @Destroy.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.5: %Destroy.Op.type.790b3b.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.a8b: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.648) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.317)]
// CHECK:STDOUT:   %Core.import_ref.72a: @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.688) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0cf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.65e = impl_witness_table (%Core.import_ref.72a), @tuple.type.as.Copy.impl.27a [concrete]
// CHECK:STDOUT:   %Core.import_ref.f63: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.82b = impl_witness_table (%Core.import_ref.f63), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f0a: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f4b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.40a)]
// CHECK:STDOUT:   %Core.import_ref.0ac: @Optional.%Optional.Get.type (%Optional.Get.type.eeb) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.82b)]
// CHECK:STDOUT:   %Main.import_ref.e5a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e66: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.9ee) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.966)]
// CHECK:STDOUT:   %Main.import_ref.1f9: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.7ff) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.4c5)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.e5a, %Main.import_ref.999, %Main.import_ref.e66, %Main.import_ref.1f9), @EmptyRange.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %.loc10_42: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_48: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_52: %tuple.type.24b = tuple_literal (%.loc10_42, %.loc10_48) [concrete = constants.%tuple.9fa]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.590) [concrete = constants.%Copy.facet.498]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.498]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.498) [concrete = constants.%EmptyRange.d6b]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.5dd = specific_constant imports.%Main.import_ref.a8b, @EmptyRange(constants.%Copy.facet.498) [concrete = constants.%EmptyRange.Make.268]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.5dd = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.268]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.498) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.d6b = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.d6b to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.d6b = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.277 = impl_witness_access constants.%Iterate.impl_witness.5f9, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.769]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.498) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.d6b = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.e77 = impl_witness_access constants.%Iterate.impl_witness.5f9, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.74e]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.498) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.62d = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.d6b = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.62d to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.62d = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.2b7 = specific_constant imports.%Core.import_ref.f0a, @Optional(constants.%Copy.facet.498) [concrete = constants.%Optional.HasValue.a1d]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.2b7 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.a1d]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.498) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.62d = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.cf8 = specific_constant imports.%Core.import_ref.0ac, @Optional(constants.%Copy.facet.498) [concrete = constants.%Optional.Get.904]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.cf8 = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.904]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.498) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.62d = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.784 to %.loc10_61.8 = call %bound_method.loc10_61.6(%.loc10_61.9)
// CHECK:STDOUT:   %.loc10_61.10: ref %tuple.type.784 = temporary %.loc10_61.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %.loc10_61.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %.loc10_61.10, element1
// CHECK:STDOUT:   %.loc10_12: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_61.11: bool = acquire_value %tuple.elem0
// CHECK:STDOUT:   %a: bool = value_binding a, %.loc10_61.11
// CHECK:STDOUT:   %.loc10_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_61.12: bool = acquire_value %tuple.elem1
// CHECK:STDOUT:   %b: bool = value_binding b, %.loc10_61.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.1(%.loc10_61.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.3: <bound method> = bound_method %var, constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_60: init %empty_tuple.type = call %Destroy.Op.bound.loc10_60(%.loc10_60.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.2(%self.param: ref %tuple.type.784) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.4(%self.param: ref %Optional.62d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_60(%self.param: ref %EmptyRange.d6b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.33d: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.410: type = pattern_type %tuple.type.33d [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.648: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.317: %EmptyRange.Make.type.648 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.201: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.688: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.27a(%T.baf, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0cf: %tuple.type.as.Copy.impl.Op.type.688 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b02: <witness> = impl_witness imports.%Copy.impl_witness_table.f49 [concrete]
// CHECK:STDOUT:   %Copy.facet.a1f: %Copy.type = facet_value %C, (%Copy.impl_witness.b02) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.7ba: <witness> = impl_witness imports.%Copy.impl_witness_table.65e, @tuple.type.as.Copy.impl.27a(%Copy.facet.a1f, %Copy.facet.a1f) [concrete]
// CHECK:STDOUT:   %Copy.facet.c7a: %Copy.type = facet_value %tuple.type.33d, (%Copy.impl_witness.7ba) [concrete]
// CHECK:STDOUT:   %EmptyRange.29c: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.685: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.3e1: %EmptyRange.Make.type.685 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.3e1, @EmptyRange.Make(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.eeb: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.82b: %Optional.Get.type.eeb = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f4b: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.40a: %Optional.HasValue.type.f4b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7ff: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.4c5: %EmptyRange.as.Iterate.impl.Next.type.7ff = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.9ee: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.966: %EmptyRange.as.Iterate.impl.NewCursor.type.9ee = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.bf1: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a18: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.69f: %EmptyRange.as.Iterate.impl.NewCursor.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.d08: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f4e: %EmptyRange.as.Iterate.impl.Next.type.d08 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.29c, (%Iterate.impl_witness.bf1) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.07c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.24c: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.18f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.07c, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.69f, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %.528: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.24c, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.0dc: type = class_type @Optional, @Optional(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.f4e, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.a14: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.4a9: %Optional.HasValue.type.a14 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.36e: type = fn_type @Optional.Get, @Optional(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %Optional.Get.490: %Optional.Get.type.36e = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.ecc: type = struct_type {.has_value: bool, .value: %tuple.type.33d} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.4a9, @Optional.HasValue(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.490, @Optional.Get(%Copy.facet.c7a) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.1: type = fn_type @Destroy.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.1: %Destroy.Op.type.790b3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.7: type = fn_type @Destroy.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.7: %Destroy.Op.type.790b3b.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a8b: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.648) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.317)]
// CHECK:STDOUT:   %Core.import_ref.72a: @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.688) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0cf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.65e = impl_witness_table (%Core.import_ref.72a), @tuple.type.as.Copy.impl.27a [concrete]
// CHECK:STDOUT:   %Main.import_ref.4f1: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_40, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.f49 = impl_witness_table (%Main.import_ref.4f1), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f0a: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f4b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.40a)]
// CHECK:STDOUT:   %Core.import_ref.0ac: @Optional.%Optional.Get.type (%Optional.Get.type.eeb) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.82b)]
// CHECK:STDOUT:   %Main.import_ref.e5a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.e66: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.9ee) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.966)]
// CHECK:STDOUT:   %Main.import_ref.1f9: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.7ff) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.4c5)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.e5a, %Main.import_ref.999, %Main.import_ref.e66, %Main.import_ref.1f9), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0d5 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.0d5 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.410 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.24b = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39) [concrete = constants.%tuple.201]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.33d, (constants.%Copy.impl_witness.7ba) [concrete = constants.%Copy.facet.c7a]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.c7a]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.c7a) [concrete = constants.%EmptyRange.29c]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.685 = specific_constant imports.%Main.import_ref.a8b, @EmptyRange(constants.%Copy.facet.c7a) [concrete = constants.%EmptyRange.Make.3e1]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.685 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.3e1]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.c7a) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.29c = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.29c to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.29c = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.18f = impl_witness_access constants.%Iterate.impl_witness.bf1, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.69f]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.c7a) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.29c = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.528 = impl_witness_access constants.%Iterate.impl_witness.bf1, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.f4e]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.c7a) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.0dc = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.29c = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.0dc to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.0dc = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.a14 = specific_constant imports.%Core.import_ref.f0a, @Optional(constants.%Copy.facet.c7a) [concrete = constants.%Optional.HasValue.4a9]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.a14 = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.4a9]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.c7a) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.0dc = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.36e = specific_constant imports.%Core.import_ref.0ac, @Optional(constants.%Copy.facet.c7a) [concrete = constants.%Optional.Get.490]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.36e = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.490]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.c7a) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.33d = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.0dc = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.33d to %.loc10_49.8 = call %bound_method.loc10_49.6(%.loc10_49.9)
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.33d = temporary %.loc10_49.8, %Optional.Get.call
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
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.1(%.loc10_49.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.3: <bound method> = bound_method %var, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT:   %Destroy.Op.call.loc10_48: init %empty_tuple.type = call %Destroy.Op.bound.loc10_48(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.3(%self.param: ref %tuple.type.33d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.ecc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.6(%self.param: ref %Optional.0dc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_48(%self.param: ref %EmptyRange.29c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

class EmptyRange(T: Core.Copy) {
  fn Make() -> Self { return {}; }

  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor(unused self) -> {} {
      return {};
    }
    fn Next(unused self, unused cursor: {}*) -> Core.Optional(T) {
      return Core.Optional(T).None();
    }
  }
}

class C {
  impl as Core.Copy {
    fn Op(unused self) -> Self { return {}; }
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
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %c.patt.aa8: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.140: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0de: %EmptyRange.Make.type.140 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.b10: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.a90: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.72e: %EmptyRange.Make.type.a90 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.72e, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e1b: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.20c: %Optional.Get.type.e1b = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.224: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c4f: %Optional.HasValue.type.224 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.c04: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cdc: %EmptyRange.as.Iterate.impl.Next.type.c04 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.561: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.31b: %EmptyRange.as.Iterate.impl.NewCursor.type.561 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.3a0: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.fdf: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.120: %EmptyRange.as.Iterate.impl.NewCursor.type.fdf = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.74b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.795: %EmptyRange.as.Iterate.impl.Next.type.74b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.b10, (%Iterate.impl_witness.3a0) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.2e7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.645: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.f81: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.2e7, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.120, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.1d5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.645, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.0e3: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.795, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.383: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.45e: %Optional.HasValue.type.383 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.599: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.dc7: %Optional.Get.type.599 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.45e, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.dc7, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a9e: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.140) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0de)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1db: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.224) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c4f)]
// CHECK:STDOUT:   %Core.import_ref.73b: @Optional.%Optional.Get.type (%Optional.Get.type.e1b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.20c)]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.b6b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.561) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.31b)]
// CHECK:STDOUT:   %Main.import_ref.b499: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.c04) = import_ref Main//empty_range, loc11_66, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cdc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.b6b, %Main.import_ref.b499), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt.aa8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.518) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.b10]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.a90 = specific_constant imports.%Main.import_ref.a9e, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.72e]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.a90 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.72e]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.b10 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.b10 to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.b10 = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.f81 = impl_witness_access constants.%Iterate.impl_witness.3a0, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.120]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.b10 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.1d5 = impl_witness_access constants.%Iterate.impl_witness.3a0, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.795]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.0e3 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.b10 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.0e3 to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %addr)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.0e3 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.383 = specific_constant imports.%Core.import_ref.1db, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.45e]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.383 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.45e]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.0e3 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.599 = specific_constant imports.%Core.import_ref.73b, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.dc7]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.599 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.dc7]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.0e3 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_36.8 = call %bound_method.loc10_36.6(%.loc10_36.9)
// CHECK:STDOUT:   %.loc10_36.10: ref %C = temporary %.loc10_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc10_36.11: %C = acquire_value %.loc10_36.10
// CHECK:STDOUT:   %C.ref.loc10_11: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc10_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.1(%.loc10_36.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.3: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.3(%.loc10_36.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.4: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.4: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.4(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%Destroy.Op.1a2547.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.5(%self.param: ref %Optional.0e3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_35(%self.param: ref %EmptyRange.b10) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt.dd0: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt.dd0 [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.140: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0de: %EmptyRange.Make.type.140 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.b10: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.a90: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.72e: %EmptyRange.Make.type.a90 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.72e, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e1b: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.20c: %Optional.Get.type.e1b = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.224: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c4f: %Optional.HasValue.type.224 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.c04: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cdc: %EmptyRange.as.Iterate.impl.Next.type.c04 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.561: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.31b: %EmptyRange.as.Iterate.impl.NewCursor.type.561 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.3a0: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.fdf: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.120: %EmptyRange.as.Iterate.impl.NewCursor.type.fdf = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.74b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.795: %EmptyRange.as.Iterate.impl.Next.type.74b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.b10, (%Iterate.impl_witness.3a0) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.2e7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.645: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.f81: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.2e7, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.120, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.1d5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.645, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.0e3: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.795, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.383: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.45e: %Optional.HasValue.type.383 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.599: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.dc7: %Optional.Get.type.599 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.45e, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.dc7, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_40.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a9e: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.140) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0de)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1db: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.224) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c4f)]
// CHECK:STDOUT:   %Core.import_ref.73b: @Optional.%Optional.Get.type (%Optional.Get.type.e1b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.20c)]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.b6b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.561) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.31b)]
// CHECK:STDOUT:   %Main.import_ref.b499: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.c04) = import_ref Main//empty_range, loc11_66, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cdc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.b6b, %Main.import_ref.b499), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt.dd0]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.518) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.b10]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.a90 = specific_constant imports.%Main.import_ref.a9e, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.72e]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.a90 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.72e]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.b10 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.b10 to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.b10 = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.f81 = impl_witness_access constants.%Iterate.impl_witness.3a0, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.120]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.b10 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.1d5 = impl_witness_access constants.%Iterate.impl_witness.3a0, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.795]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.0e3 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.b10 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.0e3 to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.0e3 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.383 = specific_constant imports.%Core.import_ref.1db, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.45e]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.383 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.45e]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.0e3 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.599 = specific_constant imports.%Core.import_ref.73b, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.dc7]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.599 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.dc7]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.0e3 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8 = call %bound_method.loc10_40.6(%.loc10_40.8)
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc11: %ptr.6b6 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc11)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.1(%.loc10_40.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_8.1: <bound method> = bound_method %c.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc10_8.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_8.1(%c.var)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.2: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.2(%.loc10_40.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_8.2: <bound method> = bound_method %c.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc10_8.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_8.2(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%Destroy.Op.1a2547.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_39: init %empty_tuple.type = call %Destroy.Op.bound.loc10_39(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.3(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.5(%self.param: ref %Optional.0e3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_39(%self.param: ref %EmptyRange.b10) {
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
// CHECK:STDOUT:   %a.patt.9e2: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt.0c3: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type.784: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %tuple.type.784 [concrete]
// CHECK:STDOUT:   %.d8a: %pattern_type.860 = tuple_pattern (%a.patt.9e2, %b.patt.0c3) [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.140: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0de: %EmptyRange.Make.type.140 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.9fa: %tuple.type.24b = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.3ed: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.9bd(%T.f84, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0f8: %tuple.type.as.Copy.impl.Op.type.3ed = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.493: <witness> = impl_witness imports.%Copy.impl_witness_table.e5a, @tuple.type.as.Copy.impl.9bd(%Copy.facet.7af, %Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.facet.701: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.493) [concrete]
// CHECK:STDOUT:   %EmptyRange.d81: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.059: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.787: %EmptyRange.Make.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.787, @EmptyRange.Make(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e1b: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.20c: %Optional.Get.type.e1b = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.224: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c4f: %Optional.HasValue.type.224 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.c04: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cdc: %EmptyRange.as.Iterate.impl.Next.type.c04 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.561: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.31b: %EmptyRange.as.Iterate.impl.NewCursor.type.561 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.65f: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.351: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.49c: %EmptyRange.as.Iterate.impl.NewCursor.type.351 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.5a2: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.3fd: %EmptyRange.as.Iterate.impl.Next.type.5a2 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.d81, (%Iterate.impl_witness.65f) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.28e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.2e8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.d54: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.28e, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.49c, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %.a85: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.2e8, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.4d1: type = class_type @Optional, @Optional(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.3fd, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.34e: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.55e: %Optional.HasValue.type.34e = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b0a: type = fn_type @Optional.Get, @Optional(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.Get.018: %Optional.Get.type.b0a = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.6f9: type = struct_type {.has_value: bool, .value: %tuple.type.784} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.55e, @Optional.HasValue(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.018, @Optional.Get(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.a9e: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.140) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0de)]
// CHECK:STDOUT:   %Core.import_ref.44a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.3ed) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0f8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.44a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1db: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.224) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c4f)]
// CHECK:STDOUT:   %Core.import_ref.73b: @Optional.%Optional.Get.type (%Optional.Get.type.e1b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.20c)]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.b6b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.561) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.31b)]
// CHECK:STDOUT:   %Main.import_ref.b499: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.c04) = import_ref Main//empty_range, loc11_66, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cdc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.b6b, %Main.import_ref.b499), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete = constants.%a.patt.9e2]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt.0c3]
// CHECK:STDOUT:     %.loc10_25: %pattern_type.860 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.d8a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %.loc10_42: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_48: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_52: %tuple.type.24b = tuple_literal (%.loc10_42, %.loc10_48) [concrete = constants.%tuple.9fa]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.493) [concrete = constants.%Copy.facet.701]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.701]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.d81]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.059 = specific_constant imports.%Main.import_ref.a9e, @EmptyRange(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.Make.787]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.059 = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.787]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.d81 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.d81 to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.d81 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.d54 = impl_witness_access constants.%Iterate.impl_witness.65f, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.49c]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.d81 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.a85 = impl_witness_access constants.%Iterate.impl_witness.65f, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.3fd]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.4d1 = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.d81 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.4d1 to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.4d1 = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.34e = specific_constant imports.%Core.import_ref.1db, @Optional(constants.%Copy.facet.701) [concrete = constants.%Optional.HasValue.55e]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.34e = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.55e]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.701) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.4d1 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.b0a = specific_constant imports.%Core.import_ref.73b, @Optional(constants.%Copy.facet.701) [concrete = constants.%Optional.Get.018]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.b0a = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.018]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.701) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.4d1 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.784 to %.loc10_61.8 = call %bound_method.loc10_61.6(%.loc10_61.9)
// CHECK:STDOUT:   %.loc10_61.10: ref %tuple.type.784 = temporary %.loc10_61.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %.loc10_61.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %.loc10_61.10, element1
// CHECK:STDOUT:   %.loc10_61.11: bool = acquire_value %tuple.elem0
// CHECK:STDOUT:   %.loc10_12: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %a: bool = wrapper_binding a, %.loc10_61.11
// CHECK:STDOUT:   %.loc10_61.12: bool = acquire_value %tuple.elem1
// CHECK:STDOUT:   %.loc10_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc10_61.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.1(%.loc10_61.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.3: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.3(%.loc10_61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.4: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.4: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.4(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%Destroy.Op.1a2547.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.4(%self.param: ref %Optional.4d1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_60(%self.param: ref %EmptyRange.d81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %a.patt.1ad: %pattern_type.98b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt.29b: %pattern_type.98b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type.748: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %tuple.type.748 [concrete]
// CHECK:STDOUT:   %.d56: %pattern_type.8b4 = tuple_pattern (%a.patt.1ad, %b.patt.29b) [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.140: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0de: %EmptyRange.Make.type.140 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.d95: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.3ed: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.9bd(%T.f84, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0f8: %tuple.type.as.Copy.impl.Op.type.3ed = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet.e83: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5b5: <witness> = impl_witness imports.%Copy.impl_witness_table.e5a, @tuple.type.as.Copy.impl.9bd(%Copy.facet.e83, %Copy.facet.e83) [concrete]
// CHECK:STDOUT:   %Copy.facet.38c: %Copy.type = facet_value %tuple.type.748, (%Copy.impl_witness.5b5) [concrete]
// CHECK:STDOUT:   %EmptyRange.8d3: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.96a: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.24e: %EmptyRange.Make.type.96a = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.24e, @EmptyRange.Make(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e1b: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.20c: %Optional.Get.type.e1b = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.224: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c4f: %Optional.HasValue.type.224 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.c04: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cdc: %EmptyRange.as.Iterate.impl.Next.type.c04 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.561: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.31b: %EmptyRange.as.Iterate.impl.NewCursor.type.561 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.28e: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.b49: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.b24: %EmptyRange.as.Iterate.impl.NewCursor.type.b49 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.af7: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.e7f: %EmptyRange.as.Iterate.impl.Next.type.af7 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.8d3, (%Iterate.impl_witness.28e) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.59a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.bd6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.522: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.59a, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.b24, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %.f09: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.bd6, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.992: type = class_type @Optional, @Optional(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.e7f, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.e1b: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.b06: %Optional.HasValue.type.e1b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.4b3: type = fn_type @Optional.Get, @Optional(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.Get.091: %Optional.Get.type.4b3 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9b1: type = struct_type {.has_value: bool, .value: %tuple.type.748} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.b06, @Optional.HasValue(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.091, @Optional.Get(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a9e: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.140) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0de)]
// CHECK:STDOUT:   %Core.import_ref.44a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.3ed) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0f8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.44a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1db: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.224) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c4f)]
// CHECK:STDOUT:   %Core.import_ref.73b: @Optional.%Optional.Get.type (%Optional.Get.type.e1b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.20c)]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.b6b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.561) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.31b)]
// CHECK:STDOUT:   %Main.import_ref.b499: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.c04) = import_ref Main//empty_range, loc11_66, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cdc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.b6b, %Main.import_ref.b499), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.98b = value_binding_pattern a [concrete = constants.%a.patt.1ad]
// CHECK:STDOUT:     %b.patt: %pattern_type.98b = value_binding_pattern b [concrete = constants.%b.patt.29b]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.8b4 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.d56]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.24b = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39) [concrete = constants.%tuple.d95]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.748, (constants.%Copy.impl_witness.5b5) [concrete = constants.%Copy.facet.38c]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.38c]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.8d3]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.96a = specific_constant imports.%Main.import_ref.a9e, @EmptyRange(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.Make.24e]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.96a = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.24e]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.8d3 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.8d3 to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.8d3 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.522 = impl_witness_access constants.%Iterate.impl_witness.28e, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.b24]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.8d3 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.f09 = impl_witness_access constants.%Iterate.impl_witness.28e, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.e7f]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.992 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.8d3 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.992 to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.992 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.e1b = specific_constant imports.%Core.import_ref.1db, @Optional(constants.%Copy.facet.38c) [concrete = constants.%Optional.HasValue.b06]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.e1b = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.b06]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.38c) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.992 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.4b3 = specific_constant imports.%Core.import_ref.73b, @Optional(constants.%Copy.facet.38c) [concrete = constants.%Optional.Get.091]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.4b3 = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.091]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.38c) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.748 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.992 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.748 to %.loc10_49.8 = call %bound_method.loc10_49.6(%.loc10_49.9)
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.748 = temporary %.loc10_49.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %.loc10_49.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %.loc10_49.10, element1
// CHECK:STDOUT:   %.loc10_49.11: %C = acquire_value %tuple.elem0
// CHECK:STDOUT:   %C.ref.loc10_12: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %a: %C = wrapper_binding a, %.loc10_49.11
// CHECK:STDOUT:   %.loc10_49.12: %C = acquire_value %tuple.elem1
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %b: %C = wrapper_binding b, %.loc10_49.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.1(%.loc10_49.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.Op.1a2547.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.3: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.Op.1a2547.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.3(%.loc10_49.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.4: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.4: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.4(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%Destroy.Op.1a2547.7
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.3(%self.param: ref %tuple.type.748) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.6(%self.param: ref %Optional.992) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_48(%self.param: ref %EmptyRange.8d3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

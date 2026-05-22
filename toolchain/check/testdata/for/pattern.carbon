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
// CHECK:STDOUT:   %pattern_type.4d2: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.45a: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.8de: %EmptyRange.Make.type.45a = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fad: <witness> = impl_witness imports.%Copy.impl_witness_table.2ad [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.fad) [concrete]
// CHECK:STDOUT:   %EmptyRange.6dc: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.e2c: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.6c7: %EmptyRange.Make.type.e2c = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.6c7, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.dc8: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ead: %Optional.Get.type.dc8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ae5: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.959: %Optional.HasValue.type.ae5 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.70e: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f25: %EmptyRange.as.Iterate.impl.Next.type.70e = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a2c: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.067: %EmptyRange.as.Iterate.impl.NewCursor.type.a2c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.4f0: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.77f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.18d: %EmptyRange.as.Iterate.impl.NewCursor.type.77f = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.ad3: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.df9: %EmptyRange.as.Iterate.impl.Next.type.ad3 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.6dc, (%Iterate.impl_witness.4f0) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.89a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ccc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.a2f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.89a, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.18d, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.f37: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ccc, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.1d0: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.df9, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.fe2: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.5bd: %Optional.HasValue.type.fe2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.045: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.c6c: %Optional.Get.type.045 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.470: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.5bd, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.c6c, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.1: type = fn_type @Destroy.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.1: %Destroy.Op.type.af7ec0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.5: type = fn_type @Destroy.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.5: %Destroy.Op.type.af7ec0.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.267: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.45a) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.8de)]
// CHECK:STDOUT:   %Main.import_ref.eb5: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_40, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.2ad = impl_witness_table (%Main.import_ref.eb5), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5cb: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.ae5) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.959)]
// CHECK:STDOUT:   %Core.import_ref.a91: @Optional.%Optional.Get.type (%Optional.Get.type.dc8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ead)]
// CHECK:STDOUT:   %Main.import_ref.30c = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.aa3: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.a2c) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.067)]
// CHECK:STDOUT:   %Main.import_ref.dbf: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.70e) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.f25)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.30c, %Main.import_ref.999, %Main.import_ref.aa3, %Main.import_ref.dbf), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.4d2 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.fad) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.6dc]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.e2c = specific_constant imports.%Main.import_ref.267, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.6c7]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.e2c = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.6c7]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.6dc = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.6dc to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.6dc = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.a2f = impl_witness_access constants.%Iterate.impl_witness.4f0, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.18d]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.6dc = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.f37 = impl_witness_access constants.%Iterate.impl_witness.4f0, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.df9]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.1d0 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.6dc = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.1d0 to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %addr)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.1d0 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.fe2 = specific_constant imports.%Core.import_ref.5cb, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.5bd]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.fe2 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.5bd]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.1d0 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.045 = specific_constant imports.%Core.import_ref.a91, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.c6c]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.045 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.c6c]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.1d0 = acquire_value %.loc10_36.2
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
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.1(%.loc10_36.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_36.3: <bound method> = bound_method %var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_36.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_36.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%Destroy.Op.1dc86d.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.470) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_36.5(%self.param: ref %Optional.1d0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_35(%self.param: ref %EmptyRange.6dc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.9ec: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.cae: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.62d: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.caa: %EmptyRange.Make.type.62d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.92c: <witness> = impl_witness imports.%Copy.impl_witness_table.6de [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.92c) [concrete]
// CHECK:STDOUT:   %EmptyRange.29c: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.f40: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.66a: %EmptyRange.Make.type.f40 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.66a, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.dc8: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ead: %Optional.Get.type.dc8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ae5: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.959: %Optional.HasValue.type.ae5 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.bc0: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.616: %EmptyRange.as.Iterate.impl.Next.type.bc0 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.1d7: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.41a: %EmptyRange.as.Iterate.impl.NewCursor.type.1d7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.020: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.51f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.d41: %EmptyRange.as.Iterate.impl.NewCursor.type.51f = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.cd5: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.56f: %EmptyRange.as.Iterate.impl.Next.type.cd5 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.29c, (%Iterate.impl_witness.020) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a52: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.cec: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.33c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a52, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.d41, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.3e67b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.cec, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.063: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.56f, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.0cd: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.49d: %Optional.HasValue.type.0cd = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.a40: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.514: %Optional.Get.type.a40 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.806: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.49d, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.514, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.1: type = fn_type @Destroy.Op.loc10_8.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.1: %Destroy.Op.type.af7ec0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc10_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.5: type = fn_type @Destroy.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.5: %Destroy.Op.type.af7ec0.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.dfd: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.62d) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.caa)]
// CHECK:STDOUT:   %Main.import_ref.eb5: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_40, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.6de = impl_witness_table (%Main.import_ref.eb5), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5cb: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.ae5) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.959)]
// CHECK:STDOUT:   %Core.import_ref.a91: @Optional.%Optional.Get.type (%Optional.Get.type.dc8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ead)]
// CHECK:STDOUT:   %Main.import_ref.30c = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.d3c: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.1d7) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.41a)]
// CHECK:STDOUT:   %Main.import_ref.0e9: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.bc0) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.616)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.30c, %Main.import_ref.999, %Main.import_ref.d3c, %Main.import_ref.0e9), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.cae = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.cae = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.92c) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.29c]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.f40 = specific_constant imports.%Main.import_ref.dfd, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.66a]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.f40 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.66a]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.29c = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.29c to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.29c = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.33c = impl_witness_access constants.%Iterate.impl_witness.020, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.d41]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.29c = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.3e67b = impl_witness_access constants.%Iterate.impl_witness.020, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.56f]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.063 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.29c = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.063 to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.063 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.0cd = specific_constant imports.%Core.import_ref.5cb, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.49d]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.0cd = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.49d]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.063 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.a40 = specific_constant imports.%Core.import_ref.a91, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.514]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.a40 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.514]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.063 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8 = call %bound_method.loc10_40.6(%.loc10_40.8)
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc11: %ptr.9ec = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc11)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_8: <bound method> = bound_method %c.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_8: init %empty_tuple.type = call %Destroy.Op.bound.loc10_8(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.1(%.loc10_40.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_40.2: <bound method> = bound_method %var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_40.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_40.2(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%Destroy.Op.1dc86d.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.2(%self.param: ref %struct_type.has_value.value.806) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_40.3(%self.param: ref %Optional.063) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_39(%self.param: ref %EmptyRange.29c) {
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
// CHECK:STDOUT:   %EmptyRange.Make.type.c22: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.9cb: %EmptyRange.Make.type.c22 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.9fa: %tuple.type.24b = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.688: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.27a(%T.baf, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0cf: %tuple.type.as.Copy.impl.Op.type.688 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a0b: <witness> = impl_witness imports.%Copy.impl_witness_table.8f5 [concrete]
// CHECK:STDOUT:   %Copy.facet.b8c: %Copy.type = facet_value bool, (%Copy.impl_witness.a0b) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.56e: <witness> = impl_witness imports.%Copy.impl_witness_table.1e8, @tuple.type.as.Copy.impl.27a(%Copy.facet.b8c, %Copy.facet.b8c) [concrete]
// CHECK:STDOUT:   %Copy.facet.581: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.56e) [concrete]
// CHECK:STDOUT:   %EmptyRange.34b: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.01d: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.915: %EmptyRange.Make.type.01d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.915, @EmptyRange.Make(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.dc8: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ead: %Optional.Get.type.dc8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ae5: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.959: %Optional.HasValue.type.ae5 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.bf5: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.545: %EmptyRange.as.Iterate.impl.Next.type.bf5 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.3ac: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f9c: %EmptyRange.as.Iterate.impl.NewCursor.type.3ac = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.ca9: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.fd8: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.bf0: %EmptyRange.as.Iterate.impl.NewCursor.type.fd8 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.f25: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.ef8: %EmptyRange.as.Iterate.impl.Next.type.f25 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.34b, (%Iterate.impl_witness.ca9) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.6be: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.138: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.34e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.6be, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.bf0, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %.2aa: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.138, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.6c5: type = class_type @Optional, @Optional(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.ef8, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.ed5: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.b4c: %Optional.HasValue.type.ed5 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.8a5: type = fn_type @Optional.Get, @Optional(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %Optional.Get.96e: %Optional.Get.type.8a5 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.6f9: type = struct_type {.has_value: bool, .value: %tuple.type.784} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.b4c, @Optional.HasValue(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.96e, @Optional.Get(%Copy.facet.581) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.5: type = fn_type @Destroy.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.5: %Destroy.Op.type.af7ec0.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.7f2: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.c22) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.9cb)]
// CHECK:STDOUT:   %Core.import_ref.cc0: @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.688) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0cf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1e8 = impl_witness_table (%Core.import_ref.cc0), @tuple.type.as.Copy.impl.27a [concrete]
// CHECK:STDOUT:   %Core.import_ref.41e: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.8f5 = impl_witness_table (%Core.import_ref.41e), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5cb: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.ae5) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.959)]
// CHECK:STDOUT:   %Core.import_ref.a91: @Optional.%Optional.Get.type (%Optional.Get.type.dc8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ead)]
// CHECK:STDOUT:   %Main.import_ref.30c = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.2bf: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.3ac) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.f9c)]
// CHECK:STDOUT:   %Main.import_ref.cc1: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.bf5) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.545)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.30c, %Main.import_ref.999, %Main.import_ref.2bf, %Main.import_ref.cc1), @EmptyRange.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.56e) [concrete = constants.%Copy.facet.581]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.581]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.581) [concrete = constants.%EmptyRange.34b]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.01d = specific_constant imports.%Main.import_ref.7f2, @EmptyRange(constants.%Copy.facet.581) [concrete = constants.%EmptyRange.Make.915]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.01d = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.915]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.581) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.34b = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.34b to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.34b = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.34e = impl_witness_access constants.%Iterate.impl_witness.ca9, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.bf0]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.581) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.34b = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.2aa = impl_witness_access constants.%Iterate.impl_witness.ca9, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.ef8]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.581) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.6c5 = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.34b = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.6c5 to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.6c5 = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.ed5 = specific_constant imports.%Core.import_ref.5cb, @Optional(constants.%Copy.facet.581) [concrete = constants.%Optional.HasValue.b4c]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.ed5 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.b4c]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.581) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.6c5 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.8a5 = specific_constant imports.%Core.import_ref.a91, @Optional(constants.%Copy.facet.581) [concrete = constants.%Optional.Get.96e]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.8a5 = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.96e]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.581) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.6c5 = acquire_value %.loc10_61.2
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
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.1(%.loc10_61.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_61.3: <bound method> = bound_method %var, constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10_61.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_61.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%Destroy.Op.1dc86d.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.4(%self.param: ref %Optional.6c5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_60(%self.param: ref %EmptyRange.34b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.fe3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.da0: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.696: type = pattern_type %tuple.type.da0 [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.baf: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.f75: type = fn_type @EmptyRange.Make, @EmptyRange(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.a07: %EmptyRange.Make.type.f75 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.521: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.688: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.27a(%T.baf, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0cf: %tuple.type.as.Copy.impl.Op.type.688 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ff3: <witness> = impl_witness imports.%Copy.impl_witness_table.460 [concrete]
// CHECK:STDOUT:   %Copy.facet.68d: %Copy.type = facet_value %C, (%Copy.impl_witness.ff3) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.17a: <witness> = impl_witness imports.%Copy.impl_witness_table.1e8, @tuple.type.as.Copy.impl.27a(%Copy.facet.68d, %Copy.facet.68d) [concrete]
// CHECK:STDOUT:   %Copy.facet.9c8: %Copy.type = facet_value %tuple.type.da0, (%Copy.impl_witness.17a) [concrete]
// CHECK:STDOUT:   %EmptyRange.ca4: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.fb3: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.0e7: %EmptyRange.Make.type.fb3 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.0e7, @EmptyRange.Make(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.Get.type.dc8: type = fn_type @Optional.Get, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ead: %Optional.Get.type.dc8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ae5: type = fn_type @Optional.HasValue, @Optional(%T.baf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.959: %Optional.HasValue.type.ae5 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.724: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.76f: %EmptyRange.as.Iterate.impl.Next.type.724 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.bae: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.baf) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.fa4: %EmptyRange.as.Iterate.impl.NewCursor.type.bae = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.683: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c7d: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.29e: %EmptyRange.as.Iterate.impl.NewCursor.type.c7d = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.ec1: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.4e4: %EmptyRange.as.Iterate.impl.Next.type.ec1 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.ca4, (%Iterate.impl_witness.683) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.6de: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.cc8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.866: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.6de, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.29e, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %.d48: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.cc8, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.c01: type = class_type @Optional, @Optional(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.4e4, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.349: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.d81: %Optional.HasValue.type.349 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.ce5: type = fn_type @Optional.Get, @Optional(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %Optional.Get.fc5: %Optional.Get.type.ce5 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.c3a: type = struct_type {.has_value: bool, .value: %tuple.type.da0} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.d81, @Optional.HasValue(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.fc5, @Optional.Get(%Copy.facet.9c8) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.1: type = fn_type @Destroy.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.1: %Destroy.Op.type.af7ec0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.7: type = fn_type @Destroy.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.7: %Destroy.Op.type.af7ec0.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.792: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.f75) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.a07)]
// CHECK:STDOUT:   %Core.import_ref.cc0: @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.688) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.27a.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0cf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1e8 = impl_witness_table (%Core.import_ref.cc0), @tuple.type.as.Copy.impl.27a [concrete]
// CHECK:STDOUT:   %Main.import_ref.eb5: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_40, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.460 = impl_witness_table (%Main.import_ref.eb5), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5cb: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.ae5) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.959)]
// CHECK:STDOUT:   %Core.import_ref.a91: @Optional.%Optional.Get.type (%Optional.Get.type.dc8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ead)]
// CHECK:STDOUT:   %Main.import_ref.30c = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.007: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.bae) = import_ref Main//empty_range, loc8_45, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.fa4)]
// CHECK:STDOUT:   %Main.import_ref.575: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.724) = import_ref Main//empty_range, loc11_72, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.76f)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.30c, %Main.import_ref.999, %Main.import_ref.007, %Main.import_ref.575), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.fe3 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.fe3 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.696 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.24b = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39) [concrete = constants.%tuple.521]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.da0, (constants.%Copy.impl_witness.17a) [concrete = constants.%Copy.facet.9c8]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.9c8]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.9c8) [concrete = constants.%EmptyRange.ca4]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.fb3 = specific_constant imports.%Main.import_ref.792, @EmptyRange(constants.%Copy.facet.9c8) [concrete = constants.%EmptyRange.Make.0e7]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.fb3 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.0e7]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.9c8) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.ca4 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.ca4 to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.ca4 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.866 = impl_witness_access constants.%Iterate.impl_witness.683, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.29e]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.9c8) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.ca4 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.d48 = impl_witness_access constants.%Iterate.impl_witness.683, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.4e4]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.9c8) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.c01 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.ca4 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.c01 to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.c01 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.349 = specific_constant imports.%Core.import_ref.5cb, @Optional(constants.%Copy.facet.9c8) [concrete = constants.%Optional.HasValue.d81]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.349 = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.d81]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.9c8) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.c01 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.ce5 = specific_constant imports.%Core.import_ref.a91, @Optional(constants.%Copy.facet.9c8) [concrete = constants.%Optional.Get.fc5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.ce5 = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.fc5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.9c8) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.da0 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.c01 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.da0 to %.loc10_49.8 = call %bound_method.loc10_49.6(%.loc10_49.9)
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.da0 = temporary %.loc10_49.8, %Optional.Get.call
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
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.1: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.1(%.loc10_49.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.Op.1dc86d.6
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.2: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_49.3: <bound method> = bound_method %var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10_49.3: init %empty_tuple.type = call %Destroy.Op.bound.loc10_49.3(%var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%Destroy.Op.1dc86d.7
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.3(%self.param: ref %tuple.type.da0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.c3a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_49.6(%self.param: ref %Optional.c01) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_48(%self.param: ref %EmptyRange.ca4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

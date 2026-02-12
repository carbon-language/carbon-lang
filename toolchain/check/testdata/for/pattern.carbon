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
// CHECK:STDOUT:   %T.56c: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0dc: %EmptyRange.Make.type.8d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fab: <witness> = impl_witness imports.%Copy.impl_witness_table.c52 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.fab) [concrete]
// CHECK:STDOUT:   %EmptyRange.77b: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.c0f: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.f9f: %EmptyRange.Make.type.c0f = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.f9f, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b9d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.682: %EmptyRange.as.Iterate.impl.Next.type.b9d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.517: type = fn_type @Optional.Get, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ece: %Optional.Get.type.517 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.927: type = fn_type @Optional.HasValue, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.966: %Optional.HasValue.type.927 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.16c: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.cbe: %EmptyRange.as.Iterate.impl.NewCursor.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.96d: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.268: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.da5: %EmptyRange.as.Iterate.impl.NewCursor.type.268 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.046: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.d08: %EmptyRange.as.Iterate.impl.Next.type.046 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.77b, (%Iterate.impl_witness.96d) [concrete]
// CHECK:STDOUT:   %.d58: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.da5, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.23b: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.0f3: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.d08, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.a17: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.858: %Optional.HasValue.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.c50: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.e0f: %Optional.Get.type.c50 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.858, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.e0f, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc10_36.1 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc10_36.2 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc10_36.3 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc10_35 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.299: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0dc)]
// CHECK:STDOUT:   %Main.import_ref.cb5: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.c52 = impl_witness_table (%Main.import_ref.cb5), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.888 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.868: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.16c) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.cbe)]
// CHECK:STDOUT:   %Main.import_ref.574: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.b9d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.682)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.888, %Main.import_ref.999, %Main.import_ref.868, %Main.import_ref.574), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.37b: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.927) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.966)]
// CHECK:STDOUT:   %Main.import_ref.97a: @Optional.%Optional.Get.type (%Optional.Get.type.517) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ece)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.fab) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.77b]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.c0f = specific_constant imports.%Main.import_ref.299, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.f9f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.c0f = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.f9f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.77b = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.77b to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.77b = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.d58 = impl_witness_access constants.%Iterate.impl_witness.96d, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.da5]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.77b = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.23b = impl_witness_access constants.%Iterate.impl_witness.96d, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.d08]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.0f3 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.77b = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.0f3 to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %addr)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.0f3 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.a17 = specific_constant imports.%Main.import_ref.37b, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.858]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.a17 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.858]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.0f3 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.c50 = specific_constant imports.%Main.import_ref.97a, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.e0f]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.c50 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.e0f]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.0f3 = acquire_value %.loc10_36.2
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
// CHECK:STDOUT:   %DestroyOp.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc10_36.1: init %empty_tuple.type = call %DestroyOp.bound.loc10_36.1(%.loc10_36.10)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc10_36.2: init %empty_tuple.type = call %DestroyOp.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_36.3: <bound method> = bound_method %var, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc10_36.3: init %empty_tuple.type = call %DestroyOp.bound.loc10_36.3(%var)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_35: init %empty_tuple.type = call %DestroyOp.bound.loc10_35(%.loc10_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_36.1(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_36.2(%self.param: ref %Optional.0f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_36.3(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_35(%self.param: ref %EmptyRange.77b) = "no_op";
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
// CHECK:STDOUT:   %T.56c: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0dc: %EmptyRange.Make.type.8d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fab: <witness> = impl_witness imports.%Copy.impl_witness_table.c52 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.fab) [concrete]
// CHECK:STDOUT:   %EmptyRange.77b: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.c0f: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.f9f: %EmptyRange.Make.type.c0f = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.f9f, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b9d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.682: %EmptyRange.as.Iterate.impl.Next.type.b9d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.517: type = fn_type @Optional.Get, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ece: %Optional.Get.type.517 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.927: type = fn_type @Optional.HasValue, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.966: %Optional.HasValue.type.927 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.16c: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.cbe: %EmptyRange.as.Iterate.impl.NewCursor.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.96d: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.268: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.da5: %EmptyRange.as.Iterate.impl.NewCursor.type.268 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.046: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.d08: %EmptyRange.as.Iterate.impl.Next.type.046 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.77b, (%Iterate.impl_witness.96d) [concrete]
// CHECK:STDOUT:   %.d58: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.da5, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.23b: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.0f3: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.d08, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.a17: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.858: %Optional.HasValue.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.c50: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.e0f: %Optional.Get.type.c50 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.858, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.e0f, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc10_8 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc10_40.1 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc10_40.2 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc10_39 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.299: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0dc)]
// CHECK:STDOUT:   %Main.import_ref.cb5: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.c52 = impl_witness_table (%Main.import_ref.cb5), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.888 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.868: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.16c) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.cbe)]
// CHECK:STDOUT:   %Main.import_ref.574: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.b9d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.682)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.888, %Main.import_ref.999, %Main.import_ref.868, %Main.import_ref.574), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.37b: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.927) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.966)]
// CHECK:STDOUT:   %Main.import_ref.97a: @Optional.%Optional.Get.type (%Optional.Get.type.517) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ece)]
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.fab) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.77b]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.c0f = specific_constant imports.%Main.import_ref.299, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.f9f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.c0f = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.f9f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.77b = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.77b to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.77b = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.d58 = impl_witness_access constants.%Iterate.impl_witness.96d, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.da5]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.77b = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.23b = impl_witness_access constants.%Iterate.impl_witness.96d, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.d08]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.0f3 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.77b = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.0f3 to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.0f3 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.a17 = specific_constant imports.%Main.import_ref.37b, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.858]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.a17 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.858]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.0f3 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.c50 = specific_constant imports.%Main.import_ref.97a, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.e0f]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.c50 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.e0f]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.0f3 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8 = call %bound_method.loc10_40.6(%.loc10_40.8)
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
// CHECK:STDOUT:   %DestroyOp.bound.loc10_8: <bound method> = bound_method %c.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc10_8: init %empty_tuple.type = call %DestroyOp.bound.loc10_8(%c.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc10_40.1: init %empty_tuple.type = call %DestroyOp.bound.loc10_40.1(%.loc10_40.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_40.2: <bound method> = bound_method %var, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc10_40.2: init %empty_tuple.type = call %DestroyOp.bound.loc10_40.2(%var)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_39: init %empty_tuple.type = call %DestroyOp.bound.loc10_39(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_8(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_40.1(%self.param: ref %Optional.0f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_40.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_39(%self.param: ref %EmptyRange.77b) = "no_op";
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
// CHECK:STDOUT:   %T.56c: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0dc: %EmptyRange.Make.type.8d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.9fa: %tuple.type.24b = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.57a: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.da7(%T.56c, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.7f4: %tuple.type.as.Copy.impl.Op.type.57a = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.819: <witness> = impl_witness imports.%Copy.impl_witness_table.164 [concrete]
// CHECK:STDOUT:   %Copy.facet.07e: %Copy.type = facet_value bool, (%Copy.impl_witness.819) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f9f: <witness> = impl_witness imports.%Copy.impl_witness_table.3c8, @tuple.type.as.Copy.impl.da7(%Copy.facet.07e, %Copy.facet.07e) [concrete]
// CHECK:STDOUT:   %Copy.facet.3ce: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.f9f) [concrete]
// CHECK:STDOUT:   %EmptyRange.717: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.a76: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.d35: %EmptyRange.Make.type.a76 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.d35, @EmptyRange.Make(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b9d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.682: %EmptyRange.as.Iterate.impl.Next.type.b9d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.517: type = fn_type @Optional.Get, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ece: %Optional.Get.type.517 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.927: type = fn_type @Optional.HasValue, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.966: %Optional.HasValue.type.927 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.16c: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.cbe: %EmptyRange.as.Iterate.impl.NewCursor.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.04a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a15: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.90c: %EmptyRange.as.Iterate.impl.NewCursor.type.a15 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.f9b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.4a9: %EmptyRange.as.Iterate.impl.Next.type.f9b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.717, (%Iterate.impl_witness.04a) [concrete]
// CHECK:STDOUT:   %.0cf: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.90c, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.165: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.0cc: type = class_type @Optional, @Optional(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.4a9, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.a20: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c8a: %Optional.HasValue.type.a20 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.36c: type = fn_type @Optional.Get, @Optional(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %Optional.Get.e9c: %Optional.Get.type.36c = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.c8a, @Optional.HasValue(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.e9c, @Optional.Get(%Copy.facet.3ce) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc10_61.1 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc10_61.2 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc10_61.3 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc10_60 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.299: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0dc)]
// CHECK:STDOUT:   %Core.import_ref.445: @tuple.type.as.Copy.impl.da7.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.57a) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.da7.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.7f4)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3c8 = impl_witness_table (%Core.import_ref.445), @tuple.type.as.Copy.impl.da7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.4d6: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.164 = impl_witness_table (%Core.import_ref.4d6), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.888 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.868: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.16c) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.cbe)]
// CHECK:STDOUT:   %Main.import_ref.574: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.b9d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.682)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.888, %Main.import_ref.999, %Main.import_ref.868, %Main.import_ref.574), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.37b: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.927) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.966)]
// CHECK:STDOUT:   %Main.import_ref.97a: @Optional.%Optional.Get.type (%Optional.Get.type.517) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ece)]
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.f9f) [concrete = constants.%Copy.facet.3ce]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.3ce]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.3ce) [concrete = constants.%EmptyRange.717]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.a76 = specific_constant imports.%Main.import_ref.299, @EmptyRange(constants.%Copy.facet.3ce) [concrete = constants.%EmptyRange.Make.d35]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.a76 = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.d35]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.3ce) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.717 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.717 to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.717 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.0cf = impl_witness_access constants.%Iterate.impl_witness.04a, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.90c]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.3ce) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.717 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.165 = impl_witness_access constants.%Iterate.impl_witness.04a, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.4a9]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.3ce) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.0cc = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.717 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.0cc to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.0cc = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.a20 = specific_constant imports.%Main.import_ref.37b, @Optional(constants.%Copy.facet.3ce) [concrete = constants.%Optional.HasValue.c8a]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.a20 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.c8a]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.3ce) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.0cc = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.36c = specific_constant imports.%Main.import_ref.97a, @Optional(constants.%Copy.facet.3ce) [concrete = constants.%Optional.Get.e9c]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.36c = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.e9c]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.3ce) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.0cc = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.784 to %.loc10_61.8 = call %bound_method.loc10_61.6(%.loc10_61.9)
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
// CHECK:STDOUT:   %DestroyOp.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc10_61.1: init %empty_tuple.type = call %DestroyOp.bound.loc10_61.1(%.loc10_61.10)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc10_61.2: init %empty_tuple.type = call %DestroyOp.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_61.3: <bound method> = bound_method %var, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc10_61.3: init %empty_tuple.type = call %DestroyOp.bound.loc10_61.3(%var)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_60: init %empty_tuple.type = call %DestroyOp.bound.loc10_60(%.loc10_60.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_61.1(%self.param: ref %tuple.type.784) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_61.2(%self.param: ref %Optional.0cc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_61.3(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_60(%self.param: ref %EmptyRange.717) = "no_op";
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
// CHECK:STDOUT:   %T.56c: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.8d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.0dc: %EmptyRange.Make.type.8d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.6f4: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.57a: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.da7(%T.56c, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.7f4: %tuple.type.as.Copy.impl.Op.type.57a = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fab: <witness> = impl_witness imports.%Copy.impl_witness_table.c52 [concrete]
// CHECK:STDOUT:   %Copy.facet.54f: %Copy.type = facet_value %C, (%Copy.impl_witness.fab) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9e2: <witness> = impl_witness imports.%Copy.impl_witness_table.3c8, @tuple.type.as.Copy.impl.da7(%Copy.facet.54f, %Copy.facet.54f) [concrete]
// CHECK:STDOUT:   %Copy.facet.ad0: %Copy.type = facet_value %tuple.type.d23, (%Copy.impl_witness.9e2) [concrete]
// CHECK:STDOUT:   %EmptyRange.849: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.7da: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.38a: %EmptyRange.Make.type.7da = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.38a, @EmptyRange.Make(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b9d: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.682: %EmptyRange.as.Iterate.impl.Next.type.b9d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.517: type = fn_type @Optional.Get, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.Get.ece: %Optional.Get.type.517 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.927: type = fn_type @Optional.HasValue, @Optional(%T.56c) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.966: %Optional.HasValue.type.927 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.16c: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.56c) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.cbe: %EmptyRange.as.Iterate.impl.NewCursor.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.a9d: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a4b: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.94e: %EmptyRange.as.Iterate.impl.NewCursor.type.a4b = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.24c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.77c: %EmptyRange.as.Iterate.impl.Next.type.24c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.849, (%Iterate.impl_witness.a9d) [concrete]
// CHECK:STDOUT:   %.826: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.94e, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.fce: type = class_type @Optional, @Optional(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.77c, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.32e: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.cc1: %Optional.HasValue.type.32e = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.733: type = fn_type @Optional.Get, @Optional(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %Optional.Get.dd4: %Optional.Get.type.733 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.cc1, @Optional.HasValue(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.dd4, @Optional.Get(%Copy.facet.ad0) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc10_49.1 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc10_49.2 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc10_49.3 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc10_48 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.299: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.8d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.0dc)]
// CHECK:STDOUT:   %Core.import_ref.445: @tuple.type.as.Copy.impl.da7.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.57a) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.da7.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.7f4)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3c8 = impl_witness_table (%Core.import_ref.445), @tuple.type.as.Copy.impl.da7 [concrete]
// CHECK:STDOUT:   %Main.import_ref.cb5: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.c52 = impl_witness_table (%Main.import_ref.cb5), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.888 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.868: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.16c) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.cbe)]
// CHECK:STDOUT:   %Main.import_ref.574: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.b9d) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.682)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.888, %Main.import_ref.999, %Main.import_ref.868, %Main.import_ref.574), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.37b: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.927) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.966)]
// CHECK:STDOUT:   %Main.import_ref.97a: @Optional.%Optional.Get.type (%Optional.Get.type.517) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.ece)]
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.d23, (constants.%Copy.impl_witness.9e2) [concrete = constants.%Copy.facet.ad0]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.ad0]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.ad0) [concrete = constants.%EmptyRange.849]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.7da = specific_constant imports.%Main.import_ref.299, @EmptyRange(constants.%Copy.facet.ad0) [concrete = constants.%EmptyRange.Make.38a]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.7da = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.38a]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.ad0) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.849 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.849 to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.849 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.826 = impl_witness_access constants.%Iterate.impl_witness.a9d, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.94e]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.ad0) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.849 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.2ac = impl_witness_access constants.%Iterate.impl_witness.a9d, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.77c]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.ad0) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.fce = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.849 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.fce to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.fce = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.32e = specific_constant imports.%Main.import_ref.37b, @Optional(constants.%Copy.facet.ad0) [concrete = constants.%Optional.HasValue.cc1]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.32e = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.cc1]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.ad0) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.fce = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.733 = specific_constant imports.%Main.import_ref.97a, @Optional(constants.%Copy.facet.ad0) [concrete = constants.%Optional.Get.dd4]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.733 = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.dd4]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.ad0) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.d23 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.fce = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.d23 to %.loc10_49.8 = call %bound_method.loc10_49.6(%.loc10_49.9)
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
// CHECK:STDOUT:   %DestroyOp.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc10_49.1: init %empty_tuple.type = call %DestroyOp.bound.loc10_49.1(%.loc10_49.10)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc10_49.2: init %empty_tuple.type = call %DestroyOp.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_49.3: <bound method> = bound_method %var, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc10_49.3: init %empty_tuple.type = call %DestroyOp.bound.loc10_49.3(%var)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_48: init %empty_tuple.type = call %DestroyOp.bound.loc10_48(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_49.1(%self.param: ref %tuple.type.d23) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_49.2(%self.param: ref %Optional.fce) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_49.3(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10_48(%self.param: ref %EmptyRange.849) = "no_op";
// CHECK:STDOUT:

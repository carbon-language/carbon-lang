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
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.e24: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b80: %EmptyRange.Make.type.e24 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1de: <witness> = impl_witness imports.%Copy.impl_witness_table.2c6 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.1de) [concrete]
// CHECK:STDOUT:   %EmptyRange.ab3: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.f28: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.203: %EmptyRange.Make.type.f28 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.203, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.a9c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f98: %EmptyRange.as.Iterate.impl.Next.type.a9c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.748: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.247: %EmptyRange.as.Iterate.impl.NewCursor.type.748 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.413: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a39: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.6d4: %EmptyRange.as.Iterate.impl.NewCursor.type.a39 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7df: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.201: %EmptyRange.as.Iterate.impl.Next.type.7df = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.ab3, (%Iterate.impl_witness.413) [concrete]
// CHECK:STDOUT:   %.c04: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.6d4, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.534: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.47f: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.201, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.970: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ac0: %Optional.HasValue.type.970 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f65: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.71a: %Optional.Get.type.f65 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ac0, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.71a, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.be8: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.857: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.be8) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.68a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.857 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.94b: %type_where = facet_value %Optional.47f, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0b9: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.94b) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9a0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0b9 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b6d: %type_where = facet_value %EmptyRange.ab3, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.888: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b6d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.218: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.888 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.849: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.e24) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b80)]
// CHECK:STDOUT:   %Main.import_ref.157: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.2c6 = impl_witness_table (%Main.import_ref.157), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.712: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.748) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.247)]
// CHECK:STDOUT:   %Main.import_ref.c9f: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.a9c) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.f98)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.712, %Main.import_ref.c9f), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.070: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.9fe: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.1de) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.ab3]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.f28 = specific_constant imports.%Main.import_ref.849, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.203]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.f28 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.203]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.ab3 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.ab3 = call %EmptyRange.Make.specific_fn() to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.ab3 = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.c04 = impl_witness_access constants.%Iterate.impl_witness.413, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.6d4]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.ab3 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.534 = impl_witness_access constants.%Iterate.impl_witness.413, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.201]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.47f = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.ab3 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.47f = call %bound_method.loc10_36.4(%.loc10_35.4, %addr) to %.loc10_36.1
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.47f = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.970 = specific_constant imports.%Main.import_ref.070, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.ac0]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.970 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.ac0]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.47f = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.f65 = specific_constant imports.%Main.import_ref.9fe, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.71a]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f65 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.71a]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.47f = acquire_value %.loc10_36.2
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.68a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.7: <bound method> = bound_method %.loc10_36.10, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_36.1: init %empty_tuple.type = call %bound_method.loc10_36.7(%.loc10_36.10)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9a0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.8: <bound method> = bound_method %.loc10_36.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_36.2: init %empty_tuple.type = call %bound_method.loc10_36.8(%.loc10_36.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_36.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.9: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_36.3: init %empty_tuple.type = call %bound_method.loc10_36.9(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.218
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
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.e24: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b80: %EmptyRange.Make.type.e24 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1de: <witness> = impl_witness imports.%Copy.impl_witness_table.2c6 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.1de) [concrete]
// CHECK:STDOUT:   %EmptyRange.ab3: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.f28: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.203: %EmptyRange.Make.type.f28 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.203, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.a9c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f98: %EmptyRange.as.Iterate.impl.Next.type.a9c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.748: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.247: %EmptyRange.as.Iterate.impl.NewCursor.type.748 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.413: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.a39: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.6d4: %EmptyRange.as.Iterate.impl.NewCursor.type.a39 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7df: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.201: %EmptyRange.as.Iterate.impl.Next.type.7df = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.ab3, (%Iterate.impl_witness.413) [concrete]
// CHECK:STDOUT:   %.c04: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.6d4, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.534: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.47f: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.201, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.970: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ac0: %Optional.HasValue.type.970 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f65: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.71a: %Optional.Get.type.f65 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ac0, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.71a, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.be8: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.857: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.be8) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.68a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.857 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.94b: %type_where = facet_value %Optional.47f, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0b9: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.94b) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9a0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0b9 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b6d: %type_where = facet_value %EmptyRange.ab3, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.888: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b6d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.218: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.888 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.849: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.e24) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b80)]
// CHECK:STDOUT:   %Main.import_ref.157: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.2c6 = impl_witness_table (%Main.import_ref.157), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.712: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.748) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.247)]
// CHECK:STDOUT:   %Main.import_ref.c9f: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.a9c) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.f98)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.712, %Main.import_ref.c9f), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.070: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.9fe: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.1de) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.ab3]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.f28 = specific_constant imports.%Main.import_ref.849, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.203]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.f28 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.203]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.ab3 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.ab3 = call %EmptyRange.Make.specific_fn() to %.loc10_39.1
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.ab3 = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.c04 = impl_witness_access constants.%Iterate.impl_witness.413, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.6d4]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.ab3 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.534 = impl_witness_access constants.%Iterate.impl_witness.413, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.201]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.47f = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.ab3 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.47f = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10) to %.loc10_40.1
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.47f = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.970 = specific_constant imports.%Main.import_ref.070, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.ac0]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.970 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.ac0]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.47f = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.f65 = specific_constant imports.%Main.import_ref.9fe, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.71a]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f65 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.71a]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.47f = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc10_40.6(%.loc10_40.8) to %.loc10_8
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc11: %ptr.019 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc11)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_8: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.68a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_8: init %empty_tuple.type = call %bound_method.loc10_8(%c.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9a0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_40.7: <bound method> = bound_method %.loc10_40.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_40.1: init %empty_tuple.type = call %bound_method.loc10_40.7(%.loc10_40.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_40.2: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_40.8: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_40.2: init %empty_tuple.type = call %bound_method.loc10_40.8(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.218
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
// CHECK:STDOUT:   %EmptyRange.Make.type.e24: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b80: %EmptyRange.Make.type.e24 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.9fa: %tuple.type.24b = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.165: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.66d(%T.04d, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.529: %tuple.type.as.Copy.impl.Op.type.165 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1f3: <witness> = impl_witness imports.%Copy.impl_witness_table.416 [concrete]
// CHECK:STDOUT:   %Copy.facet.559: %Copy.type = facet_value bool, (%Copy.impl_witness.1f3) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a0d: <witness> = impl_witness imports.%Copy.impl_witness_table.bac, @tuple.type.as.Copy.impl.66d(%Copy.facet.559, %Copy.facet.559) [concrete]
// CHECK:STDOUT:   %Copy.facet.f9d: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.a0d) [concrete]
// CHECK:STDOUT:   %EmptyRange.584: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.9be: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.e06: %EmptyRange.Make.type.9be = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.e06, @EmptyRange.Make(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.a9c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f98: %EmptyRange.as.Iterate.impl.Next.type.a9c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.748: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.247: %EmptyRange.as.Iterate.impl.NewCursor.type.748 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.60c: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.286: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.866: %EmptyRange.as.Iterate.impl.NewCursor.type.286 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.7c5: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.3b1: %EmptyRange.as.Iterate.impl.Next.type.7c5 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.584, (%Iterate.impl_witness.60c) [concrete]
// CHECK:STDOUT:   %.581: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.866, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.963: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.319: type = class_type @Optional, @Optional(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.3b1, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.825: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.cfe: %Optional.HasValue.type.825 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.3c6: type = fn_type @Optional.Get, @Optional(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %Optional.Get.181: %Optional.Get.type.3c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.cfe, @Optional.HasValue(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.181, @Optional.Get(%Copy.facet.f9d) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.d7c: %type_where = facet_value %tuple.type.784, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5f5: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d7c) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.149: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5f5 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.459: %type_where = facet_value %Optional.319, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5a3: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.459) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.3cd: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5a3 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.4cb: %type_where = facet_value %EmptyRange.584, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.36e: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4cb) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f7a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.36e = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.849: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.e24) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b80)]
// CHECK:STDOUT:   %Core.import_ref.187: @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.165) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.529)]
// CHECK:STDOUT:   %Copy.impl_witness_table.bac = impl_witness_table (%Core.import_ref.187), @tuple.type.as.Copy.impl.66d [concrete]
// CHECK:STDOUT:   %Core.import_ref.afa: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.416 = impl_witness_table (%Core.import_ref.afa), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.712: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.748) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.247)]
// CHECK:STDOUT:   %Main.import_ref.c9f: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.a9c) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.f98)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.712, %Main.import_ref.c9f), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.070: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.9fe: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.a0d) [concrete = constants.%Copy.facet.f9d]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.f9d]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.f9d) [concrete = constants.%EmptyRange.584]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.9be = specific_constant imports.%Main.import_ref.849, @EmptyRange(constants.%Copy.facet.f9d) [concrete = constants.%EmptyRange.Make.e06]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.9be = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.e06]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.f9d) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.584 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.584 = call %EmptyRange.Make.specific_fn() to %.loc10_60.1
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.584 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.581 = impl_witness_access constants.%Iterate.impl_witness.60c, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.866]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.f9d) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.584 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.963 = impl_witness_access constants.%Iterate.impl_witness.60c, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.3b1]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.f9d) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.319 = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.584 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.319 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr) to %.loc10_61.1
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.319 = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.825 = specific_constant imports.%Main.import_ref.070, @Optional(constants.%Copy.facet.f9d) [concrete = constants.%Optional.HasValue.cfe]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.825 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.cfe]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.f9d) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.319 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.3c6 = specific_constant imports.%Main.import_ref.9fe, @Optional(constants.%Copy.facet.f9d) [concrete = constants.%Optional.Get.181]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.3c6 = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.181]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.f9d) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.319 = acquire_value %.loc10_61.2
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3cd
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.8: <bound method> = bound_method %.loc10_61.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_61.2: init %empty_tuple.type = call %bound_method.loc10_61.8(%.loc10_61.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_61.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.9: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_61.3: init %empty_tuple.type = call %bound_method.loc10_61.9(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f7a
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
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.56b: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.99e: type = pattern_type %tuple.type.56b [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.04d: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.e24: type = fn_type @EmptyRange.Make, @EmptyRange(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b80: %EmptyRange.Make.type.e24 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.0d6: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.165: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.66d(%T.04d, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.529: %tuple.type.as.Copy.impl.Op.type.165 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1de: <witness> = impl_witness imports.%Copy.impl_witness_table.2c6 [concrete]
// CHECK:STDOUT:   %Copy.facet.762: %Copy.type = facet_value %C, (%Copy.impl_witness.1de) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.409: <witness> = impl_witness imports.%Copy.impl_witness_table.bac, @tuple.type.as.Copy.impl.66d(%Copy.facet.762, %Copy.facet.762) [concrete]
// CHECK:STDOUT:   %Copy.facet.fdf: %Copy.type = facet_value %tuple.type.56b, (%Copy.impl_witness.409) [concrete]
// CHECK:STDOUT:   %EmptyRange.350: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.374: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.d3d: %EmptyRange.Make.type.374 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.d3d, @EmptyRange.Make(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.a9c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.f98: %EmptyRange.as.Iterate.impl.Next.type.a9c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.3de: type = fn_type @Optional.Get, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.138: %Optional.Get.type.3de = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.366: type = fn_type @Optional.HasValue, @Optional(%T.04d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c11: %Optional.HasValue.type.366 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.748: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.04d) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.247: %EmptyRange.as.Iterate.impl.NewCursor.type.748 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.71a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c7d: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.24a: %EmptyRange.as.Iterate.impl.NewCursor.type.c7d = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.015: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.b91: %EmptyRange.as.Iterate.impl.Next.type.015 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.350, (%Iterate.impl_witness.71a) [concrete]
// CHECK:STDOUT:   %.6ca: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.24a, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.525: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.eb6: type = class_type @Optional, @Optional(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.b91, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.bec: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.bce: %Optional.HasValue.type.bec = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e0d: type = fn_type @Optional.Get, @Optional(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %Optional.Get.faa: %Optional.Get.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.bce, @Optional.HasValue(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.faa, @Optional.Get(%Copy.facet.fdf) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.c79: %type_where = facet_value %tuple.type.56b, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ffa: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.c79) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.84b: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ffa = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.af3: %type_where = facet_value %Optional.eb6, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.4bb: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.af3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a4e: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.4bb = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.315: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.746 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.bea: %type_where = facet_value %EmptyRange.350, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.2ba: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.bea) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.690: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.849: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.e24) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b80)]
// CHECK:STDOUT:   %Core.import_ref.187: @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.165) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.66d.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.529)]
// CHECK:STDOUT:   %Copy.impl_witness_table.bac = impl_witness_table (%Core.import_ref.187), @tuple.type.as.Copy.impl.66d [concrete]
// CHECK:STDOUT:   %Main.import_ref.157: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.2c6 = impl_witness_table (%Main.import_ref.157), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.f10 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.712: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.748) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.247)]
// CHECK:STDOUT:   %Main.import_ref.c9f: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.a9c) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.f98)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.f10, %Main.import_ref.999, %Main.import_ref.712, %Main.import_ref.c9f), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.070: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.366) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c11)]
// CHECK:STDOUT:   %Main.import_ref.9fe: @Optional.%Optional.Get.type (%Optional.Get.type.3de) = import_ref Main//empty_range, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.99e = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.24b = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39) [concrete = constants.%tuple.0d6]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.56b, (constants.%Copy.impl_witness.409) [concrete = constants.%Copy.facet.fdf]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.fdf]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.fdf) [concrete = constants.%EmptyRange.350]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.374 = specific_constant imports.%Main.import_ref.849, @EmptyRange(constants.%Copy.facet.fdf) [concrete = constants.%EmptyRange.Make.d3d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.374 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.d3d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.fdf) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.350 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.350 = call %EmptyRange.Make.specific_fn() to %.loc10_48.1
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.350 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.6ca = impl_witness_access constants.%Iterate.impl_witness.71a, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.24a]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.fdf) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.350 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.525 = impl_witness_access constants.%Iterate.impl_witness.71a, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.b91]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.fdf) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.eb6 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.350 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.eb6 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr) to %.loc10_49.1
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.eb6 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.bec = specific_constant imports.%Main.import_ref.070, @Optional(constants.%Copy.facet.fdf) [concrete = constants.%Optional.HasValue.bce]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.bec = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.bce]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.fdf) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.eb6 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.e0d = specific_constant imports.%Main.import_ref.9fe, @Optional(constants.%Copy.facet.fdf) [concrete = constants.%Optional.Get.faa]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.e0d = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.faa]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.fdf) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.56b = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.eb6 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.56b = call %bound_method.loc10_49.6(%.loc10_49.9) to %.loc10_49.8
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.56b = temporary %.loc10_49.8, %Optional.Get.call
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.84b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.7: <bound method> = bound_method %.loc10_49.10, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_49.1: init %empty_tuple.type = call %bound_method.loc10_49.7(%.loc10_49.10)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a4e
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.8: <bound method> = bound_method %.loc10_49.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_49.2: init %empty_tuple.type = call %bound_method.loc10_49.8(%.loc10_49.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_49.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.315
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.9: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_49.3: init %empty_tuple.type = call %bound_method.loc10_49.9(%var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.690
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_48: <bound method> = bound_method %.loc10_48.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_48: init %empty_tuple.type = call %bound_method.loc10_48(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

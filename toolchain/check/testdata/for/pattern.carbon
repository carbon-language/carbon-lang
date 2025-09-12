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
// CHECK:STDOUT:   %T.049: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.7d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.4cc: %EmptyRange.Make.type.7d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b58: <witness> = impl_witness imports.%Copy.impl_witness_table.943 [concrete]
// CHECK:STDOUT:   %Copy.facet.bfc: %Copy.type = facet_value %C, (%Copy.impl_witness.b58) [concrete]
// CHECK:STDOUT:   %EmptyRange.8bd: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.f76: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.393: %EmptyRange.Make.type.f76 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.393, @EmptyRange.Make(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.be4: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.61a: %EmptyRange.as.Iterate.impl.NewCursor.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.859: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cc1: %EmptyRange.as.Iterate.impl.Next.type.859 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.091: type = fn_type @Optional.HasValue, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.5a0: %Optional.HasValue.type.091 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.618: type = fn_type @Optional.Get, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9cc: %Optional.Get.type.618 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.a1a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.1f8: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.ddc: %EmptyRange.as.Iterate.impl.NewCursor.type.1f8 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b3e: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.67c: %EmptyRange.as.Iterate.impl.Next.type.b3e = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.8bd, (%Iterate.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %.c55: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.ddc, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.eac: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.202: type = class_type @Optional, @Optional(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.67c, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.632: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.13f: %Optional.HasValue.type.632 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e88: type = fn_type @Optional.Get, @Optional(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.Get.736: %Optional.Get.type.e88 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.13f, @Optional.HasValue(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.736, @Optional.Get(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.af6: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.5bc: %Optional.as.Destroy.impl.Op.type.af6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.00c: type = ptr_type %Optional.202 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.456: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.24b: %T.as.Destroy.impl.Op.type.456 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.8ee: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.dd8: %EmptyRange.as.Destroy.impl.Op.type.8ee = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2e0: type = ptr_type %EmptyRange.8bd [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.787: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.7d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.4cc)]
// CHECK:STDOUT:   %Main.import_ref.e69: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.943 = impl_witness_table (%Main.import_ref.e69), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.101 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f30: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.be4) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.61a)]
// CHECK:STDOUT:   %Main.import_ref.881: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.859) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cc1)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.101, %Main.import_ref.999, %Main.import_ref.8f30, %Main.import_ref.881), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.13f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.091) = import_ref Main//empty_range, inst151 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.5a0)]
// CHECK:STDOUT:   %Main.import_ref.a3b: @Optional.%Optional.Get.type (%Optional.Get.type.618) = import_ref Main//empty_range, inst152 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9cc)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%C, (constants.%Copy.impl_witness.b58) [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.8bd]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.f76 = specific_constant imports.%Main.import_ref.787, @EmptyRange(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.Make.393]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.f76 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.393]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.8bd = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.8bd = call %EmptyRange.Make.specific_fn() to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.8bd = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.c55 = impl_witness_access constants.%Iterate.impl_witness.a1a, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.ddc]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.8bd = bind_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10_36.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.eac = impl_witness_access constants.%Iterate.impl_witness.a1a, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.67c]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.202 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.8bd = bind_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.202 = call %bound_method.loc10_36.4(%.loc10_35.4, %addr.loc10_36.1) to %.loc10_36.1
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.202 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.632 = specific_constant imports.%Main.import_ref.13f, @Optional(constants.%Copy.facet.bfc) [concrete = constants.%Optional.HasValue.13f]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.632 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.13f]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.bfc) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.202 = bind_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.e88 = specific_constant imports.%Main.import_ref.a3b, @Optional(constants.%Copy.facet.bfc) [concrete = constants.%Optional.Get.736]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.e88 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.736]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.bfc) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.202 = bind_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc10_36.6(%.loc10_36.9) to %.loc10_36.8
// CHECK:STDOUT:   %C.ref.loc10_11: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_36.10: ref %C = temporary %.loc10_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc10_36.11: %C = bind_value %.loc10_36.10
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc10_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_36.10, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10_36.2: %ptr.019 = addr_of %.loc10_36.10
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc10_36.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_36.2, constants.%Optional.as.Destroy.impl.Op.5bc
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.7: <bound method> = bound_method %.loc10_36.2, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_36.3: %ptr.00c = addr_of %.loc10_36.2
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_36.7(%addr.loc10_36.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.24b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_36.8: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_36.4: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_36.8(%addr.loc10_36.4)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_35.2, constants.%EmptyRange.as.Destroy.impl.Op.dd8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_35.2, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_35: %ptr.2e0 = addr_of %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_35(%addr.loc10_35)
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
// CHECK:STDOUT:   %T.049: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.7d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.4cc: %EmptyRange.Make.type.7d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b58: <witness> = impl_witness imports.%Copy.impl_witness_table.943 [concrete]
// CHECK:STDOUT:   %Copy.facet.bfc: %Copy.type = facet_value %C, (%Copy.impl_witness.b58) [concrete]
// CHECK:STDOUT:   %EmptyRange.8bd: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.f76: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.393: %EmptyRange.Make.type.f76 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.393, @EmptyRange.Make(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.be4: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.61a: %EmptyRange.as.Iterate.impl.NewCursor.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.859: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cc1: %EmptyRange.as.Iterate.impl.Next.type.859 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.091: type = fn_type @Optional.HasValue, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.5a0: %Optional.HasValue.type.091 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.618: type = fn_type @Optional.Get, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9cc: %Optional.Get.type.618 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.a1a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.1f8: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.ddc: %EmptyRange.as.Iterate.impl.NewCursor.type.1f8 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b3e: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.67c: %EmptyRange.as.Iterate.impl.Next.type.b3e = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.8bd, (%Iterate.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %.c55: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.ddc, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.eac: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.202: type = class_type @Optional, @Optional(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.67c, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.632: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.13f: %Optional.HasValue.type.632 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e88: type = fn_type @Optional.Get, @Optional(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.Get.736: %Optional.Get.type.e88 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.13f, @Optional.HasValue(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.736, @Optional.Get(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.af6: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.5bc: %Optional.as.Destroy.impl.Op.type.af6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.00c: type = ptr_type %Optional.202 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.456: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.24b: %T.as.Destroy.impl.Op.type.456 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.8ee: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.dd8: %EmptyRange.as.Destroy.impl.Op.type.8ee = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2e0: type = ptr_type %EmptyRange.8bd [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.787: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.7d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.4cc)]
// CHECK:STDOUT:   %Main.import_ref.e69: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.943 = impl_witness_table (%Main.import_ref.e69), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.101 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f30: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.be4) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.61a)]
// CHECK:STDOUT:   %Main.import_ref.881: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.859) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cc1)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.101, %Main.import_ref.999, %Main.import_ref.8f30, %Main.import_ref.881), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.13f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.091) = import_ref Main//empty_range, inst151 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.5a0)]
// CHECK:STDOUT:   %Main.import_ref.a3b: @Optional.%Optional.Get.type (%Optional.Get.type.618) = import_ref Main//empty_range, inst152 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9cc)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%C, (constants.%Copy.impl_witness.b58) [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.8bd]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.f76 = specific_constant imports.%Main.import_ref.787, @EmptyRange(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.Make.393]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.f76 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.393]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.8bd = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.8bd = call %EmptyRange.Make.specific_fn() to %.loc10_39.1
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.8bd = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.c55 = impl_witness_access constants.%Iterate.impl_witness.a1a, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.ddc]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.8bd = bind_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10_40.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.eac = impl_witness_access constants.%Iterate.impl_witness.a1a, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.67c]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.bfc) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.202 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.8bd = bind_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.202 = call %bound_method.loc10_40.4(%.loc10_39.4, %addr.loc10_40.1) to %.loc10_40.1
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.202 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.632 = specific_constant imports.%Main.import_ref.13f, @Optional(constants.%Copy.facet.bfc) [concrete = constants.%Optional.HasValue.13f]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.632 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.13f]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.bfc) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.202 = bind_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.e88 = specific_constant imports.%Main.import_ref.a3b, @Optional(constants.%Copy.facet.bfc) [concrete = constants.%Optional.Get.736]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.e88 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.736]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.bfc) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.202 = bind_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc10_40.6(%.loc10_40.8) to %.loc10_8
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc11: %ptr.019 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc11)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10_8: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc10_8)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_40.2, constants.%Optional.as.Destroy.impl.Op.5bc
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_40.7: <bound method> = bound_method %.loc10_40.2, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_40.2: %ptr.00c = addr_of %.loc10_40.2
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_40.7(%addr.loc10_40.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.24b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_40.8: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_40.3: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_40.8(%addr.loc10_40.3)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_39.2, constants.%EmptyRange.as.Destroy.impl.Op.dd8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_39: <bound method> = bound_method %.loc10_39.2, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_39: %ptr.2e0 = addr_of %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_39(%addr.loc10_39)
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
// CHECK:STDOUT:   %T.049: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.7d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.4cc: %EmptyRange.Make.type.7d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.d93: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.177(%T.049, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.1f0: %tuple.type.as.Copy.impl.Op.type.d93 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.5da: <witness> = impl_witness imports.%Copy.impl_witness_table.185 [concrete]
// CHECK:STDOUT:   %Copy.facet.2e3: %Copy.type = facet_value bool, (%Copy.impl_witness.5da) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.df3: <witness> = impl_witness imports.%Copy.impl_witness_table.442, @tuple.type.as.Copy.impl.177(%Copy.facet.2e3, %Copy.facet.2e3) [concrete]
// CHECK:STDOUT:   %Copy.facet.e16: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.df3) [concrete]
// CHECK:STDOUT:   %EmptyRange.bb5: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.e08: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.f6d: %EmptyRange.Make.type.e08 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.f6d, @EmptyRange.Make(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.be4: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.61a: %EmptyRange.as.Iterate.impl.NewCursor.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.859: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cc1: %EmptyRange.as.Iterate.impl.Next.type.859 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.091: type = fn_type @Optional.HasValue, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.5a0: %Optional.HasValue.type.091 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.618: type = fn_type @Optional.Get, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9cc: %Optional.Get.type.618 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.0ab: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.3b4: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.5b3: %EmptyRange.as.Iterate.impl.NewCursor.type.3b4 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.1b4: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.df7: %EmptyRange.as.Iterate.impl.Next.type.1b4 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.bb5, (%Iterate.impl_witness.0ab) [concrete]
// CHECK:STDOUT:   %.1a1: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.5b3, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.03b: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.9b3: type = class_type @Optional, @Optional(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.df7, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.bcc: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1e6: %Optional.HasValue.type.bcc = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.ee0: type = fn_type @Optional.Get, @Optional(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Optional.Get.5bb: %Optional.Get.type.ee0 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b85: type = ptr_type %tuple.type.784 [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.1e6, @Optional.HasValue(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.5bb, @Optional.Get(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ca1: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.af8: %T.as.Destroy.impl.Op.type.ca1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.cec: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.d4b: %Optional.as.Destroy.impl.Op.type.cec = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.9dd: type = ptr_type %Optional.9b3 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.456: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.24b: %T.as.Destroy.impl.Op.type.456 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.6f7: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%Copy.facet.e16) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.49a: %EmptyRange.as.Destroy.impl.Op.type.6f7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4e1: type = ptr_type %EmptyRange.bb5 [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.787: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.7d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.4cc)]
// CHECK:STDOUT:   %Core.import_ref.d98: @tuple.type.as.Copy.impl.177.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.d93) = import_ref Core//prelude/parts/copy, loc48_31, loaded [symbolic = @tuple.type.as.Copy.impl.177.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.1f0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.442 = impl_witness_table (%Core.import_ref.d98), @tuple.type.as.Copy.impl.177 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a3a: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc20_31, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.185 = impl_witness_table (%Core.import_ref.a3a), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.101 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f30: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.be4) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.61a)]
// CHECK:STDOUT:   %Main.import_ref.881: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.859) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cc1)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.101, %Main.import_ref.999, %Main.import_ref.8f30, %Main.import_ref.881), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.13f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.091) = import_ref Main//empty_range, inst151 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.5a0)]
// CHECK:STDOUT:   %Main.import_ref.a3b: @Optional.%Optional.Get.type (%Optional.Get.type.618) = import_ref Main//empty_range, inst152 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9cc)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_25: %pattern_type.860 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Bool.call.loc10_42: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %Bool.call.loc10_48: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc10_52: %tuple.type.24b = tuple_literal (%Bool.call.loc10_42, %Bool.call.loc10_48)
// CHECK:STDOUT:   %.loc10_53.1: type = value_of_initializer %Bool.call.loc10_42 [concrete = bool]
// CHECK:STDOUT:   %.loc10_53.2: type = converted %Bool.call.loc10_42, %.loc10_53.1 [concrete = bool]
// CHECK:STDOUT:   %.loc10_53.3: type = value_of_initializer %Bool.call.loc10_48 [concrete = bool]
// CHECK:STDOUT:   %.loc10_53.4: type = converted %Bool.call.loc10_48, %.loc10_53.3 [concrete = bool]
// CHECK:STDOUT:   %Copy.facet.loc10_53.1: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5da) [concrete = constants.%Copy.facet.2e3]
// CHECK:STDOUT:   %.loc10_53.5: %Copy.type = converted bool, %Copy.facet.loc10_53.1 [concrete = constants.%Copy.facet.2e3]
// CHECK:STDOUT:   %Copy.facet.loc10_53.2: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5da) [concrete = constants.%Copy.facet.2e3]
// CHECK:STDOUT:   %.loc10_53.6: %Copy.type = converted bool, %Copy.facet.loc10_53.2 [concrete = constants.%Copy.facet.2e3]
// CHECK:STDOUT:   %Copy.facet.loc10_53.3: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.df3) [concrete = constants.%Copy.facet.e16]
// CHECK:STDOUT:   %.loc10_53.7: %Copy.type = converted %.loc10_52, %Copy.facet.loc10_53.3 [concrete = constants.%Copy.facet.e16]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.e16) [concrete = constants.%EmptyRange.bb5]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.e08 = specific_constant imports.%Main.import_ref.787, @EmptyRange(constants.%Copy.facet.e16) [concrete = constants.%EmptyRange.Make.f6d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.e08 = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.f6d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.e16) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.bb5 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.bb5 = call %EmptyRange.Make.specific_fn() to %.loc10_60.1
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.bb5 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.1a1 = impl_witness_access constants.%Iterate.impl_witness.0ab, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.5b3]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.e16) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.bb5 = bind_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10_61.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.03b = impl_witness_access constants.%Iterate.impl_witness.0ab, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.df7]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.e16) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.9b3 = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.bb5 = bind_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.9b3 = call %bound_method.loc10_61.4(%.loc10_60.4, %addr.loc10_61.1) to %.loc10_61.1
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.9b3 = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.bcc = specific_constant imports.%Main.import_ref.13f, @Optional(constants.%Copy.facet.e16) [concrete = constants.%Optional.HasValue.1e6]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.bcc = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.1e6]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.e16) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.9b3 = bind_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.ee0 = specific_constant imports.%Main.import_ref.a3b, @Optional(constants.%Copy.facet.e16) [concrete = constants.%Optional.Get.5bb]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.ee0 = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.5bb]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.e16) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.9b3 = bind_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.784 = call %bound_method.loc10_61.6(%.loc10_61.9) to %.loc10_61.8
// CHECK:STDOUT:   %.loc10_61.10: ref %tuple.type.784 = temporary %.loc10_61.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %.loc10_61.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %.loc10_61.10, element1
// CHECK:STDOUT:   %.loc10_12.1: type = splice_block %.loc10_12.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc10_12: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_12.2: type = value_of_initializer %Bool.call.loc10_12 [concrete = bool]
// CHECK:STDOUT:     %.loc10_12.3: type = converted %Bool.call.loc10_12, %.loc10_12.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_61.11: bool = bind_value %tuple.elem0
// CHECK:STDOUT:   %a: bool = bind_name a, %.loc10_61.11
// CHECK:STDOUT:   %.loc10_21.1: type = splice_block %.loc10_21.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc10_21: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_21.2: type = value_of_initializer %Bool.call.loc10_21 [concrete = bool]
// CHECK:STDOUT:     %.loc10_21.3: type = converted %Bool.call.loc10_21, %.loc10_21.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_61.12: bool = bind_value %tuple.elem1
// CHECK:STDOUT:   %b: bool = bind_name b, %.loc10_61.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%T.as.Destroy.impl.Op.af8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.7: <bound method> = bound_method %.loc10_61.10, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_61.2: %ptr.b85 = addr_of %.loc10_61.10
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10_61.1: init %empty_tuple.type = call %bound_method.loc10_61.7(%addr.loc10_61.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_61.2, constants.%Optional.as.Destroy.impl.Op.d4b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.8: <bound method> = bound_method %.loc10_61.2, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_61.3: %ptr.9dd = addr_of %.loc10_61.2
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_61.8(%addr.loc10_61.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10_61.2: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.24b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.9: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_61.4: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10_61.2: init %empty_tuple.type = call %bound_method.loc10_61.9(%addr.loc10_61.4)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_60.2, constants.%EmptyRange.as.Destroy.impl.Op.49a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_60: <bound method> = bound_method %.loc10_60.2, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_60: %ptr.4e1 = addr_of %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_60(%addr.loc10_60)
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
// CHECK:STDOUT:   %T.049: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.7d6: type = fn_type @EmptyRange.Make, @EmptyRange(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.4cc: %EmptyRange.Make.type.7d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %U: %Copy.type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.d93: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.177(%T.049, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.1f0: %tuple.type.as.Copy.impl.Op.type.d93 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b58: <witness> = impl_witness imports.%Copy.impl_witness_table.943 [concrete]
// CHECK:STDOUT:   %Copy.facet.bfc: %Copy.type = facet_value %C, (%Copy.impl_witness.b58) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e55: <witness> = impl_witness imports.%Copy.impl_witness_table.442, @tuple.type.as.Copy.impl.177(%Copy.facet.bfc, %Copy.facet.bfc) [concrete]
// CHECK:STDOUT:   %Copy.facet.1a8: %Copy.type = facet_value %tuple.type.56b, (%Copy.impl_witness.e55) [concrete]
// CHECK:STDOUT:   %EmptyRange.426: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.480: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.464: %EmptyRange.Make.type.480 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.464, @EmptyRange.Make(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.be4: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.61a: %EmptyRange.as.Iterate.impl.NewCursor.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.859: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.049) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cc1: %EmptyRange.as.Iterate.impl.Next.type.859 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.091: type = fn_type @Optional.HasValue, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.5a0: %Optional.HasValue.type.091 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.618: type = fn_type @Optional.Get, @Optional(%T.049) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9cc: %Optional.Get.type.618 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.999: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.52f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.097: %EmptyRange.as.Iterate.impl.NewCursor.type.52f = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.941: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.7f8: %EmptyRange.as.Iterate.impl.Next.type.941 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.426, (%Iterate.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.dd0: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.097, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.53a: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.681: type = class_type @Optional, @Optional(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.7f8, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.bac: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.d51: %Optional.HasValue.type.bac = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d8d: type = fn_type @Optional.Get, @Optional(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Optional.Get.504: %Optional.Get.type.d8d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.d51, @Optional.HasValue(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.504, @Optional.Get(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ee9: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.de5: %T.as.Destroy.impl.Op.type.ee9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.9f0: type = ptr_type %tuple.type.56b [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.91e: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.e94: %Optional.as.Destroy.impl.Op.type.91e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.015: type = ptr_type %Optional.681 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.456: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.24b: %T.as.Destroy.impl.Op.type.456 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.bd6: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%Copy.facet.1a8) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.90d: %EmptyRange.as.Destroy.impl.Op.type.bd6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.271: type = ptr_type %EmptyRange.426 [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.787: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.7d6) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.4cc)]
// CHECK:STDOUT:   %Core.import_ref.d98: @tuple.type.as.Copy.impl.177.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.d93) = import_ref Core//prelude/parts/copy, loc48_31, loaded [symbolic = @tuple.type.as.Copy.impl.177.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.1f0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.442 = impl_witness_table (%Core.import_ref.d98), @tuple.type.as.Copy.impl.177 [concrete]
// CHECK:STDOUT:   %Main.import_ref.e69: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_33, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.943 = impl_witness_table (%Main.import_ref.e69), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.101 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f30: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.be4) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.61a)]
// CHECK:STDOUT:   %Main.import_ref.881: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.859) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.cc1)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.101, %Main.import_ref.999, %Main.import_ref.8f30, %Main.import_ref.881), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.13f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.091) = import_ref Main//empty_range, inst151 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.5a0)]
// CHECK:STDOUT:   %Main.import_ref.a3b: @Optional.%Optional.Get.type (%Optional.Get.type.618) = import_ref Main//empty_range, inst152 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9cc)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.99e = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.24b = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39)
// CHECK:STDOUT:   %Copy.facet.loc10_41.1: %Copy.type = facet_value constants.%C, (constants.%Copy.impl_witness.b58) [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %.loc10_41.1: %Copy.type = converted constants.%C, %Copy.facet.loc10_41.1 [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %Copy.facet.loc10_41.2: %Copy.type = facet_value constants.%C, (constants.%Copy.impl_witness.b58) [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %.loc10_41.2: %Copy.type = converted constants.%C, %Copy.facet.loc10_41.2 [concrete = constants.%Copy.facet.bfc]
// CHECK:STDOUT:   %Copy.facet.loc10_41.3: %Copy.type = facet_value constants.%tuple.type.56b, (constants.%Copy.impl_witness.e55) [concrete = constants.%Copy.facet.1a8]
// CHECK:STDOUT:   %.loc10_41.3: %Copy.type = converted %.loc10_40, %Copy.facet.loc10_41.3 [concrete = constants.%Copy.facet.1a8]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.1a8) [concrete = constants.%EmptyRange.426]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.480 = specific_constant imports.%Main.import_ref.787, @EmptyRange(constants.%Copy.facet.1a8) [concrete = constants.%EmptyRange.Make.464]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.480 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.464]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.1a8) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.426 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.426 = call %EmptyRange.Make.specific_fn() to %.loc10_48.1
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.426 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.dd0 = impl_witness_access constants.%Iterate.impl_witness.999, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.097]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.1a8) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.426 = bind_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10_49.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.53a = impl_witness_access constants.%Iterate.impl_witness.999, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.7f8]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.1a8) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.681 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.426 = bind_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.681 = call %bound_method.loc10_49.4(%.loc10_48.4, %addr.loc10_49.1) to %.loc10_49.1
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.681 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.bac = specific_constant imports.%Main.import_ref.13f, @Optional(constants.%Copy.facet.1a8) [concrete = constants.%Optional.HasValue.d51]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.bac = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.d51]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.1a8) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.681 = bind_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.d8d = specific_constant imports.%Main.import_ref.a3b, @Optional(constants.%Copy.facet.1a8) [concrete = constants.%Optional.Get.504]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.d8d = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.504]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.1a8) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.56b = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.681 = bind_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.56b = call %bound_method.loc10_49.6(%.loc10_49.9) to %.loc10_49.8
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.56b = temporary %.loc10_49.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %.loc10_49.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %.loc10_49.10, element1
// CHECK:STDOUT:   %C.ref.loc10_12: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_49.11: %C = bind_value %tuple.elem0
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc10_49.11
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_49.12: %C = bind_value %tuple.elem1
// CHECK:STDOUT:   %b: %C = bind_name b, %.loc10_49.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%T.as.Destroy.impl.Op.de5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.7: <bound method> = bound_method %.loc10_49.10, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_49.2: %ptr.9f0 = addr_of %.loc10_49.10
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10_49.1: init %empty_tuple.type = call %bound_method.loc10_49.7(%addr.loc10_49.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_49.2, constants.%Optional.as.Destroy.impl.Op.e94
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.8: <bound method> = bound_method %.loc10_49.2, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_49.3: %ptr.015 = addr_of %.loc10_49.2
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_49.8(%addr.loc10_49.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10_49.2: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.24b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_49.9: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_49.4: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10_49.2: init %empty_tuple.type = call %bound_method.loc10_49.9(%addr.loc10_49.4)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_48.2, constants.%EmptyRange.as.Destroy.impl.Op.90d
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_48: <bound method> = bound_method %.loc10_48.2, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_48: %ptr.271 = addr_of %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_48(%addr.loc10_48)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

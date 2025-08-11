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

class EmptyRange(T:! type) {
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

// --- value.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

class C {}

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

class C {}

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

class C {}

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
class Y {}

fn Run() {
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+7]]:7: error: cannot implicitly convert expression of type `Y` to `X` [ConversionFailure]
  // CHECK:STDERR:   for (x: X in EmptyRange(Y).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+4]]:7: note: type `Y` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   for (x: X in EmptyRange(Y).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (x: X in EmptyRange(Y).Make()) {
  }
}

// CHECK:STDOUT: --- value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.bd2: type = fn_type @EmptyRange.Make, @EmptyRange(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.ed1: %EmptyRange.Make.type.bd2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.cc8: type = class_type @EmptyRange, @EmptyRange(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.557: type = fn_type @EmptyRange.Make, @EmptyRange(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.74d: %EmptyRange.Make.type.557 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.74d, @EmptyRange.Make(%C) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.f5f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.ec1: %EmptyRange.as.Iterate.impl.NewCursor.type.f5f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.264: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.08e: %EmptyRange.as.Iterate.impl.Next.type.264 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f81: type = fn_type @Optional.HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6fd: %Optional.HasValue.type.f81 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.b8f: type = fn_type @Optional.Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9c8: %Optional.Get.type.b8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.52a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.951: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.535: %EmptyRange.as.Iterate.impl.NewCursor.type.951 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.63f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.899: %EmptyRange.as.Iterate.impl.Next.type.63f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.cc8, (%Iterate.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %.2f6: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.535, @EmptyRange.as.Iterate.impl.NewCursor(%C) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.5d5: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.cf0: type = class_type @Optional, @Optional(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.899, @EmptyRange.as.Iterate.impl.Next(%C) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.71d: type = fn_type @Optional.HasValue, @Optional(%C) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.513: %Optional.HasValue.type.71d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.115: type = fn_type @Optional.Get, @Optional(%C) [concrete]
// CHECK:STDOUT:   %Optional.Get.9c1: %Optional.Get.type.115 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.513, @Optional.HasValue(%C) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.9c1, @Optional.Get(%C) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.954: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.793: %Optional.as.Destroy.impl.Op.type.954 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8e6: type = ptr_type %Optional.cf0 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.069: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d5a: %T.as.Destroy.impl.Op.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.a22: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.256: %EmptyRange.as.Destroy.impl.Op.type.a22 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.35d: type = ptr_type %EmptyRange.cc8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6ea: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.bd2) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.ed1)]
// CHECK:STDOUT:   %Main.import_ref.6ce = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.57b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.f5f) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.ec1)]
// CHECK:STDOUT:   %Main.import_ref.170: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.264) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.08e)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.6ce, %Main.import_ref.999, %Main.import_ref.57b, %Main.import_ref.170), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.7f9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f81) = import_ref Main//empty_range, inst138 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6fd)]
// CHECK:STDOUT:   %Main.import_ref.d10: @Optional.%Optional.Get.type (%Optional.Get.type.b8f) = import_ref Main//empty_range, inst139 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9c8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc12_27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%C) [concrete = constants.%EmptyRange.cc8]
// CHECK:STDOUT:   %.loc12_29: %EmptyRange.Make.type.557 = specific_constant imports.%Main.import_ref.6ea, @EmptyRange(constants.%C) [concrete = constants.%EmptyRange.Make.74d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.557 = name_ref Make, %.loc12_29 [concrete = constants.%EmptyRange.Make.74d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%C) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc12_35.1: ref %EmptyRange.cc8 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.cc8 = call %EmptyRange.Make.specific_fn() to %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.2: ref %EmptyRange.cc8 = temporary %.loc12_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.2f6 = impl_witness_access constants.%Iterate.impl_witness.52a, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.535]
// CHECK:STDOUT:   %bound_method.loc12_36.1: <bound method> = bound_method %.loc12_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc12_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%C) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.2: <bound method> = bound_method %.loc12_35.2, %specific_fn.loc12_36.1
// CHECK:STDOUT:   %.loc12_35.3: %EmptyRange.cc8 = bind_value %.loc12_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc12_36.2(%.loc12_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc12_36.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.5d5 = impl_witness_access constants.%Iterate.impl_witness.52a, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.899]
// CHECK:STDOUT:   %bound_method.loc12_36.3: <bound method> = bound_method %.loc12_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc12_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%C) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.4: <bound method> = bound_method %.loc12_35.2, %specific_fn.loc12_36.2
// CHECK:STDOUT:   %.loc12_36.1: ref %Optional.cf0 = temporary_storage
// CHECK:STDOUT:   %.loc12_35.4: %EmptyRange.cc8 = bind_value %.loc12_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.cf0 = call %bound_method.loc12_36.4(%.loc12_35.4, %addr.loc12_36.1) to %.loc12_36.1
// CHECK:STDOUT:   %.loc12_36.2: ref %Optional.cf0 = temporary %.loc12_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc12_36.3: %Optional.HasValue.type.71d = specific_constant imports.%Main.import_ref.7f9, @Optional(constants.%C) [concrete = constants.%Optional.HasValue.513]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.71d = name_ref HasValue, %.loc12_36.3 [concrete = constants.%Optional.HasValue.513]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc12_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%C) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.5: <bound method> = bound_method %.loc12_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc12_36.4: %Optional.cf0 = bind_value %.loc12_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc12_36.5(%.loc12_36.4)
// CHECK:STDOUT:   %.loc12_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc12_36.6: bool = converted %Optional.HasValue.call, %.loc12_36.5
// CHECK:STDOUT:   if %.loc12_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc12_36.7: %Optional.Get.type.115 = specific_constant imports.%Main.import_ref.d10, @Optional(constants.%C) [concrete = constants.%Optional.Get.9c1]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.115 = name_ref Get, %.loc12_36.7 [concrete = constants.%Optional.Get.9c1]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc12_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%C) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.6: <bound method> = bound_method %.loc12_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc12_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc12_36.9: %Optional.cf0 = bind_value %.loc12_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc12_36.6(%.loc12_36.9) to %.loc12_36.8
// CHECK:STDOUT:   %C.ref.loc12_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc12_36.10: ref %C = temporary %.loc12_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc12_36.11: %C = bind_value %.loc12_36.10
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc12_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_36.8, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc12_36.2: %ptr.019 = addr_of %.loc12_36.8
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc12_36.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_36.1, constants.%Optional.as.Destroy.impl.Op.793
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_36.7: <bound method> = bound_method %.loc12_36.1, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_36.3: %ptr.8e6 = addr_of %.loc12_36.1
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_36.7(%addr.loc12_36.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.d5a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_36.8: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_36.4: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_36.8(%addr.loc12_36.4)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_35.1, constants.%EmptyRange.as.Destroy.impl.Op.256
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_35.1, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_35: %ptr.35d = addr_of %.loc12_35.1
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_35(%addr.loc12_35)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.bd2: type = fn_type @EmptyRange.Make, @EmptyRange(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.ed1: %EmptyRange.Make.type.bd2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.cc8: type = class_type @EmptyRange, @EmptyRange(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.557: type = fn_type @EmptyRange.Make, @EmptyRange(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.74d: %EmptyRange.Make.type.557 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.74d, @EmptyRange.Make(%C) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.f5f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.ec1: %EmptyRange.as.Iterate.impl.NewCursor.type.f5f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.264: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.08e: %EmptyRange.as.Iterate.impl.Next.type.264 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f81: type = fn_type @Optional.HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6fd: %Optional.HasValue.type.f81 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.b8f: type = fn_type @Optional.Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9c8: %Optional.Get.type.b8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.52a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.951: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.535: %EmptyRange.as.Iterate.impl.NewCursor.type.951 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.63f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.899: %EmptyRange.as.Iterate.impl.Next.type.63f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.cc8, (%Iterate.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %.2f6: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.535, @EmptyRange.as.Iterate.impl.NewCursor(%C) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.5d5: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.cf0: type = class_type @Optional, @Optional(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.899, @EmptyRange.as.Iterate.impl.Next(%C) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.71d: type = fn_type @Optional.HasValue, @Optional(%C) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.513: %Optional.HasValue.type.71d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.115: type = fn_type @Optional.Get, @Optional(%C) [concrete]
// CHECK:STDOUT:   %Optional.Get.9c1: %Optional.Get.type.115 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.513, @Optional.HasValue(%C) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.9c1, @Optional.Get(%C) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.954: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.793: %Optional.as.Destroy.impl.Op.type.954 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8e6: type = ptr_type %Optional.cf0 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.069: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d5a: %T.as.Destroy.impl.Op.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.a22: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.256: %EmptyRange.as.Destroy.impl.Op.type.a22 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.35d: type = ptr_type %EmptyRange.cc8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6ea: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.bd2) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.ed1)]
// CHECK:STDOUT:   %Main.import_ref.6ce = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.57b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.f5f) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.ec1)]
// CHECK:STDOUT:   %Main.import_ref.170: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.264) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.08e)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.6ce, %Main.import_ref.999, %Main.import_ref.57b, %Main.import_ref.170), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.7f9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f81) = import_ref Main//empty_range, inst138 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6fd)]
// CHECK:STDOUT:   %Main.import_ref.d10: @Optional.%Optional.Get.type (%Optional.Get.type.b8f) = import_ref Main//empty_range, inst139 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9c8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc12_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%C) [concrete = constants.%EmptyRange.cc8]
// CHECK:STDOUT:   %.loc12_33: %EmptyRange.Make.type.557 = specific_constant imports.%Main.import_ref.6ea, @EmptyRange(constants.%C) [concrete = constants.%EmptyRange.Make.74d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.557 = name_ref Make, %.loc12_33 [concrete = constants.%EmptyRange.Make.74d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%C) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc12_39.1: ref %EmptyRange.cc8 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.cc8 = call %EmptyRange.Make.specific_fn() to %.loc12_39.1
// CHECK:STDOUT:   %.loc12_39.2: ref %EmptyRange.cc8 = temporary %.loc12_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.2f6 = impl_witness_access constants.%Iterate.impl_witness.52a, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.535]
// CHECK:STDOUT:   %bound_method.loc12_40.1: <bound method> = bound_method %.loc12_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc12_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%C) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.2: <bound method> = bound_method %.loc12_39.2, %specific_fn.loc12_40.1
// CHECK:STDOUT:   %.loc12_39.3: %EmptyRange.cc8 = bind_value %.loc12_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc12_40.2(%.loc12_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc12_40.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.5d5 = impl_witness_access constants.%Iterate.impl_witness.52a, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.899]
// CHECK:STDOUT:   %bound_method.loc12_40.3: <bound method> = bound_method %.loc12_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc12_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%C) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.4: <bound method> = bound_method %.loc12_39.2, %specific_fn.loc12_40.2
// CHECK:STDOUT:   %.loc12_40.1: ref %Optional.cf0 = temporary_storage
// CHECK:STDOUT:   %.loc12_39.4: %EmptyRange.cc8 = bind_value %.loc12_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.cf0 = call %bound_method.loc12_40.4(%.loc12_39.4, %addr.loc12_40.1) to %.loc12_40.1
// CHECK:STDOUT:   %.loc12_40.2: ref %Optional.cf0 = temporary %.loc12_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc12_40.3: %Optional.HasValue.type.71d = specific_constant imports.%Main.import_ref.7f9, @Optional(constants.%C) [concrete = constants.%Optional.HasValue.513]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.71d = name_ref HasValue, %.loc12_40.3 [concrete = constants.%Optional.HasValue.513]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc12_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%C) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.5: <bound method> = bound_method %.loc12_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc12_40.4: %Optional.cf0 = bind_value %.loc12_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc12_40.5(%.loc12_40.4)
// CHECK:STDOUT:   %.loc12_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc12_40.6: bool = converted %Optional.HasValue.call, %.loc12_40.5
// CHECK:STDOUT:   if %.loc12_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc12_40.7: %Optional.Get.type.115 = specific_constant imports.%Main.import_ref.d10, @Optional(constants.%C) [concrete = constants.%Optional.Get.9c1]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.115 = name_ref Get, %.loc12_40.7 [concrete = constants.%Optional.Get.9c1]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc12_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%C) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.6: <bound method> = bound_method %.loc12_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc12_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc12_40.8: %Optional.cf0 = bind_value %.loc12_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C = call %bound_method.loc12_40.6(%.loc12_40.8) to %.loc12_8
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc12_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc13: %ptr.019 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr.loc13)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc12_8.1: <bound method> = bound_method %.loc12_8, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc12_8.1: %ptr.019 = addr_of %.loc12_8
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc12_8.1: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound.loc12_8.1(%addr.loc12_8.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc12_8.2: <bound method> = bound_method %c.var, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc12_8.2: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc12_8.2: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound.loc12_8.2(%addr.loc12_8.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_40.1, constants.%Optional.as.Destroy.impl.Op.793
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_40.7: <bound method> = bound_method %.loc12_40.1, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_40.2: %ptr.8e6 = addr_of %.loc12_40.1
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_40.7(%addr.loc12_40.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.d5a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_40.8: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_40.3: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_40.8(%addr.loc12_40.3)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_39.1, constants.%EmptyRange.as.Destroy.impl.Op.256
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_39: <bound method> = bound_method %.loc12_39.1, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_39: %ptr.35d = addr_of %.loc12_39.1
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_39(%addr.loc12_39)
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.bd2: type = fn_type @EmptyRange.Make, @EmptyRange(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.ed1: %EmptyRange.Make.type.bd2 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %EmptyRange.2f3: type = class_type @EmptyRange, @EmptyRange(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.4fb: type = fn_type @EmptyRange.Make, @EmptyRange(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.743: %EmptyRange.Make.type.4fb = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.743, @EmptyRange.Make(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.f5f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.ec1: %EmptyRange.as.Iterate.impl.NewCursor.type.f5f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.264: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.08e: %EmptyRange.as.Iterate.impl.Next.type.264 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f81: type = fn_type @Optional.HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6fd: %Optional.HasValue.type.f81 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.b8f: type = fn_type @Optional.Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9c8: %Optional.Get.type.b8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.6db: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.d32: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.763: %EmptyRange.as.Iterate.impl.NewCursor.type.d32 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.482: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.e60: %EmptyRange.as.Iterate.impl.Next.type.482 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.2f3, (%Iterate.impl_witness.6db) [concrete]
// CHECK:STDOUT:   %.2bf: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.763, @EmptyRange.as.Iterate.impl.NewCursor(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.6aa: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.79e: type = class_type @Optional, @Optional(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.e60, @EmptyRange.as.Iterate.impl.Next(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.414: type = fn_type @Optional.HasValue, @Optional(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.4f9: %Optional.HasValue.type.414 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.ebd: type = fn_type @Optional.Get, @Optional(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Optional.Get.d99: %Optional.Get.type.ebd = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b85: type = ptr_type %tuple.type.784 [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.4f9, @Optional.HasValue(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.d99, @Optional.Get(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.973: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.149: %T.as.Destroy.impl.Op.type.973 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.e36: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.060: %Optional.as.Destroy.impl.Op.type.e36 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.07d: type = ptr_type %Optional.79e [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.8b2: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ca6: %T.as.Destroy.impl.Op.type.8b2 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.eb6: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%tuple.type.784) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.fb4: %EmptyRange.as.Destroy.impl.Op.type.eb6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5cf: type = ptr_type %EmptyRange.2f3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6ea: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.bd2) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.ed1)]
// CHECK:STDOUT:   %Main.import_ref.6ce = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.57b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.f5f) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.ec1)]
// CHECK:STDOUT:   %Main.import_ref.170: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.264) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.08e)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.6ce, %Main.import_ref.999, %Main.import_ref.57b, %Main.import_ref.170), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.7f9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f81) = import_ref Main//empty_range, inst138 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6fd)]
// CHECK:STDOUT:   %Main.import_ref.d10: @Optional.%Optional.Get.type (%Optional.Get.type.b8f) = import_ref Main//empty_range, inst139 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9c8)]
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
// CHECK:STDOUT:   %.loc10_53.5: type = converted %.loc10_52, constants.%tuple.type.784 [concrete = constants.%tuple.type.784]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%tuple.type.784) [concrete = constants.%EmptyRange.2f3]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.4fb = specific_constant imports.%Main.import_ref.6ea, @EmptyRange(constants.%tuple.type.784) [concrete = constants.%EmptyRange.Make.743]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.4fb = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.743]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%tuple.type.784) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.2f3 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.2f3 = call %EmptyRange.Make.specific_fn() to %.loc10_60.1
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.2f3 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.2bf = impl_witness_access constants.%Iterate.impl_witness.6db, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.763]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%tuple.type.784) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.2f3 = bind_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc10_61.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.6aa = impl_witness_access constants.%Iterate.impl_witness.6db, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.e60]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%tuple.type.784) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.79e = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.2f3 = bind_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.79e = call %bound_method.loc10_61.4(%.loc10_60.4, %addr.loc10_61.1) to %.loc10_61.1
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.79e = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.414 = specific_constant imports.%Main.import_ref.7f9, @Optional(constants.%tuple.type.784) [concrete = constants.%Optional.HasValue.4f9]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.414 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.4f9]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%tuple.type.784) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.79e = bind_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.ebd = specific_constant imports.%Main.import_ref.d10, @Optional(constants.%tuple.type.784) [concrete = constants.%Optional.Get.d99]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.ebd = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.d99]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%tuple.type.784) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.79e = bind_value %.loc10_61.2
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
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.8, constants.%T.as.Destroy.impl.Op.149
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.7: <bound method> = bound_method %.loc10_61.8, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_61.2: %ptr.b85 = addr_of %.loc10_61.8
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10_61.1: init %empty_tuple.type = call %bound_method.loc10_61.7(%addr.loc10_61.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_61.1, constants.%Optional.as.Destroy.impl.Op.060
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.8: <bound method> = bound_method %.loc10_61.1, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_61.3: %ptr.07d = addr_of %.loc10_61.1
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_61.8(%addr.loc10_61.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10_61.2: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.ca6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_61.9: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_61.4: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10_61.2: init %empty_tuple.type = call %bound_method.loc10_61.9(%addr.loc10_61.4)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_60.1, constants.%EmptyRange.as.Destroy.impl.Op.fb4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_60: <bound method> = bound_method %.loc10_60.1, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10_60: %ptr.5cf = addr_of %.loc10_60.1
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_60(%addr.loc10_60)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.56b: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.99e: type = pattern_type %tuple.type.56b [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.bd2: type = fn_type @EmptyRange.Make, @EmptyRange(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.ed1: %EmptyRange.Make.type.bd2 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %EmptyRange.90a: type = class_type @EmptyRange, @EmptyRange(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.829: type = fn_type @EmptyRange.Make, @EmptyRange(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.d4f: %EmptyRange.Make.type.829 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.d4f, @EmptyRange.Make(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.f5f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.ec1: %EmptyRange.as.Iterate.impl.NewCursor.type.f5f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.264: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.08e: %EmptyRange.as.Iterate.impl.Next.type.264 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f81: type = fn_type @Optional.HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6fd: %Optional.HasValue.type.f81 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.b8f: type = fn_type @Optional.Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.Get.9c8: %Optional.Get.type.b8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.627: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.59c: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.263: %EmptyRange.as.Iterate.impl.NewCursor.type.59c = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.70f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.852: %EmptyRange.as.Iterate.impl.Next.type.70f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.90a, (%Iterate.impl_witness.627) [concrete]
// CHECK:STDOUT:   %.8bd: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.263, @EmptyRange.as.Iterate.impl.NewCursor(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %.8d0: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.657: type = class_type @Optional, @Optional(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.852, @EmptyRange.as.Iterate.impl.Next(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.2f1: type = fn_type @Optional.HasValue, @Optional(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.bef: %Optional.HasValue.type.2f1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.4a4: type = fn_type @Optional.Get, @Optional(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Optional.Get.8a7: %Optional.Get.type.4a4 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.bef, @Optional.HasValue(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.8a7, @Optional.Get(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.12e: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.34b: %T.as.Destroy.impl.Op.type.12e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.9f0: type = ptr_type %tuple.type.56b [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.type.e58: type = fn_type @Optional.as.Destroy.impl.Op, @Optional.as.Destroy.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.7d3: %Optional.as.Destroy.impl.Op.type.e58 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.036: type = ptr_type %Optional.657 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.069: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d5a: %T.as.Destroy.impl.Op.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.type.310: type = fn_type @EmptyRange.as.Destroy.impl.Op, @EmptyRange.as.Destroy.impl(%tuple.type.56b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.e9a: %EmptyRange.as.Destroy.impl.Op.type.310 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f9b: type = ptr_type %EmptyRange.90a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6ea: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.bd2) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.ed1)]
// CHECK:STDOUT:   %Main.import_ref.6ce = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.57b: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.f5f) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.ec1)]
// CHECK:STDOUT:   %Main.import_ref.170: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.264) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.08e)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.6ce, %Main.import_ref.999, %Main.import_ref.57b, %Main.import_ref.170), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.7f9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.f81) = import_ref Main//empty_range, inst138 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6fd)]
// CHECK:STDOUT:   %Main.import_ref.d10: @Optional.%Optional.Get.type (%Optional.Get.type.b8f) = import_ref Main//empty_range, inst139 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.9c8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c48 = binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc12_19: %pattern_type.99e = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc12_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_39: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc12_40: %tuple.type.24b = tuple_literal (%C.ref.loc12_36, %C.ref.loc12_39)
// CHECK:STDOUT:   %.loc12_41: type = converted %.loc12_40, constants.%tuple.type.56b [concrete = constants.%tuple.type.56b]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%tuple.type.56b) [concrete = constants.%EmptyRange.90a]
// CHECK:STDOUT:   %.loc12_42: %EmptyRange.Make.type.829 = specific_constant imports.%Main.import_ref.6ea, @EmptyRange(constants.%tuple.type.56b) [concrete = constants.%EmptyRange.Make.d4f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.829 = name_ref Make, %.loc12_42 [concrete = constants.%EmptyRange.Make.d4f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%tuple.type.56b) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc12_48.1: ref %EmptyRange.90a = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.90a = call %EmptyRange.Make.specific_fn() to %.loc12_48.1
// CHECK:STDOUT:   %.loc12_48.2: ref %EmptyRange.90a = temporary %.loc12_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.8bd = impl_witness_access constants.%Iterate.impl_witness.627, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.263]
// CHECK:STDOUT:   %bound_method.loc12_49.1: <bound method> = bound_method %.loc12_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc12_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%tuple.type.56b) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_49.2: <bound method> = bound_method %.loc12_48.2, %specific_fn.loc12_49.1
// CHECK:STDOUT:   %.loc12_48.3: %EmptyRange.90a = bind_value %.loc12_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc12_49.2(%.loc12_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc12_49.1: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.8d0 = impl_witness_access constants.%Iterate.impl_witness.627, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.852]
// CHECK:STDOUT:   %bound_method.loc12_49.3: <bound method> = bound_method %.loc12_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc12_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%tuple.type.56b) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_49.4: <bound method> = bound_method %.loc12_48.2, %specific_fn.loc12_49.2
// CHECK:STDOUT:   %.loc12_49.1: ref %Optional.657 = temporary_storage
// CHECK:STDOUT:   %.loc12_48.4: %EmptyRange.90a = bind_value %.loc12_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.657 = call %bound_method.loc12_49.4(%.loc12_48.4, %addr.loc12_49.1) to %.loc12_49.1
// CHECK:STDOUT:   %.loc12_49.2: ref %Optional.657 = temporary %.loc12_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc12_49.3: %Optional.HasValue.type.2f1 = specific_constant imports.%Main.import_ref.7f9, @Optional(constants.%tuple.type.56b) [concrete = constants.%Optional.HasValue.bef]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.2f1 = name_ref HasValue, %.loc12_49.3 [concrete = constants.%Optional.HasValue.bef]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc12_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%tuple.type.56b) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_49.5: <bound method> = bound_method %.loc12_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc12_49.4: %Optional.657 = bind_value %.loc12_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc12_49.5(%.loc12_49.4)
// CHECK:STDOUT:   %.loc12_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc12_49.6: bool = converted %Optional.HasValue.call, %.loc12_49.5
// CHECK:STDOUT:   if %.loc12_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc12_49.7: %Optional.Get.type.4a4 = specific_constant imports.%Main.import_ref.d10, @Optional(constants.%tuple.type.56b) [concrete = constants.%Optional.Get.8a7]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.4a4 = name_ref Get, %.loc12_49.7 [concrete = constants.%Optional.Get.8a7]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc12_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%tuple.type.56b) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_49.6: <bound method> = bound_method %.loc12_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc12_49.8: ref %tuple.type.56b = temporary_storage
// CHECK:STDOUT:   %.loc12_49.9: %Optional.657 = bind_value %.loc12_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.56b = call %bound_method.loc12_49.6(%.loc12_49.9) to %.loc12_49.8
// CHECK:STDOUT:   %.loc12_49.10: ref %tuple.type.56b = temporary %.loc12_49.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %.loc12_49.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %.loc12_49.10, element1
// CHECK:STDOUT:   %C.ref.loc12_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc12_49.11: %C = bind_value %tuple.elem0
// CHECK:STDOUT:   %a: %C = bind_name a, %.loc12_49.11
// CHECK:STDOUT:   %C.ref.loc12_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc12_49.12: %C = bind_value %tuple.elem1
// CHECK:STDOUT:   %b: %C = bind_name b, %.loc12_49.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc12_49.1: <bound method> = bound_method %.loc12_49.8, constants.%T.as.Destroy.impl.Op.34b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_49.7: <bound method> = bound_method %.loc12_49.8, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc12_49.2: %ptr.9f0 = addr_of %.loc12_49.8
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc12_49.1: init %empty_tuple.type = call %bound_method.loc12_49.7(%addr.loc12_49.2)
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_49.1, constants.%Optional.as.Destroy.impl.Op.7d3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_49.8: <bound method> = bound_method %.loc12_49.1, %Optional.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_49.3: %ptr.036 = addr_of %.loc12_49.1
// CHECK:STDOUT:   %Optional.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_49.8(%addr.loc12_49.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc12_49.2: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.d5a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_49.9: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc12_49.4: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc12_49.2: init %empty_tuple.type = call %bound_method.loc12_49.9(%addr.loc12_49.4)
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_48.1, constants.%EmptyRange.as.Destroy.impl.Op.e9a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_48: <bound method> = bound_method %.loc12_48.1, %EmptyRange.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12_48: %ptr.f9b = addr_of %.loc12_48.1
// CHECK:STDOUT:   %EmptyRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_48(%addr.loc12_48)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

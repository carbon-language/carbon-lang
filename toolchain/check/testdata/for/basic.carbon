// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/for.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/for/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/for/basic.carbon

// --- fail_not_range.carbon

library "[[@TEST_NAME]]";

fn Run() {
  // TODO: These diagnostics could be better. If nothing else, we should only diagnose once.
  // CHECK:STDERR: fail_not_range.carbon:[[@LINE+8]]:7: error: cannot access member of interface `Core.Iterate` in type `{}` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (c: {} in {}) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_not_range.carbon:[[@LINE+4]]:7: error: cannot access member of interface `Core.Iterate` in type `{}` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (c: {} in {}) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~
  // CHECK:STDERR:
  for (c: {} in {}) {
  }
}

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

// --- simple.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

class C {}

fn Body(c: C);
fn AfterLoop();

fn Run() {
  //@dump-sem-ir-begin
  for (c: C in EmptyRange(C).Make()) {
    Body(c);
  }
  AfterLoop();
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

// CHECK:STDOUT: --- simple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %AfterLoop.type: type = fn_type @AfterLoop [concrete]
// CHECK:STDOUT:   %AfterLoop: %AfterLoop.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Make.type.bd2: type = fn_type @Make, @EmptyRange(%T) [symbolic]
// CHECK:STDOUT:   %Make.ed1: %Make.type.bd2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.cc8: type = class_type @EmptyRange, @EmptyRange(%C) [concrete]
// CHECK:STDOUT:   %Make.type.557: type = fn_type @Make, @EmptyRange(%C) [concrete]
// CHECK:STDOUT:   %Make.74d: %Make.type.557 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.74d, @Make(%C) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %NewCursor.type.427: type = fn_type @NewCursor.1 [concrete]
// CHECK:STDOUT:   %NewCursor.type.f5f: type = fn_type @NewCursor.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %NewCursor.ec1: %NewCursor.type.f5f = struct_value () [symbolic]
// CHECK:STDOUT:   %Next.type.264: type = fn_type @Next.1, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Next.08e: %Next.type.264 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasValue.type.f81: type = fn_type @HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %HasValue.6fd: %HasValue.type.f81 = struct_value () [symbolic]
// CHECK:STDOUT:   %Get.type.b8f: type = fn_type @Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Get.9c8: %Get.type.b8f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.52a: <witness> = impl_witness imports.%Iterate.impl_witness_table, @impl(%C) [concrete]
// CHECK:STDOUT:   %NewCursor.type.951: type = fn_type @NewCursor.2, @impl(%C) [concrete]
// CHECK:STDOUT:   %NewCursor.535: %NewCursor.type.951 = struct_value () [concrete]
// CHECK:STDOUT:   %Next.type.63f: type = fn_type @Next.1, @impl(%C) [concrete]
// CHECK:STDOUT:   %Next.899: %Next.type.63f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.cc8, (%Iterate.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %.2f6: type = fn_type_with_self_type %NewCursor.type.427, %Iterate.facet [concrete]
// CHECK:STDOUT:   %NewCursor.specific_fn: <specific function> = specific_function %NewCursor.535, @NewCursor.2(%C) [concrete]
// CHECK:STDOUT:   %Next.type.941: type = fn_type @Next.2 [concrete]
// CHECK:STDOUT:   %.5d5: type = fn_type_with_self_type %Next.type.941, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.cf0: type = class_type @Optional, @Optional(%C) [concrete]
// CHECK:STDOUT:   %Next.specific_fn: <specific function> = specific_function %Next.899, @Next.1(%C) [concrete]
// CHECK:STDOUT:   %HasValue.type.71d: type = fn_type @HasValue, @Optional(%C) [concrete]
// CHECK:STDOUT:   %HasValue.513: %HasValue.type.71d = struct_value () [concrete]
// CHECK:STDOUT:   %Get.type.115: type = fn_type @Get, @Optional(%C) [concrete]
// CHECK:STDOUT:   %Get.9c1: %Get.type.115 = struct_value () [concrete]
// CHECK:STDOUT:   %HasValue.specific_fn: <specific function> = specific_function %HasValue.513, @HasValue(%C) [concrete]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.9c1, @Get(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6ea: @EmptyRange.%Make.type (%Make.type.bd2) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%Make (constants.%Make.ed1)]
// CHECK:STDOUT:   %Main.import_ref.6ce = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.57b: @impl.%NewCursor.type (%NewCursor.type.f5f) = import_ref Main//empty_range, loc8_38, loaded [symbolic = @impl.%NewCursor (constants.%NewCursor.ec1)]
// CHECK:STDOUT:   %Main.import_ref.170: @impl.%Next.type (%Next.type.264) = import_ref Main//empty_range, loc11_58, loaded [symbolic = @impl.%Next (constants.%Next.08e)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.6ce, %Main.import_ref.999, %Main.import_ref.57b, %Main.import_ref.170), @impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.7f9: @Optional.%HasValue.type (%HasValue.type.f81) = import_ref Main//empty_range, inst136 [indirect], loaded [symbolic = @Optional.%HasValue (constants.%HasValue.6fd)]
// CHECK:STDOUT:   %Main.import_ref.d10: @Optional.%Get.type (%Get.type.b8f) = import_ref Main//empty_range, inst137 [indirect], loaded [symbolic = @Optional.%Get (constants.%Get.9c8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc13_27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%C) [concrete = constants.%EmptyRange.cc8]
// CHECK:STDOUT:   %.loc13_29: %Make.type.557 = specific_constant imports.%Main.import_ref.6ea, @EmptyRange(constants.%C) [concrete = constants.%Make.74d]
// CHECK:STDOUT:   %Make.ref: %Make.type.557 = name_ref Make, %.loc13_29 [concrete = constants.%Make.74d]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%C) [concrete = constants.%Make.specific_fn]
// CHECK:STDOUT:   %.loc13_35.1: ref %EmptyRange.cc8 = temporary_storage
// CHECK:STDOUT:   %Make.call: init %EmptyRange.cc8 = call %Make.specific_fn() to %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.2: ref %EmptyRange.cc8 = temporary %.loc13_35.1, %Make.call
// CHECK:STDOUT:   %impl.elem2: %.2f6 = impl_witness_access constants.%Iterate.impl_witness.52a, element2 [concrete = constants.%NewCursor.535]
// CHECK:STDOUT:   %bound_method.loc13_36.1: <bound method> = bound_method %.loc13_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc13_36.1: <specific function> = specific_function %impl.elem2, @NewCursor.2(constants.%C) [concrete = constants.%NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_36.2: <bound method> = bound_method %.loc13_35.2, %specific_fn.loc13_36.1
// CHECK:STDOUT:   %.loc13_35.3: %EmptyRange.cc8 = bind_value %.loc13_35.2
// CHECK:STDOUT:   %NewCursor.call: init %empty_struct_type = call %bound_method.loc13_36.2(%.loc13_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var invalid
// CHECK:STDOUT:   assign %var, %NewCursor.call
// CHECK:STDOUT:   br !.loc13_36.1
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc13_36.1:
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.5d5 = impl_witness_access constants.%Iterate.impl_witness.52a, element3 [concrete = constants.%Next.899]
// CHECK:STDOUT:   %bound_method.loc13_36.3: <bound method> = bound_method %.loc13_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc13_36.2: <specific function> = specific_function %impl.elem3, @Next.1(constants.%C) [concrete = constants.%Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_36.4: <bound method> = bound_method %.loc13_35.2, %specific_fn.loc13_36.2
// CHECK:STDOUT:   %.loc13_36.1: ref %Optional.cf0 = temporary_storage
// CHECK:STDOUT:   %.loc13_35.4: %EmptyRange.cc8 = bind_value %.loc13_35.2
// CHECK:STDOUT:   %Next.call: init %Optional.cf0 = call %bound_method.loc13_36.4(%.loc13_35.4, %addr) to %.loc13_36.1
// CHECK:STDOUT:   %.loc13_36.2: ref %Optional.cf0 = temporary %.loc13_36.1, %Next.call
// CHECK:STDOUT:   %.loc13_36.3: %HasValue.type.71d = specific_constant imports.%Main.import_ref.7f9, @Optional(constants.%C) [concrete = constants.%HasValue.513]
// CHECK:STDOUT:   %HasValue.ref: %HasValue.type.71d = name_ref HasValue, %.loc13_36.3 [concrete = constants.%HasValue.513]
// CHECK:STDOUT:   %HasValue.bound: <bound method> = bound_method %.loc13_36.2, %HasValue.ref
// CHECK:STDOUT:   %HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @HasValue(constants.%C) [concrete = constants.%HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_36.5: <bound method> = bound_method %.loc13_36.2, %HasValue.specific_fn
// CHECK:STDOUT:   %.loc13_36.4: %Optional.cf0 = bind_value %.loc13_36.2
// CHECK:STDOUT:   %HasValue.call: init bool = call %bound_method.loc13_36.5(%.loc13_36.4)
// CHECK:STDOUT:   %.loc13_36.5: bool = value_of_initializer %HasValue.call
// CHECK:STDOUT:   %.loc13_36.6: bool = converted %HasValue.call, %.loc13_36.5
// CHECK:STDOUT:   if %.loc13_36.6 br !.loc13_36.2 else br !.loc13_36.3
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc13_36.2:
// CHECK:STDOUT:   %.loc13_36.7: %Get.type.115 = specific_constant imports.%Main.import_ref.d10, @Optional(constants.%C) [concrete = constants.%Get.9c1]
// CHECK:STDOUT:   %Get.ref: %Get.type.115 = name_ref Get, %.loc13_36.7 [concrete = constants.%Get.9c1]
// CHECK:STDOUT:   %Get.bound: <bound method> = bound_method %.loc13_36.2, %Get.ref
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%C) [concrete = constants.%Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_36.6: <bound method> = bound_method %.loc13_36.2, %Get.specific_fn
// CHECK:STDOUT:   %.loc13_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_36.9: %Optional.cf0 = bind_value %.loc13_36.2
// CHECK:STDOUT:   %Get.call: init %C = call %bound_method.loc13_36.6(%.loc13_36.9) to %.loc13_36.8
// CHECK:STDOUT:   %C.ref.loc13_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc13_36.10: ref %C = temporary %.loc13_36.8, %Get.call
// CHECK:STDOUT:   %c: ref %C = bind_name c, %.loc13_36.10
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc14: %C = bind_value %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%.loc14)
// CHECK:STDOUT:   br !.loc13_36.1
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc13_36.3:
// CHECK:STDOUT:   %AfterLoop.ref: %AfterLoop.type = name_ref AfterLoop, file.%AfterLoop.decl [concrete = constants.%AfterLoop]
// CHECK:STDOUT:   %AfterLoop.call: init %empty_tuple.type = call %AfterLoop.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

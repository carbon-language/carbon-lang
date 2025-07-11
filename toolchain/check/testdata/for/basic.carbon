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

// --- trivial.carbon

library "[[@TEST_NAME]]";

class TrivialRange {
  impl as Core.Iterate where .CursorType = () and .ElementType = () {
    fn NewCursor[self: Self]() {}
    fn Next[self: Self](cursor: ()*) -> Core.Optional(()) {
      return Core.Optional(()).None();
    }
  }
}

fn Body();
fn AfterLoop();

fn Run() {
  //@dump-sem-ir-begin
  for (_: () in {} as TrivialRange) {
    Body();
  }
  AfterLoop();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TrivialRange: type = class_type @TrivialRange [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NewCursor.type.427: type = fn_type @NewCursor.1 [concrete]
// CHECK:STDOUT:   %Next.type.941: type = fn_type @Next.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %HasValue.type.a15: type = fn_type @HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %HasValue.73f: %HasValue.type.a15 = struct_value () [symbolic]
// CHECK:STDOUT:   %Get.type.e03: type = fn_type @Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Get.971: %Get.type.e03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @TrivialRange.%Iterate.impl_witness_table [concrete]
// CHECK:STDOUT:   %NewCursor.type.bd2331.1: type = fn_type @NewCursor.2 [concrete]
// CHECK:STDOUT:   %NewCursor.c20a1c.1: %NewCursor.type.bd2331.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Optional.f9a: type = class_type @Optional, @Optional(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Next.type.9f0: type = fn_type @Next.2 [concrete]
// CHECK:STDOUT:   %Next.029: %Next.type.9f0 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.d50: %Iterate.type = facet_value %TrivialRange, (%Iterate.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %NewCursor.type.bd2331.2: type = fn_type @NewCursor.3 [concrete]
// CHECK:STDOUT:   %NewCursor.c20a1c.2: %NewCursor.type.bd2331.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasValue.type.b7a: type = fn_type @HasValue, @Optional(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %HasValue.ada: %HasValue.type.b7a = struct_value () [concrete]
// CHECK:STDOUT:   %Get.type.130: type = fn_type @Get, @Optional(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Get.6e8: %Get.type.130 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.7e4: type = fn_type @Op.2, @Destroy.impl(%Optional.f9a) [concrete]
// CHECK:STDOUT:   %Op.466: %Op.type.7e4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.511: type = ptr_type %Optional.f9a [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %AfterLoop.type: type = fn_type @AfterLoop [concrete]
// CHECK:STDOUT:   %AfterLoop: %AfterLoop.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialRange.val: %TrivialRange = struct_value () [concrete]
// CHECK:STDOUT:   %.510: type = fn_type_with_self_type %NewCursor.type.427, %Iterate.facet.d50 [concrete]
// CHECK:STDOUT:   %.708: type = fn_type_with_self_type %Next.type.941, %Iterate.facet.d50 [concrete]
// CHECK:STDOUT:   %HasValue.specific_fn: <specific function> = specific_function %HasValue.ada, @HasValue(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.6e8, @Get(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.type.8d6: type = fn_type @Op.2, @Destroy.impl(%TrivialRange) [concrete]
// CHECK:STDOUT:   %Op.90d: %Op.type.8d6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.41d: type = ptr_type %TrivialRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Optional.%HasValue.type (%HasValue.type.a15) = import_ref Core//prelude/parts/iterate, inst98 [indirect], loaded [symbolic = @Optional.%HasValue (constants.%HasValue.73f)]
// CHECK:STDOUT:   %Core.import_ref.4fd: @Optional.%Get.type (%Get.type.e03) = import_ref Core//prelude/parts/iterate, inst99 [indirect], loaded [symbolic = @Optional.%Get (constants.%Get.971)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.cb1 = binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %TrivialRange.ref: type = name_ref TrivialRange, file.%TrivialRange.decl [concrete = constants.%TrivialRange]
// CHECK:STDOUT:   %.loc18_18.2: ref %TrivialRange = temporary_storage
// CHECK:STDOUT:   %.loc18_18.3: init %TrivialRange = class_init (), %.loc18_18.2 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %.loc18_18.4: ref %TrivialRange = temporary %.loc18_18.2, %.loc18_18.3
// CHECK:STDOUT:   %.loc18_20.1: ref %TrivialRange = converted %.loc18_18.1, %.loc18_18.4
// CHECK:STDOUT:   %impl.elem2: %.510 = impl_witness_access constants.%Iterate.impl_witness, element2 [concrete = constants.%NewCursor.c20a1c.2]
// CHECK:STDOUT:   %bound_method.loc18_35.1: <bound method> = bound_method %.loc18_20.1, %impl.elem2
// CHECK:STDOUT:   %.loc18_20.2: %TrivialRange = bind_value %.loc18_20.1
// CHECK:STDOUT:   %NewCursor.ref: %NewCursor.type.bd2331.1 = name_ref NewCursor, @Iterate.impl.%NewCursor.decl.loc6_32.1 [concrete = constants.%NewCursor.c20a1c.1]
// CHECK:STDOUT:   %NewCursor.bound: <bound method> = bound_method %.loc18_20.2, %NewCursor.ref
// CHECK:STDOUT:   %NewCursor.call: init %empty_tuple.type = call %NewCursor.bound(%.loc18_20.2)
// CHECK:STDOUT:   %var: ref %empty_tuple.type = var invalid
// CHECK:STDOUT:   assign %var, %NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc18_35.1: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.708 = impl_witness_access constants.%Iterate.impl_witness, element3 [concrete = constants.%Next.029]
// CHECK:STDOUT:   %bound_method.loc18_35.2: <bound method> = bound_method %.loc18_20.1, %impl.elem3
// CHECK:STDOUT:   %.loc18_35.1: ref %Optional.f9a = temporary_storage
// CHECK:STDOUT:   %.loc18_20.3: %TrivialRange = bind_value %.loc18_20.1
// CHECK:STDOUT:   %Next.call: init %Optional.f9a = call %bound_method.loc18_35.2(%.loc18_20.3, %addr.loc18_35.1) to %.loc18_35.1
// CHECK:STDOUT:   %.loc18_35.2: ref %Optional.f9a = temporary %.loc18_35.1, %Next.call
// CHECK:STDOUT:   %.loc18_35.3: %HasValue.type.b7a = specific_constant imports.%Core.import_ref.cd6, @Optional(constants.%empty_tuple.type) [concrete = constants.%HasValue.ada]
// CHECK:STDOUT:   %HasValue.ref: %HasValue.type.b7a = name_ref HasValue, %.loc18_35.3 [concrete = constants.%HasValue.ada]
// CHECK:STDOUT:   %HasValue.bound: <bound method> = bound_method %.loc18_35.2, %HasValue.ref
// CHECK:STDOUT:   %HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @HasValue(constants.%empty_tuple.type) [concrete = constants.%HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.3: <bound method> = bound_method %.loc18_35.2, %HasValue.specific_fn
// CHECK:STDOUT:   %.loc18_35.4: %Optional.f9a = bind_value %.loc18_35.2
// CHECK:STDOUT:   %HasValue.call: init bool = call %bound_method.loc18_35.3(%.loc18_35.4)
// CHECK:STDOUT:   %.loc18_35.5: bool = value_of_initializer %HasValue.call
// CHECK:STDOUT:   %.loc18_35.6: bool = converted %HasValue.call, %.loc18_35.5
// CHECK:STDOUT:   if %.loc18_35.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc18_35.7: %Get.type.130 = specific_constant imports.%Core.import_ref.4fd, @Optional(constants.%empty_tuple.type) [concrete = constants.%Get.6e8]
// CHECK:STDOUT:   %Get.ref: %Get.type.130 = name_ref Get, %.loc18_35.7 [concrete = constants.%Get.6e8]
// CHECK:STDOUT:   %Get.bound: <bound method> = bound_method %.loc18_35.2, %Get.ref
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%empty_tuple.type) [concrete = constants.%Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.4: <bound method> = bound_method %.loc18_35.2, %Get.specific_fn
// CHECK:STDOUT:   %.loc18_35.8: %Optional.f9a = bind_value %.loc18_35.2
// CHECK:STDOUT:   %Get.call: init %empty_tuple.type = call %bound_method.loc18_35.4(%.loc18_35.8)
// CHECK:STDOUT:   %.loc18_12.1: type = splice_block %.loc18_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_12.3: type = converted %.loc18_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_35.9: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc18_35.10: ref %empty_tuple.type = temporary %.loc18_35.9, %Get.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_35.11: %empty_tuple.type = converted %Get.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %_: %empty_tuple.type = bind_name _, %.loc18_35.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref()
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %AfterLoop.ref: %AfterLoop.type = name_ref AfterLoop, file.%AfterLoop.decl [concrete = constants.%AfterLoop]
// CHECK:STDOUT:   %AfterLoop.call: init %empty_tuple.type = call %AfterLoop.ref()
// CHECK:STDOUT:   %Op.bound.loc18_35.1: <bound method> = bound_method %.loc18_35.9, constants.%Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.5: <bound method> = bound_method %.loc18_35.9, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc18_35.2: %ptr.843 = addr_of %.loc18_35.9
// CHECK:STDOUT:   %no_op.loc18_35.1: init %empty_tuple.type = call %bound_method.loc18_35.5(%addr.loc18_35.2)
// CHECK:STDOUT:   %Op.bound.loc18_35.2: <bound method> = bound_method %.loc18_35.1, constants.%Op.466
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.6: <bound method> = bound_method %.loc18_35.1, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc18_35.3: %ptr.511 = addr_of %.loc18_35.1
// CHECK:STDOUT:   %no_op.loc18_35.2: init %empty_tuple.type = call %bound_method.loc18_35.6(%addr.loc18_35.3)
// CHECK:STDOUT:   %Op.bound.loc18_35.3: <bound method> = bound_method %var, constants.%Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.7: <bound method> = bound_method %var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc18_35.4: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %no_op.loc18_35.3: init %empty_tuple.type = call %bound_method.loc18_35.7(%addr.loc18_35.4)
// CHECK:STDOUT:   %Op.bound.loc18_18: <bound method> = bound_method %.loc18_18.2, constants.%Op.90d
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_18: <bound method> = bound_method %.loc18_18.2, %Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc18_18: %ptr.41d = addr_of %.loc18_18.2
// CHECK:STDOUT:   %no_op.loc18_18: init %empty_tuple.type = call %bound_method.loc18_18(%addr.loc18_18)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

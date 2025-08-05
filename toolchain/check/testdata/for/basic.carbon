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
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.a15: type = fn_type @Optional.HasValue, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.73f: %Optional.HasValue.type.a15 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.e03: type = fn_type @Optional.Get, @Optional(%T) [symbolic]
// CHECK:STDOUT:   %Optional.Get.971: %Optional.Get.type.e03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @TrivialRange.%Iterate.impl_witness_table [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.bd2331.1: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_32.1 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.c20a1c.1: %TrivialRange.as.Iterate.impl.NewCursor.type.bd2331.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Optional.f9a: type = class_type @Optional, @Optional(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.type: type = fn_type @TrivialRange.as.Iterate.impl.Next [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next: %TrivialRange.as.Iterate.impl.Next.type = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.d50: %Iterate.type = facet_value %TrivialRange, (%Iterate.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.bd2331.2: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_32.2 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.c20a1c.2: %TrivialRange.as.Iterate.impl.NewCursor.type.bd2331.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.41d: type = ptr_type %TrivialRange [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Destroy.impl.Op.type: type = fn_type @TrivialRange.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Destroy.impl.Op: %TrivialRange.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.b7a: type = fn_type @Optional.HasValue, @Optional(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ada: %Optional.HasValue.type.b7a = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.130: type = fn_type @Optional.Get, @Optional(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Optional.Get.6e8: %Optional.Get.type.130 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.7e4: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%Optional.f9a) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.466: %T.as.Destroy.impl.Op.type.7e4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.511: type = ptr_type %Optional.f9a [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.a63: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ea3: %T.as.Destroy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %AfterLoop.type: type = fn_type @AfterLoop [concrete]
// CHECK:STDOUT:   %AfterLoop: %AfterLoop.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialRange.val: %TrivialRange = struct_value () [concrete]
// CHECK:STDOUT:   %.510: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet.d50 [concrete]
// CHECK:STDOUT:   %.708: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet.d50 [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ada, @Optional.HasValue(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.6e8, @Optional.Get(%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.a15) = import_ref Core//prelude/parts/iterate, inst98 [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.73f)]
// CHECK:STDOUT:   %Core.import_ref.4fd: @Optional.%Optional.Get.type (%Optional.Get.type.e03) = import_ref Core//prelude/parts/iterate, inst99 [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.971)]
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
// CHECK:STDOUT:   %impl.elem2: %.510 = impl_witness_access constants.%Iterate.impl_witness, element2 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.c20a1c.2]
// CHECK:STDOUT:   %bound_method.loc18_35.1: <bound method> = bound_method %.loc18_20.1, %impl.elem2
// CHECK:STDOUT:   %.loc18_20.2: %TrivialRange = bind_value %.loc18_20.1
// CHECK:STDOUT:   %NewCursor.ref: %TrivialRange.as.Iterate.impl.NewCursor.type.bd2331.1 = name_ref NewCursor, @TrivialRange.as.Iterate.impl.%TrivialRange.as.Iterate.impl.NewCursor.decl.loc6_32.1 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.c20a1c.1]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound: <bound method> = bound_method %.loc18_20.2, %NewCursor.ref
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.call: init %empty_tuple.type = call %TrivialRange.as.Iterate.impl.NewCursor.bound(%.loc18_20.2)
// CHECK:STDOUT:   %var: ref %empty_tuple.type = var invalid
// CHECK:STDOUT:   assign %var, %TrivialRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc18_35.1: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.708 = impl_witness_access constants.%Iterate.impl_witness, element3 [concrete = constants.%TrivialRange.as.Iterate.impl.Next]
// CHECK:STDOUT:   %bound_method.loc18_35.2: <bound method> = bound_method %.loc18_20.1, %impl.elem3
// CHECK:STDOUT:   %.loc18_35.1: ref %Optional.f9a = temporary_storage
// CHECK:STDOUT:   %.loc18_20.3: %TrivialRange = bind_value %.loc18_20.1
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.call: init %Optional.f9a = call %bound_method.loc18_35.2(%.loc18_20.3, %addr.loc18_35.1) to %.loc18_35.1
// CHECK:STDOUT:   %.loc18_35.2: ref %Optional.f9a = temporary %.loc18_35.1, %TrivialRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc18_35.3: %Optional.HasValue.type.b7a = specific_constant imports.%Core.import_ref.cd6, @Optional(constants.%empty_tuple.type) [concrete = constants.%Optional.HasValue.ada]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.b7a = name_ref HasValue, %.loc18_35.3 [concrete = constants.%Optional.HasValue.ada]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc18_35.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%empty_tuple.type) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.3: <bound method> = bound_method %.loc18_35.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc18_35.4: %Optional.f9a = bind_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc18_35.3(%.loc18_35.4)
// CHECK:STDOUT:   %.loc18_35.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc18_35.6: bool = converted %Optional.HasValue.call, %.loc18_35.5
// CHECK:STDOUT:   if %.loc18_35.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc18_35.7: %Optional.Get.type.130 = specific_constant imports.%Core.import_ref.4fd, @Optional(constants.%empty_tuple.type) [concrete = constants.%Optional.Get.6e8]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.130 = name_ref Get, %.loc18_35.7 [concrete = constants.%Optional.Get.6e8]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc18_35.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%empty_tuple.type) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.4: <bound method> = bound_method %.loc18_35.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc18_35.8: %Optional.f9a = bind_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.Get.call: init %empty_tuple.type = call %bound_method.loc18_35.4(%.loc18_35.8)
// CHECK:STDOUT:   %.loc18_12.1: type = splice_block %.loc18_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_12.3: type = converted %.loc18_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_35.9: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc18_35.10: ref %empty_tuple.type = temporary %.loc18_35.9, %Optional.Get.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_35.11: %empty_tuple.type = converted %Optional.Get.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %_: %empty_tuple.type = bind_name _, %.loc18_35.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref()
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %AfterLoop.ref: %AfterLoop.type = name_ref AfterLoop, file.%AfterLoop.decl [concrete = constants.%AfterLoop]
// CHECK:STDOUT:   %AfterLoop.call: init %empty_tuple.type = call %AfterLoop.ref()
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18_35.1: <bound method> = bound_method %.loc18_35.9, constants.%T.as.Destroy.impl.Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.5: <bound method> = bound_method %.loc18_35.9, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc18_35.2: %ptr.843 = addr_of %.loc18_35.9
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18_35.1: init %empty_tuple.type = call %bound_method.loc18_35.5(%addr.loc18_35.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18_35.2: <bound method> = bound_method %.loc18_35.1, constants.%T.as.Destroy.impl.Op.466
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.6: <bound method> = bound_method %.loc18_35.1, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc18_35.3: %ptr.511 = addr_of %.loc18_35.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18_35.2: init %empty_tuple.type = call %bound_method.loc18_35.6(%addr.loc18_35.3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18_35.3: <bound method> = bound_method %var, constants.%T.as.Destroy.impl.Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.7: <bound method> = bound_method %var, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc18_35.4: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18_35.3: init %empty_tuple.type = call %bound_method.loc18_35.7(%addr.loc18_35.4)
// CHECK:STDOUT:   %TrivialRange.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc18_18.2, constants.%TrivialRange.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc18_18: %ptr.41d = addr_of %.loc18_18.2
// CHECK:STDOUT:   %TrivialRange.as.Destroy.impl.Op.call: init %empty_tuple.type = call %TrivialRange.as.Destroy.impl.Op.bound(%addr.loc18_18)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

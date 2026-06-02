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
  // CHECK:STDERR:   for (unused c: {} in {}) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_not_range.carbon:[[@LINE+4]]:7: error: cannot access member of interface `Core.Iterate` in type `{}` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (unused c: {} in {}) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (unused c: {} in {}) {
  }
}

// --- trivial.carbon

library "[[@TEST_NAME]]";

class TrivialRange {
  impl as Core.Iterate where .CursorType = () and .ElementType = () {
    fn NewCursor[unused self: Self]() {}
    fn Next[unused self: Self](unused cursor: ()*) -> Core.Optional(()) {
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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f9d: type = fn_type @Optional.Get, @Optional(%T.f49) [symbolic]
// CHECK:STDOUT:   %Optional.Get.09b: %Optional.Get.type.f9d = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.2b3: type = fn_type @Optional.HasValue, @Optional(%T.f49) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.fed: %Optional.HasValue.type.2b3 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9a4: <witness> = impl_witness imports.%Copy.impl_witness_table.bfa [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %empty_tuple.type, (%Copy.impl_witness.9a4) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @TrivialRange.as.Iterate.impl.%Iterate.impl_witness_table [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.d8a5e3.1: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_39.1 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bf18e5.1: %TrivialRange.as.Iterate.impl.NewCursor.type.d8a5e3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Optional.23a: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.type: type = fn_type @TrivialRange.as.Iterate.impl.Next [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next: %TrivialRange.as.Iterate.impl.Next.type = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %TrivialRange, (%Iterate.impl_witness) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.adc: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.045: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.d8a5e3.2: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_39.2 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bf18e5.2: %TrivialRange.as.Iterate.impl.NewCursor.type.d8a5e3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.952: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c0d: %Optional.HasValue.type.952 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.28b: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.7e1: %Optional.Get.type.28b = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.da2: type = struct_type {.has_value: bool, .value: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %AfterLoop.type: type = fn_type @AfterLoop [concrete]
// CHECK:STDOUT:   %AfterLoop: %AfterLoop.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialRange.val: %TrivialRange = struct_value () [concrete]
// CHECK:STDOUT:   %.8ab: ref %TrivialRange = temporary invalid, %TrivialRange.val [concrete]
// CHECK:STDOUT:   %.c0f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.adc, %Iterate.facet [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound.da2: <bound method> = bound_method %.8ab, %TrivialRange.as.Iterate.impl.NewCursor.bf18e5.2 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound.862: <bound method> = bound_method %TrivialRange.val, %TrivialRange.as.Iterate.impl.NewCursor.bf18e5.1 [concrete]
// CHECK:STDOUT:   %.c28: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.045, %Iterate.facet [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.bound: <bound method> = bound_method %.8ab, %TrivialRange.as.Iterate.impl.Next [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.c0d, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.7e1, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.1: type = fn_type @Destroy.Op.loc18_35.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.1: %Destroy.Op.type.790b3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc18_35.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc18_20.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.8ab, %Destroy.Op.2dd605.6 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Copy.impl.Op.type: type = fn_type @empty_tuple.type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Copy.impl.Op: %empty_tuple.type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cc8: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.2b3) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.fed)]
// CHECK:STDOUT:   %Core.import_ref.dac: @Optional.%Optional.Get.type (%Optional.Get.type.f9d) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.09b)]
// CHECK:STDOUT:   %Core.import_ref.f63: %empty_tuple.type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%empty_tuple.type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bfa = impl_witness_table (%Core.import_ref.f63), @empty_tuple.type.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.cb1 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_18.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %TrivialRange.ref: type = name_ref TrivialRange, file.%TrivialRange.decl [concrete = constants.%TrivialRange]
// CHECK:STDOUT:   %.loc18_18.2: ref %TrivialRange = temporary_storage
// CHECK:STDOUT:   %.loc18_18.3: init %TrivialRange to %.loc18_18.2 = class_init () [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %.loc18_20.1: init %TrivialRange = converted %.loc18_18.1, %.loc18_18.3 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %.loc18_20.2: ref %TrivialRange = temporary %.loc18_18.2, %.loc18_20.1 [concrete = constants.%.8ab]
// CHECK:STDOUT:   %impl.elem2: %.c0f = impl_witness_access constants.%Iterate.impl_witness, element2 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.bf18e5.2]
// CHECK:STDOUT:   %bound_method.loc18_35.1: <bound method> = bound_method %.loc18_20.2, %impl.elem2 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.bound.da2]
// CHECK:STDOUT:   %.loc18_20.3: %TrivialRange = acquire_value %.loc18_20.2 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %NewCursor.ref: %TrivialRange.as.Iterate.impl.NewCursor.type.d8a5e3.1 = name_ref NewCursor, @TrivialRange.as.Iterate.impl.%TrivialRange.as.Iterate.impl.NewCursor.decl.loc6_39.1 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.bf18e5.1]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound: <bound method> = bound_method %.loc18_20.3, %NewCursor.ref [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.bound.862]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.call: init %empty_tuple.type = call %TrivialRange.as.Iterate.impl.NewCursor.bound(%.loc18_20.3)
// CHECK:STDOUT:   %var: ref %empty_tuple.type = var invalid
// CHECK:STDOUT:   assign %var, %TrivialRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.c28 = impl_witness_access constants.%Iterate.impl_witness, element3 [concrete = constants.%TrivialRange.as.Iterate.impl.Next]
// CHECK:STDOUT:   %bound_method.loc18_35.2: <bound method> = bound_method %.loc18_20.2, %impl.elem3 [concrete = constants.%TrivialRange.as.Iterate.impl.Next.bound]
// CHECK:STDOUT:   %.loc18_35.1: ref %Optional.23a = temporary_storage
// CHECK:STDOUT:   %.loc18_20.4: %TrivialRange = acquire_value %.loc18_20.2 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.call: init %Optional.23a to %.loc18_35.1 = call %bound_method.loc18_35.2(%.loc18_20.4, %addr)
// CHECK:STDOUT:   %.loc18_35.2: ref %Optional.23a = temporary %.loc18_35.1, %TrivialRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc18_35.3: %Optional.HasValue.type.952 = specific_constant imports.%Core.import_ref.cc8, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.c0d]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.952 = name_ref HasValue, %.loc18_35.3 [concrete = constants.%Optional.HasValue.c0d]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc18_35.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.3: <bound method> = bound_method %.loc18_35.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc18_35.4: %Optional.23a = acquire_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc18_35.3(%.loc18_35.4)
// CHECK:STDOUT:   %.loc18_35.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc18_35.6: bool = converted %Optional.HasValue.call, %.loc18_35.5
// CHECK:STDOUT:   if %.loc18_35.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc18_35.7: %Optional.Get.type.28b = specific_constant imports.%Core.import_ref.dac, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.7e1]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.28b = name_ref Get, %.loc18_35.7 [concrete = constants.%Optional.Get.7e1]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc18_35.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.4: <bound method> = bound_method %.loc18_35.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc18_35.8: %Optional.23a = acquire_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.Get.call: init %empty_tuple.type = call %bound_method.loc18_35.4(%.loc18_35.8)
// CHECK:STDOUT:   %.loc18_12.1: type = splice_block %.loc18_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_12.3: type = converted %.loc18_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_35.9: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc18_35.10: ref %empty_tuple.type = temporary %.loc18_35.9, %Optional.Get.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_35.11: %empty_tuple.type = converted %Optional.Get.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %_: %empty_tuple.type = value_binding _, %.loc18_35.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref()
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %AfterLoop.ref: %AfterLoop.type = name_ref AfterLoop, file.%AfterLoop.decl [concrete = constants.%AfterLoop]
// CHECK:STDOUT:   %AfterLoop.call: init %empty_tuple.type = call %AfterLoop.ref()
// CHECK:STDOUT:   %Destroy.Op.bound.loc18_35.1: <bound method> = bound_method %.loc18_35.10, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc18_35.1: init %empty_tuple.type = call %Destroy.Op.bound.loc18_35.1(%.loc18_35.10)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18_35.2: <bound method> = bound_method %.loc18_35.2, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call.loc18_35.2: init %empty_tuple.type = call %Destroy.Op.bound.loc18_35.2(%.loc18_35.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18_35.3: <bound method> = bound_method %var, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc18_35.3: init %empty_tuple.type = call %Destroy.Op.bound.loc18_35.3(%var)
// CHECK:STDOUT:   %Destroy.Op.call.loc18_20: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.8ab)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_35.1(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_35.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_35.3(%self.param: ref %struct_type.has_value.value.da2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_35.4(%self.param: ref %Optional.23a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_20.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_20.2(%self.param: ref %TrivialRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.417: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c9a: <witness> = impl_witness imports.%Copy.impl_witness_table.955 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %empty_tuple.type, (%Copy.impl_witness.c9a) [concrete]
// CHECK:STDOUT:   %Iterate.NewCursor.type: type = fn_type @Iterate.NewCursor [concrete]
// CHECK:STDOUT:   %Iterate.Next.type: type = fn_type @Iterate.Next [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.fdd: type = fn_type @Optional.HasValue, @Optional(%T.417) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.01a: %Optional.HasValue.type.fdd = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.e67: type = fn_type @Optional.Get, @Optional(%T.417) [symbolic]
// CHECK:STDOUT:   %Optional.Get.47b: %Optional.Get.type.e67 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @TrivialRange.%Iterate.impl_witness_table [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.fa48a5.1: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_32.1 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.c82dff.1: %TrivialRange.as.Iterate.impl.NewCursor.type.fa48a5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Optional.68c: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.type: type = fn_type @TrivialRange.as.Iterate.impl.Next [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next: %TrivialRange.as.Iterate.impl.Next.type = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %TrivialRange, (%Iterate.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.fa48a5.2: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_32.2 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.c82dff.2: %TrivialRange.as.Iterate.impl.NewCursor.type.fa48a5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.cac: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.5e3: %Optional.HasValue.type.cac = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.a4e: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.1d7: %Optional.Get.type.a4e = struct_value () [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %AfterLoop.type: type = fn_type @AfterLoop [concrete]
// CHECK:STDOUT:   %AfterLoop: %AfterLoop.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialRange.val: %TrivialRange = struct_value () [concrete]
// CHECK:STDOUT:   %.799: type = fn_type_with_self_type %Iterate.NewCursor.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %.d92: type = fn_type_with_self_type %Iterate.Next.type, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.5e3, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.1d7, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.ff9: %type_where = facet_value %empty_tuple.type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bb8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.ff9) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.72f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bb8 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.136: %type_where = facet_value %Optional.68c, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.37e: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.136) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.84c: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.37e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.17e: type = ptr_type %Optional.68c [concrete]
// CHECK:STDOUT:   %facet_value.441: %type_where = facet_value %TrivialRange, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.56c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.441) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a5a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.56c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.41d: type = ptr_type %TrivialRange [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Copy.impl.Op.type: type = fn_type @empty_tuple.type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Copy.impl.Op: %empty_tuple.type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.1a3: %empty_tuple.type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%empty_tuple.type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.955 = impl_witness_table (%Core.import_ref.1a3), @empty_tuple.type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.654: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.fdd) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.01a)]
// CHECK:STDOUT:   %Core.import_ref.f5c: @Optional.%Optional.Get.type (%Optional.Get.type.e67) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.47b)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.cb1 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %TrivialRange.ref: type = name_ref TrivialRange, file.%TrivialRange.decl [concrete = constants.%TrivialRange]
// CHECK:STDOUT:   %.loc18_18.2: ref %TrivialRange = temporary_storage
// CHECK:STDOUT:   %.loc18_18.3: init %TrivialRange = class_init (), %.loc18_18.2 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %.loc18_18.4: ref %TrivialRange = temporary %.loc18_18.2, %.loc18_18.3
// CHECK:STDOUT:   %.loc18_20.1: ref %TrivialRange = converted %.loc18_18.1, %.loc18_18.4
// CHECK:STDOUT:   %impl.elem2: %.799 = impl_witness_access constants.%Iterate.impl_witness, element2 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.c82dff.2]
// CHECK:STDOUT:   %bound_method.loc18_35.1: <bound method> = bound_method %.loc18_20.1, %impl.elem2
// CHECK:STDOUT:   %.loc18_20.2: %TrivialRange = acquire_value %.loc18_20.1
// CHECK:STDOUT:   %NewCursor.ref: %TrivialRange.as.Iterate.impl.NewCursor.type.fa48a5.1 = name_ref NewCursor, @TrivialRange.as.Iterate.impl.%TrivialRange.as.Iterate.impl.NewCursor.decl.loc6_32.1 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.c82dff.1]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound: <bound method> = bound_method %.loc18_20.2, %NewCursor.ref
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.call: init %empty_tuple.type = call %TrivialRange.as.Iterate.impl.NewCursor.bound(%.loc18_20.2)
// CHECK:STDOUT:   %var: ref %empty_tuple.type = var invalid
// CHECK:STDOUT:   assign %var, %TrivialRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc18_35.1: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.d92 = impl_witness_access constants.%Iterate.impl_witness, element3 [concrete = constants.%TrivialRange.as.Iterate.impl.Next]
// CHECK:STDOUT:   %bound_method.loc18_35.2: <bound method> = bound_method %.loc18_20.1, %impl.elem3
// CHECK:STDOUT:   %.loc18_35.1: ref %Optional.68c = temporary_storage
// CHECK:STDOUT:   %.loc18_20.3: %TrivialRange = acquire_value %.loc18_20.1
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.call: init %Optional.68c = call %bound_method.loc18_35.2(%.loc18_20.3, %addr.loc18_35.1) to %.loc18_35.1
// CHECK:STDOUT:   %.loc18_35.2: ref %Optional.68c = temporary %.loc18_35.1, %TrivialRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc18_35.3: %Optional.HasValue.type.cac = specific_constant imports.%Core.import_ref.654, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.5e3]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.cac = name_ref HasValue, %.loc18_35.3 [concrete = constants.%Optional.HasValue.5e3]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc18_35.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.3: <bound method> = bound_method %.loc18_35.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc18_35.4: %Optional.68c = acquire_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc18_35.3(%.loc18_35.4)
// CHECK:STDOUT:   %.loc18_35.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc18_35.6: bool = converted %Optional.HasValue.call, %.loc18_35.5
// CHECK:STDOUT:   if %.loc18_35.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc18_35.7: %Optional.Get.type.a4e = specific_constant imports.%Core.import_ref.f5c, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.1d7]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.a4e = name_ref Get, %.loc18_35.7 [concrete = constants.%Optional.Get.1d7]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc18_35.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.4: <bound method> = bound_method %.loc18_35.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc18_35.8: %Optional.68c = acquire_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.Get.call: init %empty_tuple.type = call %bound_method.loc18_35.4(%.loc18_35.8)
// CHECK:STDOUT:   %.loc18_12.1: type = splice_block %.loc18_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_12.2: %empty_tuple.type = tuple_literal ()
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc18_35.1: <bound method> = bound_method %.loc18_35.10, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.5: <bound method> = bound_method %.loc18_35.10, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc18_35.2: %ptr.843 = addr_of %.loc18_35.10
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc18_35.1: init %empty_tuple.type = call %bound_method.loc18_35.5(%addr.loc18_35.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc18_35.2: <bound method> = bound_method %.loc18_35.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.84c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.6: <bound method> = bound_method %.loc18_35.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc18_35.3: %ptr.17e = addr_of %.loc18_35.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc18_35.2: init %empty_tuple.type = call %bound_method.loc18_35.6(%addr.loc18_35.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc18_35.3: <bound method> = bound_method %var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_35.7: <bound method> = bound_method %var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc18_35.4: %ptr.843 = addr_of %var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc18_35.3: init %empty_tuple.type = call %bound_method.loc18_35.7(%addr.loc18_35.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc18_18: <bound method> = bound_method %.loc18_18.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a5a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18_18: <bound method> = bound_method %.loc18_18.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc18_18: %ptr.41d = addr_of %.loc18_18.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc18_18: init %empty_tuple.type = call %bound_method.loc18_18(%addr.loc18_18)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

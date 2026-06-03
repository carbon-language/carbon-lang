// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_access.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

class Class(T:! Core.Copy) {
  var x: T;

  fn Get[self: Self]() -> T {
    //@dump-sem-ir-begin
    return self.x;
    //@dump-sem-ir-end
  }

  fn GetAddr[ref self: Self]() -> T* {
    //@dump-sem-ir-begin
    return &self.x;
    //@dump-sem-ir-end
  }
}

fn DirectFieldAccess(x: Class(i32)) -> i32 {
  //@dump-sem-ir-begin
  return x.x;
  //@dump-sem-ir-end
}

fn MethodCall(x: Class(i32)) -> i32 {
  //@dump-sem-ir-begin
  return x.Get();
  //@dump-sem-ir-end
}

fn AddrMethodCall(p: Class(i32)*) -> i32 {
  //@dump-sem-ir-begin
  return *p->GetAddr();
  //@dump-sem-ir-end
}

// --- static_member_fn_call.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  fn Make() -> Class(T) { return {}; }
}

fn StaticMemberFunctionCall(T:! type) -> Class(T) {
  //@dump-sem-ir-begin
  return Class(T).Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.98f: type = class_type @Class, @Class(%T.f49) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f49 [symbolic]
// CHECK:STDOUT:   %require_complete.77e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.c0fdca.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.a2d: type = unbound_element_type %Class.98f, %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.16a: type = pattern_type %Class.98f [symbolic]
// CHECK:STDOUT:   %.dedae6.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.44e: type = fn_type @Class.Get, @Class(%T.f49) [symbolic]
// CHECK:STDOUT:   %Class.Get.b53: %Class.Get.type.44e = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.a3f: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.0e7: Core.Form = init_form %ptr.a3f [symbolic]
// CHECK:STDOUT:   %pattern_type.eec: type = pattern_type %ptr.a3f [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.60b: type = fn_type @Class.GetAddr, @Class(%T.f49) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.2a7: %Class.GetAddr.type.60b = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.419: type = struct_type {.x: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.dd3: <witness> = complete_type_witness %struct_type.x.419 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2dd: <witness> = lookup_impl_witness %T.f49, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e9f10f.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f49) [symbolic]
// CHECK:STDOUT:   %.e16: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e9f10f.2, %T.f49 [symbolic]
// CHECK:STDOUT:   %impl.elem0.41e: %.e16 = impl_witness_access %Copy.lookup_impl_witness.2dd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7ec: <specific function> = specific_impl_function %impl.elem0.41e, @Copy.WithSelf.Op(%T.f49) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.fd9: <witness> = lookup_impl_witness %ptr.a3f, @Copy [symbolic]
// CHECK:STDOUT:   %.7ac: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet.d43: %Copy.type = facet_value %ptr.a3f, (%Copy.lookup_impl_witness.fd9) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d43) [symbolic]
// CHECK:STDOUT:   %.93e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4e8, %Copy.facet.d43 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d58: %.93e = impl_witness_access %Copy.lookup_impl_witness.fd9, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.436: <specific function> = specific_impl_function %impl.elem0.d58, @Copy.WithSelf.Op(%Copy.facet.d43) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.14d: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Class.b6c: type = class_type @Class, @Class(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %pattern_type.b80: type = pattern_type %Class.b6c [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.760: type = unbound_element_type %Class.b6c, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.46c: type = fn_type @Class.Get, @Class(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Class.Get.494: %Class.Get.type.46c = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.f08: type = fn_type @Class.GetAddr, @Class(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.dfd: %Class.GetAddr.type.f08 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.06a: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.9a8: <witness> = complete_type_witness %struct_type.x.06a [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet.14d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.494, @Class.Get(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %ptr.67a: type = ptr_type %Class.b6c [concrete]
// CHECK:STDOUT:   %ptr.847: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.060: Core.Form = init_form %ptr.847 [concrete]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %ptr.847 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.dfd, @Class.GetAddr(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %complete_type.34c: <witness> = complete_type_witness %ptr.847 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c37: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ae4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e4: %ptr.as.Copy.impl.Op.type.ae4 = struct_value () [concrete]
// CHECK:STDOUT:   %.b35: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.17a: %Copy.type = facet_value %ptr.847, (%Copy.impl_witness.c37) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.165: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.17a) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.165, %Copy.facet.17a [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e4, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc18_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.98f
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     .Get = %Class.Get.decl
// CHECK:STDOUT:     .GetAddr = %Class.GetAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Get(@Class.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_27.1 [symbolic = %Class.elem (constants.%Class.elem.a2d)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc7_27.1 [symbolic = %require_complete.loc9 (constants.%require_complete.77e)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e9f10f.2)]
// CHECK:STDOUT:   %.loc9_16.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_16.3 (constants.%.e16)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2dd)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.3 (%.e16) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.98f)) -> out %return.param: @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.98f) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.a2d) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.3 (%.e16) = impl_witness_access constants.%Copy.lookup_impl_witness.2dd, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.WithSelf.Op(constants.%T.f49) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:     %bound_method.loc9_16.2: <bound method> = bound_method %.loc9_16.2, %specific_impl_fn.loc9_16.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) to %.loc7_27.1 = call %bound_method.loc9_16.2(%.loc9_16.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc13_36.1 [symbolic = %Class.elem (constants.%Class.elem.a2d)]
// CHECK:STDOUT:   %.loc15_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc13_36.1) [symbolic = %.loc15_12.1 (constants.%.7ac)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.fd9)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d43)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.4e8)]
// CHECK:STDOUT:   %.loc15_12.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc15_12.2 (constants.%.93e)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.2 (%.93e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.d58)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.436)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class.98f)) -> out %return.param: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.a3f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.98f) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.a2d) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.as_type.loc13_36.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.a3f) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.2 (%.93e) = impl_witness_access constants.%Copy.lookup_impl_witness.fd9, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.d58)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.d43) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.436)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc13_36.1 (%ptr.a3f) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.b6c) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.b6c = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.760 = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.b6c) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.b6c = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.46c = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.14d) [concrete = constants.%Class.Get.494]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.46c = name_ref Get, %.loc28 [concrete = constants.%Class.Get.494]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.14d) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.67a) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.67a = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.b6c = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.f08 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.14d) [concrete = constants.%Class.GetAddr.dfd]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.f08 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.dfd]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.14d) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.847 = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.847 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.847 = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc34_10.1: <bound method> = bound_method %.loc34_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_10.2: <bound method> = bound_method %.loc34_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc34_10.2(%.loc34_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.f49) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f49
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.77e
// CHECK:STDOUT:   %Class => constants.%Class.98f
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.a2d
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.44e
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.b53
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.60b
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.2a7
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.419
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.dd3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.f49) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %Class => constants.%Class.98f
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.16a
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.dedae6.2
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.c0fdca.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f49) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %Class => constants.%Class.98f
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.16a
// CHECK:STDOUT:   %T.as_type.loc13_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.a3f
// CHECK:STDOUT:   %.loc13_36.1 => constants.%.0e7
// CHECK:STDOUT:   %pattern_type.loc13_36 => constants.%pattern_type.eec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.14d) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.14d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.b6c
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.760
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.46c
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.494
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.f08
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.dfd
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.06a
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.9a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.14d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.14d
// CHECK:STDOUT:   %Class => constants.%Class.b6c
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.b80
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.7ea
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.104
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.9a8
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.760
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc9_16.3 => constants.%.604
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1a4
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.ce7
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.14d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.14d
// CHECK:STDOUT:   %Class => constants.%Class.b6c
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.b80
// CHECK:STDOUT:   %T.as_type.loc13_36.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.847
// CHECK:STDOUT:   %.loc13_36.1 => constants.%.060
// CHECK:STDOUT:   %pattern_type.loc13_36 => constants.%pattern_type.7b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_22 => constants.%complete_type.9a8
// CHECK:STDOUT:   %require_complete.loc13_32 => constants.%complete_type.34c
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.760
// CHECK:STDOUT:   %.loc15_12.1 => constants.%.b35
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.c37
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.17a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.165
// CHECK:STDOUT:   %.loc15_12.2 => constants.%.cd7
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.6e4
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.188: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.f8f: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Make: %Class.Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.specific_fn: <specific function> = specific_function %Class.Make, @Class.Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8_30.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc8_49.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T.loc8_30.1) [symbolic = %Class.Make.type (constants.%Class.Make.type)]
// CHECK:STDOUT:   %Class.Make: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = struct_value () [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:   %Class.Make.specific_fn.loc10_18.2: <specific function> = specific_function %Class.Make, @Class.Make(%T.loc8_30.1) [symbolic = %Class.Make.specific_fn.loc10_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc10: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_30.2 [symbolic = %T.loc8_30.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc10: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.1 (constants.%Class)]
// CHECK:STDOUT:     %.loc10: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = specific_constant @Class.%Class.Make.decl, @Class(constants.%T) [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = name_ref Make, %.loc10 [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Class.Make.specific_fn.loc10_18.1: <specific function> = specific_function %Make.ref, @Class.Make(constants.%T) [symbolic = %Class.Make.specific_fn.loc10_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Class.Make.call: init @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) to %.loc8_49.1 = call %Class.Make.specific_fn.loc10_18.1()
// CHECK:STDOUT:     return %Class.Make.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.loc8_30.1 => constants.%T
// CHECK:STDOUT:   %Class.loc8_49.1 => constants.%Class
// CHECK:STDOUT:   %.loc8_49.2 => constants.%.188
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8f
// CHECK:STDOUT: }
// CHECK:STDOUT:

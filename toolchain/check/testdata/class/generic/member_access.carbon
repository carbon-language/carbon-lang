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
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.338: type = class_type @Class, @Class(%T.228) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.228 [symbolic]
// CHECK:STDOUT:   %require_complete.c28: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.4b69d0.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.735: type = unbound_element_type %Class.338, %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.0a7: type = pattern_type %Class.338 [symbolic]
// CHECK:STDOUT:   %.9ff809.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.e5b: type = fn_type @Class.Get, @Class(%T.228) [symbolic]
// CHECK:STDOUT:   %Class.Get.af1: %Class.Get.type.e5b = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.812: Core.Form = init_form %ptr.432 [symbolic]
// CHECK:STDOUT:   %pattern_type.c61: type = pattern_type %ptr.432 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.ef5: type = fn_type @Class.GetAddr, @Class(%T.228) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.d6e: %Class.GetAddr.type.ef5 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.f20: type = struct_type {.x: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.f15: <witness> = complete_type_witness %struct_type.x.f20 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c49: <witness> = lookup_impl_witness %T.228, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2ec0c.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.228) [symbolic]
// CHECK:STDOUT:   %.12f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2ec0c.2, %T.228 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cbd: %.12f = impl_witness_access %Copy.lookup_impl_witness.c49, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c5f: <specific function> = specific_impl_function %impl.elem0.cbd, @Copy.WithSelf.Op(%T.228) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.923: <witness> = lookup_impl_witness %ptr.432, @Copy [symbolic]
// CHECK:STDOUT:   %.3dd: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet.e44: %Copy.type = facet_value %ptr.432, (%Copy.lookup_impl_witness.923) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6da: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.e44) [symbolic]
// CHECK:STDOUT:   %.a4a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6da, %Copy.facet.e44 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d87: %.a4a = impl_witness_access %Copy.lookup_impl_witness.923, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.728: <specific function> = specific_impl_function %impl.elem0.d87, @Copy.WithSelf.Op(%Copy.facet.e44) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.66a: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Class.f02: type = class_type @Class, @Class(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %pattern_type.e44: type = pattern_type %Class.f02 [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.1b2: type = unbound_element_type %Class.f02, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.8a3: type = fn_type @Class.Get, @Class(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %Class.Get.bd7: %Class.Get.type.8a3 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.f59: type = fn_type @Class.GetAddr, @Class(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.f38: %Class.GetAddr.type.f59 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.2bb: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.x.2bb [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet.66a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.bd7, @Class.Get(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %ptr.ea9: type = ptr_type %Class.f02 [concrete]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.793: Core.Form = init_form %ptr.143 [concrete]
// CHECK:STDOUT:   %pattern_type.663: type = pattern_type %ptr.143 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.f38, @Class.GetAddr(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %complete_type.76e: <witness> = complete_type_witness %ptr.143 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.45b: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.eac: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e2: %ptr.as.Copy.impl.Op.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %.064: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.425: %Copy.type = facet_value %ptr.143, (%Copy.impl_witness.45b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.425) [concrete]
// CHECK:STDOUT:   %.948: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72c, %Copy.facet.425 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e2, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Class.338
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_27.1 [symbolic = %Class.elem (constants.%Class.elem.735)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc7_27.1 [symbolic = %require_complete.loc9 (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2ec0c.2)]
// CHECK:STDOUT:   %.loc9_16.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_16.3 (constants.%.12f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c49)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.3 (%.12f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.338)) -> out %return.param: @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.338) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.735) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.3 (%.12f) = impl_witness_access constants.%Copy.lookup_impl_witness.c49, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.WithSelf.Op(constants.%T.228) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.c5f)]
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc13_36.1 [symbolic = %Class.elem (constants.%Class.elem.735)]
// CHECK:STDOUT:   %.loc15_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc13_36.1) [symbolic = %.loc15_12.1 (constants.%.3dd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.923)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.e44)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.6da)]
// CHECK:STDOUT:   %.loc15_12.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc15_12.2 (constants.%.a4a)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.2 (%.a4a) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.d87)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.728)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class.338)) -> out %return.param: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.432) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.338) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.735) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.as_type.loc13_36.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.432) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.2 (%.a4a) = impl_witness_access constants.%Copy.lookup_impl_witness.923, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.d87)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.e44) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.728)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc13_36.1 (%ptr.432) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.f02) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.f02 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.1b2 = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.f02) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.f02 = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.8a3 = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.66a) [concrete = constants.%Class.Get.bd7]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.8a3 = name_ref Get, %.loc28 [concrete = constants.%Class.Get.bd7]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.66a) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.ea9) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.ea9 = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.f02 = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.f59 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.66a) [concrete = constants.%Class.GetAddr.f38]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.f59 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.f38]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.66a) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.143 = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.143 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.143 = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc34_10.1: <bound method> = bound_method %.loc34_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_10.2: <bound method> = bound_method %.loc34_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc34_10.2(%.loc34_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.228) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.228
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c28
// CHECK:STDOUT:   %Class => constants.%Class.338
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.735
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.e5b
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.af1
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.ef5
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.d6e
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.f20
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.f15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.228) {
// CHECK:STDOUT:   %T => constants.%T.228
// CHECK:STDOUT:   %Class => constants.%Class.338
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.0a7
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.9ff809.2
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.4b69d0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.228) {
// CHECK:STDOUT:   %T => constants.%T.228
// CHECK:STDOUT:   %Class => constants.%Class.338
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.0a7
// CHECK:STDOUT:   %T.as_type.loc13_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.432
// CHECK:STDOUT:   %.loc13_36.1 => constants.%.812
// CHECK:STDOUT:   %pattern_type.loc13_36 => constants.%pattern_type.c61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.66a) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.66a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.f02
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.1b2
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.8a3
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.bd7
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.f59
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.f38
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2bb
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.fa5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.66a) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.66a
// CHECK:STDOUT:   %Class => constants.%Class.f02
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.e44
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.70c
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.c6b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.fa5
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.1b2
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.afe
// CHECK:STDOUT:   %.loc9_16.3 => constants.%.a5f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.32d
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.c4a
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.66a) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.66a
// CHECK:STDOUT:   %Class => constants.%Class.f02
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.e44
// CHECK:STDOUT:   %T.as_type.loc13_36.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.143
// CHECK:STDOUT:   %.loc13_36.1 => constants.%.793
// CHECK:STDOUT:   %pattern_type.loc13_36 => constants.%pattern_type.663
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_22 => constants.%complete_type.fa5
// CHECK:STDOUT:   %require_complete.loc13_32 => constants.%complete_type.76e
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.1b2
// CHECK:STDOUT:   %.loc15_12.1 => constants.%.064
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.45b
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.425
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.72c
// CHECK:STDOUT:   %.loc15_12.2 => constants.%.948
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.6e2
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
// CHECK:STDOUT:   %.ac7: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.5e3: type = pattern_type %Class [symbolic]
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
// CHECK:STDOUT:   %.loc8_49.2 => constants.%.ac7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5e3
// CHECK:STDOUT: }
// CHECK:STDOUT:

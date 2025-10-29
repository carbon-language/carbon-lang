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

  fn GetAddr[addr self: Self*]() -> T* {
    //@dump-sem-ir-begin
    return &self->x;
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
// CHECK:STDOUT:   %T.417: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.32e: type = class_type @Class, @Class(%T.417) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.417 [symbolic]
// CHECK:STDOUT:   %require_complete.d74: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.bc9: type = unbound_element_type %Class.32e, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.fe4: type = pattern_type %Class.32e [symbolic]
// CHECK:STDOUT:   %pattern_type.f14b96.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.620: type = fn_type @Class.Get, @Class(%T.417) [symbolic]
// CHECK:STDOUT:   %Class.Get.cd8: %Class.Get.type.620 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5a6: type = ptr_type %Class.32e [symbolic]
// CHECK:STDOUT:   %pattern_type.ed4: type = pattern_type %ptr.5a6 [symbolic]
// CHECK:STDOUT:   %ptr.d8c: type = ptr_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.12f: type = pattern_type %ptr.d8c [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.6b3: type = fn_type @Class.GetAddr, @Class(%T.417) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.ed1: %Class.GetAddr.type.6b3 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.d0a: type = struct_type {.x: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.40f: <witness> = complete_type_witness %struct_type.x.d0a [symbolic]
// CHECK:STDOUT:   %require_complete.f79: <witness> = require_complete_type %Class.32e [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c42: <witness> = lookup_impl_witness %T.417, @Copy [symbolic]
// CHECK:STDOUT:   %.a79: type = fn_type_with_self_type %Copy.Op.type, %T.417 [symbolic]
// CHECK:STDOUT:   %impl.elem0.fac: %.a79 = impl_witness_access %Copy.lookup_impl_witness.c42, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.103: <specific function> = specific_impl_function %impl.elem0.fac, @Copy.Op(%T.417) [symbolic]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.75b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.692: %ptr.as.Copy.impl.Op.type.75b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.34c: <witness> = lookup_impl_witness %ptr.d8c, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.89c: %Copy.type = facet_value %ptr.d8c, (%Copy.lookup_impl_witness.34c) [symbolic]
// CHECK:STDOUT:   %.dba: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.89c [symbolic]
// CHECK:STDOUT:   %impl.elem0.f88: %.dba = impl_witness_access %Copy.lookup_impl_witness.34c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5e8: <specific function> = specific_impl_function %impl.elem0.f88, @Copy.Op(%Copy.facet.89c) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.c50: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %Class.c06: type = class_type @Class, @Class(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %pattern_type.ece: type = pattern_type %Class.c06 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.ef0: type = unbound_element_type %Class.c06, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.36b: type = fn_type @Class.Get, @Class(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %Class.Get.a74: %Class.Get.type.36b = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.d89: type = fn_type @Class.GetAddr, @Class(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.070: %Class.GetAddr.type.d89 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c50 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.a74, @Class.Get(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %ptr.621: type = ptr_type %Class.c06 [concrete]
// CHECK:STDOUT:   %pattern_type.185: type = pattern_type %ptr.621 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.070, @Class.GetAddr(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT:   %complete_type.ae6: <witness> = complete_type_witness %ptr.621 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.be3: <witness> = impl_witness imports.%Copy.impl_witness_table.67d, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d7c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.687: %ptr.as.Copy.impl.Op.type.d7c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.687: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.be3) [concrete]
// CHECK:STDOUT:   %.cbc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.687 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.687, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.659: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.75b) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.692)]
// CHECK:STDOUT:   %Copy.impl_witness_table.67d = impl_witness_table (%Core.import_ref.659), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: %Copy.type) {
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
// CHECK:STDOUT:     .Self = constants.%Class.32e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5_8
// CHECK:STDOUT:     .Get = %Class.Get.decl
// CHECK:STDOUT:     .GetAddr = %Class.GetAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Get(@Class.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.bc9)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9 (constants.%require_complete.d74)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c42)]
// CHECK:STDOUT:   %.loc9_16.4: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc9_16.4 (constants.%.a79)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.4 (%.a79) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.fac)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.103)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.32e)) -> %return.param: @Class.Get.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.32e) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.bc9) = name_ref x, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.4 (%.a79) = impl_witness_access constants.%Copy.lookup_impl_witness.c42, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.fac)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.Op(constants.%T.417) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.103)]
// CHECK:STDOUT:     %bound_method.loc9_16.2: <bound method> = bound_method %.loc9_16.2, %specific_impl_fn.loc9_16.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_16.3: init @Class.Get.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc9_16.2(%.loc9_16.2) to %.loc7_24
// CHECK:STDOUT:     return %.loc9_16.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %Class [symbolic = %require_complete.loc15 (constants.%require_complete.f79)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.bc9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_38.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.34c)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_38.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.89c)]
// CHECK:STDOUT:   %.loc15_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc15_12.2 (constants.%.dba)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.2 (%.dba) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.f88)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.5e8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetAddr.%ptr.loc13_29.1 (%ptr.5a6)) -> @Class.GetAddr.%ptr.loc13_38.1 (%ptr.d8c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetAddr.%ptr.loc13_29.1 (%ptr.5a6) = name_ref self, %self
// CHECK:STDOUT:     %.loc15_17.1: ref @Class.GetAddr.%Class (%Class.32e) = deref %self.ref
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.bc9) = name_ref x, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:     %.loc15_17.2: ref @Class.GetAddr.%T.binding.as_type (%T.binding.as_type) = class_element_access %.loc15_17.1, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_38.1 (%ptr.d8c) = addr_of %.loc15_17.2
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.2 (%.dba) = impl_witness_access constants.%Copy.lookup_impl_witness.34c, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.f88)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.Op(constants.%Copy.facet.89c) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.5e8)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %.loc15_12.1: init @Class.GetAddr.%ptr.loc13_38.1 (%ptr.d8c) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %.loc15_12.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.c06) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.c06 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.ef0 = name_ref x, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.c06) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.c06 = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.36b = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.c50) [concrete = constants.%Class.Get.a74]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.36b = name_ref Get, %.loc28 [concrete = constants.%Class.Get.a74]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.c50) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.621) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.621 = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.c06 = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.d89 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.c50) [concrete = constants.%Class.GetAddr.070]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.d89 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.070]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.c50) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %addr: %ptr.621 = addr_of %.loc34_12.1
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.235 = call %bound_method.loc34_22(%addr)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.235 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.235 = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc34_10.1: <bound method> = bound_method %.loc34_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_10.2: <bound method> = bound_method %.loc34_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc34_10.2(%.loc34_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.417) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.417
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d74
// CHECK:STDOUT:   %Class => constants.%Class.32e
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.bc9
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.620
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.cd8
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.6b3
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.ed1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.d0a
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.40f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.417) {
// CHECK:STDOUT:   %T => constants.%T.417
// CHECK:STDOUT:   %Class => constants.%Class.32e
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.fe4
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.f14b96.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.417) {
// CHECK:STDOUT:   %T => constants.%T.417
// CHECK:STDOUT:   %Class => constants.%Class.32e
// CHECK:STDOUT:   %ptr.loc13_29.1 => constants.%ptr.5a6
// CHECK:STDOUT:   %pattern_type.loc13_19 => constants.%pattern_type.ed4
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %ptr.loc13_38.1 => constants.%ptr.d8c
// CHECK:STDOUT:   %pattern_type.loc13_34 => constants.%pattern_type.12f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.c50) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.c50
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.c06
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.ef0
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.36b
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.a74
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.d89
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.070
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.c50) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.c50
// CHECK:STDOUT:   %Class => constants.%Class.c06
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.ece
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.ef0
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.fb7
// CHECK:STDOUT:   %.loc9_16.4 => constants.%.65f
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.dfd
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.c50) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.c50
// CHECK:STDOUT:   %Class => constants.%Class.c06
// CHECK:STDOUT:   %ptr.loc13_29.1 => constants.%ptr.621
// CHECK:STDOUT:   %pattern_type.loc13_19 => constants.%pattern_type.185
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_38.1 => constants.%ptr.235
// CHECK:STDOUT:   %pattern_type.loc13_34 => constants.%pattern_type.fe8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_34 => constants.%complete_type.3d0
// CHECK:STDOUT:   %require_complete.loc13_23 => constants.%complete_type.ae6
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.ef0
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.be3
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.687
// CHECK:STDOUT:   %.loc15_12.2 => constants.%.cbc
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.687
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
// CHECK:STDOUT:   %pattern_type.566: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Make: %Class.Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.specific_fn: <specific function> = specific_function %Class.Make, @Class.Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8_29.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc8_49.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T.loc8_29.1) [symbolic = %Class.Make.type (constants.%Class.Make.type)]
// CHECK:STDOUT:   %Class.Make: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = struct_value () [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:   %Class.Make.specific_fn.loc10_18.2: <specific function> = specific_function %Class.Make, @Class.Make(%T.loc8_29.1) [symbolic = %Class.Make.specific_fn.loc10_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc10: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_29.2 [symbolic = %T.loc8_29.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc10: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.1 (constants.%Class)]
// CHECK:STDOUT:     %.loc10: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = specific_constant @Class.%Class.Make.decl, @Class(constants.%T) [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = name_ref Make, %.loc10 [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Class.Make.specific_fn.loc10_18.1: <specific function> = specific_function %Make.ref, @Class.Make(constants.%T) [symbolic = %Class.Make.specific_fn.loc10_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Class.Make.call: init @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) = call %Class.Make.specific_fn.loc10_18.1() to %.loc8
// CHECK:STDOUT:     return %Class.Make.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.loc8_29.1 => constants.%T
// CHECK:STDOUT:   %Class.loc8_49.1 => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.566
// CHECK:STDOUT: }
// CHECK:STDOUT:

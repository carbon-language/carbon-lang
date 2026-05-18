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
// CHECK:STDOUT:   %T.0e9: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.baf: type = class_type @Class, @Class(%T.0e9) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.0e9 [symbolic]
// CHECK:STDOUT:   %require_complete.8de: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.ecae91.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.c34: type = unbound_element_type %Class.baf, %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.d54: type = pattern_type %Class.baf [symbolic]
// CHECK:STDOUT:   %.c437f1.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.522: type = fn_type @Class.Get, @Class(%T.0e9) [symbolic]
// CHECK:STDOUT:   %Class.Get.bc7: %Class.Get.type.522 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.ada: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.1fa: Core.Form = init_form %ptr.ada [symbolic]
// CHECK:STDOUT:   %pattern_type.637: type = pattern_type %ptr.ada [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.13a: type = fn_type @Class.GetAddr, @Class(%T.0e9) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.12d: %Class.GetAddr.type.13a = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.b3c: type = struct_type {.x: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.274: <witness> = complete_type_witness %struct_type.x.b3c [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %T.0e9, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e87e4c.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.0e9) [symbolic]
// CHECK:STDOUT:   %.499: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e87e4c.2, %T.0e9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.615: %.499 = impl_witness_access %Copy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.baf: <specific function> = specific_impl_function %impl.elem0.615, @Copy.WithSelf.Op(%T.0e9) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.8a8: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6bf: %ptr.as.Copy.impl.Op.type.8a8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9a: <witness> = lookup_impl_witness %ptr.ada, @Copy [symbolic]
// CHECK:STDOUT:   %.ceb: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet.97b: %Copy.type = facet_value %ptr.ada, (%Copy.lookup_impl_witness.d9a) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.491: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.97b) [symbolic]
// CHECK:STDOUT:   %.5d1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.491, %Copy.facet.97b [symbolic]
// CHECK:STDOUT:   %impl.elem0.7e6: %.5d1 = impl_witness_access %Copy.lookup_impl_witness.d9a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.024: <specific function> = specific_impl_function %impl.elem0.7e6, @Copy.WithSelf.Op(%Copy.facet.97b) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f98: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6d3: %Int.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.ddd: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Class.b86: type = class_type @Class, @Class(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %pattern_type.7fe: type = pattern_type %Class.b86 [concrete]
// CHECK:STDOUT:   %.f87: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.0bd: type = unbound_element_type %Class.b86, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.a27: type = fn_type @Class.Get, @Class(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %Class.Get.a82: %Class.Get.type.a27 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.ffc: type = fn_type @Class.GetAddr, @Class(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.da8: %Class.GetAddr.type.ffc = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.60b: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.b36: <witness> = complete_type_witness %struct_type.x.60b [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.83a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.83a, %Copy.facet.ddd [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.6d3, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.a82, @Class.Get(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %ptr.1e6: type = ptr_type %Class.b86 [concrete]
// CHECK:STDOUT:   %ptr.2b7: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.036: Core.Form = init_form %ptr.2b7 [concrete]
// CHECK:STDOUT:   %pattern_type.a72: type = pattern_type %ptr.2b7 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.da8, @Class.GetAddr(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %complete_type.49b: <witness> = complete_type_witness %ptr.2b7 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.213: <witness> = impl_witness imports.%Copy.impl_witness_table.3cb, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.17d: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.dee: %ptr.as.Copy.impl.Op.type.17d = struct_value () [concrete]
// CHECK:STDOUT:   %.bb1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.a97: %Copy.type = facet_value %ptr.2b7, (%Copy.impl_witness.213) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9c0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.a97) [concrete]
// CHECK:STDOUT:   %.421: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9c0, %Copy.facet.a97 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.dee, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.3ef: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.8a8) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.6bf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3cb = impl_witness_table (%Core.import_ref.3ef), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Class.baf
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_27.1 [symbolic = %Class.elem (constants.%Class.elem.c34)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc7_27.1 [symbolic = %require_complete.loc9 (constants.%require_complete.8de)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e87e4c.2)]
// CHECK:STDOUT:   %.loc9_16.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_16.3 (constants.%.499)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.3 (%.499) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.615)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.baf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.baf)) -> out %return.param: @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.baf) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.c34) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.as_type.loc7_27.1 (%T.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.3 (%.499) = impl_witness_access constants.%Copy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.615)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.WithSelf.Op(constants.%T.0e9) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.baf)]
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc13_36.1 [symbolic = %Class.elem (constants.%Class.elem.c34)]
// CHECK:STDOUT:   %.loc15_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc13_36.1) [symbolic = %.loc15_12.1 (constants.%.ceb)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9a)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.97b)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.491)]
// CHECK:STDOUT:   %.loc15_12.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc15_12.2 (constants.%.5d1)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.2 (%.5d1) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.7e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.024)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class.baf)) -> out %return.param: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.ada) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.baf) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.c34) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.as_type.loc13_36.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.ada) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.2 (%.5d1) = impl_witness_access constants.%Copy.lookup_impl_witness.d9a, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.7e6)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.97b) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.024)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc13_36.1 (%ptr.ada) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.b86) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.b86 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.0bd = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.b86) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.b86 = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.a27 = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.ddd) [concrete = constants.%Class.Get.a82]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.a27 = name_ref Get, %.loc28 [concrete = constants.%Class.Get.a82]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.ddd) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.1e6) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1e6 = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.b86 = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.ffc = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.ddd) [concrete = constants.%Class.GetAddr.da8]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.ffc = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.da8]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.ddd) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.2b7 = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.2b7 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.2b7 = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc34_10.1: <bound method> = bound_method %.loc34_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_10.2: <bound method> = bound_method %.loc34_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc34_10.2(%.loc34_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.0e9) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.0e9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8de
// CHECK:STDOUT:   %Class => constants.%Class.baf
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.c34
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.522
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.bc7
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.13a
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.12d
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.b3c
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.274
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.0e9) {
// CHECK:STDOUT:   %T => constants.%T.0e9
// CHECK:STDOUT:   %Class => constants.%Class.baf
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.d54
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.c437f1.2
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.ecae91.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.0e9) {
// CHECK:STDOUT:   %T => constants.%T.0e9
// CHECK:STDOUT:   %Class => constants.%Class.baf
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.d54
// CHECK:STDOUT:   %T.as_type.loc13_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.ada
// CHECK:STDOUT:   %.loc13_36.1 => constants.%.1fa
// CHECK:STDOUT:   %pattern_type.loc13_36 => constants.%pattern_type.637
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.ddd) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.ddd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.b86
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.0bd
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.a27
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.a82
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.ffc
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.da8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.60b
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.b36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.ddd) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.ddd
// CHECK:STDOUT:   %Class => constants.%Class.b86
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %T.as_type.loc7_27.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.f87
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.17a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.b36
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.0bd
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.83a
// CHECK:STDOUT:   %.loc9_16.3 => constants.%.2ec
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f1f
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.6d3
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.ddd) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.ddd
// CHECK:STDOUT:   %Class => constants.%Class.b86
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %T.as_type.loc13_36.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.2b7
// CHECK:STDOUT:   %.loc13_36.1 => constants.%.036
// CHECK:STDOUT:   %pattern_type.loc13_36 => constants.%pattern_type.a72
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_22 => constants.%complete_type.b36
// CHECK:STDOUT:   %require_complete.loc13_32 => constants.%complete_type.49b
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.0bd
// CHECK:STDOUT:   %.loc15_12.1 => constants.%.bb1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.213
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.a97
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.9c0
// CHECK:STDOUT:   %.loc15_12.2 => constants.%.421
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.dee
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
// CHECK:STDOUT:   %.a41: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.a7d: type = pattern_type %Class [symbolic]
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
// CHECK:STDOUT:   %.loc8_49.2 => constants.%.a41
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a7d
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

class Class(T: Core.Copy) {
  var x: T;

  fn Get(self) -> T {
    //@dump-sem-ir-begin
    return self.x;
    //@dump-sem-ir-end
  }

  fn GetAddr(ref self) -> T* {
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

class Class(T: type) {
  fn Make() -> Class(T) { return {}; }
}

fn StaticMemberFunctionCall(generic T: type) -> Class(T) {
  //@dump-sem-ir-begin
  return Class(T).Make();
  //@dump-sem-ir-end
}

// --- access_private_method_after_specific_eval.carbon

library "[[@TEST_NAME]]";

interface I(T: type) {
  fn F(self, t: T);
}

class A {}
class B {}

impl A as I(B) {
  fn F(unused self, unused b: B) {}
}

class C {
  private fn G(ref self) {
    let a: A = {};
    a.(I(B).F)({});
    self.G();
  }
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.c7a: type = class_type @Class, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.d19: type = unbound_element_type %Class.c7a, %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.5d2: type = pattern_type %Class.c7a [symbolic]
// CHECK:STDOUT:   %self.param_patt.791: %pattern_type.5d2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.099: %pattern_type.5d2 = wrapper_binding_pattern self, %self.param_patt.791 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.17e: type = fn_type @Class.Get, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.Get.948: %Class.Get.type.17e = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.140: %pattern_type.5d2 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.8cd: %pattern_type.5d2 = wrapper_binding_pattern self, %self.param_patt.140 [symbolic]
// CHECK:STDOUT:   %ptr.215: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.4fb: Core.Form = init_form %ptr.215 [symbolic]
// CHECK:STDOUT:   %pattern_type.d5b: type = pattern_type %ptr.215 [symbolic]
// CHECK:STDOUT:   %return.param_patt.4df: %pattern_type.d5b = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c21: %pattern_type.d5b = return_slot_pattern %return.param_patt.4df, %ptr.215 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.4f0: type = fn_type @Class.GetAddr, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.4cf: %Class.GetAddr.type.4f0 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.ac1: type = struct_type {.x: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.b8d: <witness> = complete_type_witness %struct_type.x.ac1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb5: %.8bf = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ac: <specific function> = specific_impl_function %impl.elem0.bb5, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cf6: <witness> = lookup_impl_witness %ptr.215, @Copy [symbolic]
// CHECK:STDOUT:   %.708: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet.ed3: %Copy.type = facet_value %ptr.215, (%Copy.lookup_impl_witness.cf6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.732: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ed3) [symbolic]
// CHECK:STDOUT:   %.1d4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.732, %Copy.facet.ed3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a74: %.1d4 = impl_witness_access %Copy.lookup_impl_witness.cf6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ca9: <specific function> = specific_impl_function %impl.elem0.a74, @Copy.WithSelf.Op(%Copy.facet.ed3) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.828: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Class.d92: type = class_type @Class, @Class(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %pattern_type.b3f: type = pattern_type %Class.d92 [concrete]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.39a: type = unbound_element_type %Class.d92, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.a8d: type = fn_type @Class.Get, @Class(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %Class.Get.8a3: %Class.Get.type.a8d = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.a65: type = fn_type @Class.GetAddr, @Class(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.c63: %Class.GetAddr.type.a65 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.d18: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.67a: <witness> = complete_type_witness %struct_type.x.d18 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet.828 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.f56: %pattern_type.b3f = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a72: %pattern_type.b3f = wrapper_binding_pattern self, %self.param_patt.f56 [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.8a3, @Class.Get(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %ptr.86b: type = ptr_type %Class.d92 [concrete]
// CHECK:STDOUT:   %self.param_patt.0ed: %pattern_type.b3f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.2da: %pattern_type.b3f = wrapper_binding_pattern self, %self.param_patt.0ed [concrete]
// CHECK:STDOUT:   %ptr.1fe: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.857: Core.Form = init_form %ptr.1fe [concrete]
// CHECK:STDOUT:   %pattern_type.55f: type = pattern_type %ptr.1fe [concrete]
// CHECK:STDOUT:   %return.param_patt.10d: %pattern_type.55f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.568: %pattern_type.55f = return_slot_pattern %return.param_patt.10d, %ptr.1fe [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.c63, @Class.GetAddr(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %complete_type.526: <witness> = complete_type_witness %ptr.1fe [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1fc: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.194: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.e42: %ptr.as.Copy.impl.Op.type.194 = struct_value () [concrete]
// CHECK:STDOUT:   %.52e: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.b57: %Copy.type = facet_value %ptr.1fe, (%Copy.impl_witness.1fc) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4c0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b57) [concrete]
// CHECK:STDOUT:   %.b1d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4c0, %Copy.facet.b57 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.e42, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Class.c7a
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %field_decl
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_19.1 [symbolic = %Class.elem (constants.%Class.elem.d19)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc7_19.1 [symbolic = %require_complete.loc9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc9_16.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_16.5 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.5 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.c7a)) -> out %return.param: @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.c7a) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.d19) = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.5 (%.8bf) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %.loc9_16.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc9_16.4: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:     %bound_method.loc9_16.2: <bound method> = bound_method %.loc9_16.2, %specific_impl_fn.loc9_16.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) to %.loc7_19.1 = call %bound_method.loc9_16.2(%.loc9_16.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc13_28.1 [symbolic = %Class.elem (constants.%Class.elem.d19)]
// CHECK:STDOUT:   %.loc15_12.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc13_28.1) [symbolic = %.loc15_12.3 (constants.%.708)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cf6)]
// CHECK:STDOUT:   %Copy.facet.loc15_12.3: %Copy.type = facet_value %ptr.loc13_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc15_12.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.732)]
// CHECK:STDOUT:   %.loc15_12.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc15_12.3 [symbolic = %.loc15_12.4 (constants.%.1d4)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.4 (%.1d4) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.a74)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.WithSelf.Op(%Copy.facet.loc15_12.3) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.ca9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class.c7a)) -> out %return.param: @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.c7a) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.d19) = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.as_type.loc13_28.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.4 (%.1d4) = impl_witness_access constants.%Copy.lookup_impl_witness.cf6, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.a74)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %Copy.facet.loc15_12.1: %Copy.type = facet_value constants.%ptr.215, (constants.%Copy.lookup_impl_witness.cf6) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %.loc15_12.1: %Copy.type = converted constants.%ptr.215, %Copy.facet.loc15_12.1 [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %Copy.facet.loc15_12.2: %Copy.type = facet_value constants.%ptr.215, (constants.%Copy.lookup_impl_witness.cf6) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %.loc15_12.2: %Copy.type = converted constants.%ptr.215, %Copy.facet.loc15_12.2 [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.ed3) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.ca9)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.d92) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.d92 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.39a = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.d92) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.d92 = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.a8d = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.828) [concrete = constants.%Class.Get.8a3]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.a8d = name_ref Get, %.loc28 [concrete = constants.%Class.Get.8a3]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.828) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.86b) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.86b = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.d92 = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.a65 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.828) [concrete = constants.%Class.GetAddr.c63]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.a65 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.c63]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.828) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.1fe = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.1fe = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.1fe = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc34_10.1: <bound method> = bound_method %.loc34_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_10.2: <bound method> = bound_method %.loc34_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc34_10.2(%.loc34_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f84
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc5_10.2 => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %Class => constants.%Class.c7a
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.d19
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.17e
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.948
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.4f0
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.4cf
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ac1
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.b8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class.c7a
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.5d2
// CHECK:STDOUT:   %self.param_patt.loc7_10.2 => constants.%self.param_patt.791
// CHECK:STDOUT:   %self.patt.loc7_10.2 => constants.%self.patt.099
// CHECK:STDOUT:   %T.as_type.loc7_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_19.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %return.param_patt.loc7_19.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc7_16.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class.c7a
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.5d2
// CHECK:STDOUT:   %self.param_patt.loc13_18.2 => constants.%self.param_patt.140
// CHECK:STDOUT:   %self.patt.loc13_18.2 => constants.%self.patt.8cd
// CHECK:STDOUT:   %T.as_type.loc13_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc13_28.1 => constants.%ptr.215
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.4fb
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.d5b
// CHECK:STDOUT:   %return.param_patt.loc13_28.2 => constants.%return.param_patt.4df
// CHECK:STDOUT:   %return.patt.loc13_24.2 => constants.%return.patt.c21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.828) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.828
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc5_10.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.d92
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.39a
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.a8d
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.8a3
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.a65
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.c63
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.d18
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.67a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.828) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.828
// CHECK:STDOUT:   %Class => constants.%Class.d92
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.b3f
// CHECK:STDOUT:   %self.param_patt.loc7_10.2 => constants.%self.param_patt.f56
// CHECK:STDOUT:   %self.patt.loc7_10.2 => constants.%self.patt.a72
// CHECK:STDOUT:   %T.as_type.loc7_19.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_19.2 => constants.%.55c
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.9d9
// CHECK:STDOUT:   %return.param_patt.loc7_19.2 => constants.%return.param_patt.13f
// CHECK:STDOUT:   %return.patt.loc7_16.2 => constants.%return.patt.a79
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.67a
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.39a
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.bc1
// CHECK:STDOUT:   %.loc9_16.5 => constants.%.481
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.0ff
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.617
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.828) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.828
// CHECK:STDOUT:   %Class => constants.%Class.d92
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.b3f
// CHECK:STDOUT:   %self.param_patt.loc13_18.2 => constants.%self.param_patt.0ed
// CHECK:STDOUT:   %self.patt.loc13_18.2 => constants.%self.patt.2da
// CHECK:STDOUT:   %T.as_type.loc13_28.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_28.1 => constants.%ptr.1fe
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.857
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.55f
// CHECK:STDOUT:   %return.param_patt.loc13_28.2 => constants.%return.param_patt.10d
// CHECK:STDOUT:   %return.patt.loc13_24.2 => constants.%return.patt.568
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_18 => constants.%complete_type.67a
// CHECK:STDOUT:   %require_complete.loc13_24 => constants.%complete_type.526
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.39a
// CHECK:STDOUT:   %.loc15_12.3 => constants.%.52e
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1fc
// CHECK:STDOUT:   %Copy.facet.loc15_12.3 => constants.%Copy.facet.b57
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.4c0
// CHECK:STDOUT:   %.loc15_12.4 => constants.%.b1d
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.e42
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.b65: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.f4e: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.f4e = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt: %pattern_type.f4e = return_slot_pattern %return.param_patt, %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Make: %Class.Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.specific_fn: <specific function> = specific_function %Class.Make, @Class.Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8_38.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc8_56.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T.loc8_38.1) [symbolic = %Class.Make.type (constants.%Class.Make.type)]
// CHECK:STDOUT:   %Class.Make: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = struct_value () [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:   %Class.Make.specific_fn.loc10_18.2: <specific function> = specific_function %Class.Make, @Class.Make(%T.loc8_38.1) [symbolic = %Class.Make.specific_fn.loc10_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @StaticMemberFunctionCall.%Class.loc8_56.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc10: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_38.2 [symbolic = %T.loc8_38.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc10: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_56.1 (constants.%Class)]
// CHECK:STDOUT:     %.loc10: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = specific_constant @Class.%Class.Make.decl, @Class(constants.%T) [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = name_ref Make, %.loc10 [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Class.Make.specific_fn.loc10_18.1: <specific function> = specific_function %Make.ref, @Class.Make(constants.%T) [symbolic = %Class.Make.specific_fn.loc10_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Class.Make.call: init @StaticMemberFunctionCall.%Class.loc8_56.1 (%Class) to %.loc8_56.1 = call %Class.Make.specific_fn.loc10_18.1()
// CHECK:STDOUT:     return %Class.Make.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_38.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_38.1 => constants.%T
// CHECK:STDOUT:   %Class.loc8_56.1 => constants.%Class
// CHECK:STDOUT:   %.loc8_56.2 => constants.%.b65
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4e
// CHECK:STDOUT:   %return.param_patt.loc8_56.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc8_46.2 => constants.%return.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:

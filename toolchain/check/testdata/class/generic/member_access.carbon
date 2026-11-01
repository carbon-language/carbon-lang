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
// CHECK:STDOUT:   %Class.a52: type = class_type @Class, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.b17: type = unbound_element_type %Class.a52, %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.f62: type = pattern_type %Class.a52 [symbolic]
// CHECK:STDOUT:   %self.param_patt.4c1: %pattern_type.f62 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.f3a: %pattern_type.f62 = wrapper_binding_pattern self, %self.param_patt.4c1 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.757: type = fn_type @Class.Get, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.Get.495: %Class.Get.type.757 = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.1bb: %pattern_type.f62 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d0f: %pattern_type.f62 = wrapper_binding_pattern self, %self.param_patt.1bb [symbolic]
// CHECK:STDOUT:   %ptr.215: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.4fb: Core.Form = init_form %ptr.215 [symbolic]
// CHECK:STDOUT:   %pattern_type.d5b: type = pattern_type %ptr.215 [symbolic]
// CHECK:STDOUT:   %return.param_patt.4df: %pattern_type.d5b = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c21: %pattern_type.d5b = return_slot_pattern %return.param_patt.4df, %ptr.215 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.e08: type = fn_type @Class.GetAddr, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.a83: %Class.GetAddr.type.e08 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Copy.facet.ed3: %Copy.type = facet_value %ptr.215, (%Copy.lookup_impl_witness.cf6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.732: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ed3) [symbolic]
// CHECK:STDOUT:   %.1d4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.732, %Copy.facet.ed3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a74: %.1d4 = impl_witness_access %Copy.lookup_impl_witness.cf6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ca9: <specific function> = specific_impl_function %impl.elem0.a74, @Copy.WithSelf.Op(%Copy.facet.ed3) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Class.4cd: type = class_type @Class, @Class(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %pattern_type.245: type = pattern_type %Class.4cd [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.elem.4b5: type = unbound_element_type %Class.4cd, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.b33: type = fn_type @Class.Get, @Class(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %Class.Get.e4b: %Class.Get.type.b33 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.399: type = fn_type @Class.GetAddr, @Class(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.597: %Class.GetAddr.type.399 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.991: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c49: <witness> = complete_type_witness %struct_type.x.991 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.d7b: %pattern_type.245 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e4e: %pattern_type.245 = wrapper_binding_pattern self, %self.param_patt.d7b [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.e4b, @Class.Get(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %ptr.dd1: type = ptr_type %Class.4cd [concrete]
// CHECK:STDOUT:   %self.param_patt.9b2: %pattern_type.245 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bfe: %pattern_type.245 = wrapper_binding_pattern self, %self.param_patt.9b2 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.43c: Core.Form = init_form %ptr.a17 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %return.param_patt.6e5: %pattern_type.7dc = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.1b4: %pattern_type.7dc = return_slot_pattern %return.param_patt.6e5, %ptr.a17 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.597, @Class.GetAddr(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.83a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.652: %Copy.type = facet_value %ptr.a17, (%Copy.impl_witness.83a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9a2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.652) [concrete]
// CHECK:STDOUT:   %.309: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9a2, %Copy.facet.652 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dfc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.81f: %ptr.as.Copy.impl.Op.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.81f, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Class.a52
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_19.1 [symbolic = %Class.elem (constants.%Class.elem.b17)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc7_19.1 [symbolic = %require_complete.loc9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc9_16.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_16.5 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.5 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.a52)) -> out %return.param: @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.a52) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.b17) = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc13_28.1 [symbolic = %Class.elem (constants.%Class.elem.b17)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cf6)]
// CHECK:STDOUT:   %Copy.facet.loc15_12.3: %Copy.type = facet_value %ptr.loc13_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc15_12.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.732)]
// CHECK:STDOUT:   %.loc15_12.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc15_12.3 [symbolic = %.loc15_12.3 (constants.%.1d4)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.3 (%.1d4) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.a74)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.WithSelf.Op(%Copy.facet.loc15_12.3) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.ca9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class.a52)) -> out %return.param: @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.a52) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.b17) = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.as_type.loc13_28.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.3 (%.1d4) = impl_witness_access constants.%Copy.lookup_impl_witness.cf6, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.a74)]
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
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.4cd) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.4cd = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.4b5 = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.4cd) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.4cd = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.b33 = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.95d) [concrete = constants.%Class.Get.e4b]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.b33 = name_ref Get, %.loc28 [concrete = constants.%Class.Get.e4b]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.95d) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.dd1) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.dd1 = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.4cd = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.399 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.95d) [concrete = constants.%Class.GetAddr.597]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.399 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.597]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.95d) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.a17 = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.a17 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.a17 = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
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
// CHECK:STDOUT:   %Class => constants.%Class.a52
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.b17
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.757
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.495
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.e08
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.a83
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ac1
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.b8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class.a52
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.f62
// CHECK:STDOUT:   %self.param_patt.loc7_10.2 => constants.%self.param_patt.4c1
// CHECK:STDOUT:   %self.patt.loc7_10.2 => constants.%self.patt.f3a
// CHECK:STDOUT:   %T.as_type.loc7_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_19.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %return.param_patt.loc7_19.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc7_16.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class.a52
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.f62
// CHECK:STDOUT:   %self.param_patt.loc13_18.2 => constants.%self.param_patt.1bb
// CHECK:STDOUT:   %self.patt.loc13_18.2 => constants.%self.patt.d0f
// CHECK:STDOUT:   %T.as_type.loc13_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc13_28.1 => constants.%ptr.215
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.4fb
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.d5b
// CHECK:STDOUT:   %return.param_patt.loc13_28.2 => constants.%return.param_patt.4df
// CHECK:STDOUT:   %return.patt.loc13_24.2 => constants.%return.patt.c21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.95d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc5_10.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.4cd
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.4b5
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.b33
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.e4b
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.399
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.597
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.991
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.c49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.95d
// CHECK:STDOUT:   %Class => constants.%Class.4cd
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.245
// CHECK:STDOUT:   %self.param_patt.loc7_10.2 => constants.%self.param_patt.d7b
// CHECK:STDOUT:   %self.patt.loc7_10.2 => constants.%self.patt.e4e
// CHECK:STDOUT:   %T.as_type.loc7_19.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_19.2 => constants.%.0b3
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.d94
// CHECK:STDOUT:   %return.param_patt.loc7_19.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc7_16.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.c49
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.4b5
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.f76
// CHECK:STDOUT:   %.loc9_16.5 => constants.%.2b9
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.8af
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.cae
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.95d
// CHECK:STDOUT:   %Class => constants.%Class.4cd
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.245
// CHECK:STDOUT:   %self.param_patt.loc13_18.2 => constants.%self.param_patt.9b2
// CHECK:STDOUT:   %self.patt.loc13_18.2 => constants.%self.patt.bfe
// CHECK:STDOUT:   %T.as_type.loc13_28.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_28.1 => constants.%ptr.a17
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.43c
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %return.param_patt.loc13_28.2 => constants.%return.param_patt.6e5
// CHECK:STDOUT:   %return.patt.loc13_24.2 => constants.%return.patt.1b4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_18 => constants.%complete_type.c49
// CHECK:STDOUT:   %require_complete.loc13_24 => constants.%complete_type.1b1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.4b5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.83a
// CHECK:STDOUT:   %Copy.facet.loc15_12.3 => constants.%Copy.facet.652
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.9a2
// CHECK:STDOUT:   %.loc15_12.3 => constants.%.309
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.81f
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
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.917: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.28f: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.28f = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt: %pattern_type.28f = return_slot_pattern %return.param_patt, %Class [symbolic]
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
// CHECK:STDOUT:   %.loc8_56.2 => constants.%.917
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.28f
// CHECK:STDOUT:   %return.param_patt.loc8_56.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc8_46.2 => constants.%return.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %Class.bac: type = class_type @Class, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.2e6: type = unbound_element_type %Class.bac, %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.037: type = pattern_type %Class.bac [symbolic]
// CHECK:STDOUT:   %self.param_patt.82d: %pattern_type.037 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d7a: %pattern_type.037 = wrapper_binding_pattern self, %self.param_patt.82d [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.fce: type = fn_type @Class.Get, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.Get.04a: %Class.Get.type.fce = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.076: %pattern_type.037 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.503: %pattern_type.037 = wrapper_binding_pattern self, %self.param_patt.076 [symbolic]
// CHECK:STDOUT:   %ptr.215: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.4fb: Core.Form = init_form %ptr.215 [symbolic]
// CHECK:STDOUT:   %pattern_type.d5b: type = pattern_type %ptr.215 [symbolic]
// CHECK:STDOUT:   %return.param_patt.4df: %pattern_type.d5b = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c21: %pattern_type.d5b = return_slot_pattern %return.param_patt.4df, %ptr.215 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.031: type = fn_type @Class.GetAddr, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.ed6: %Class.GetAddr.type.031 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.ac1: type = struct_type {.x: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.b8d: <witness> = complete_type_witness %struct_type.x.ac1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cf6: <witness> = lookup_impl_witness %ptr.215, @Copy [symbolic]
// CHECK:STDOUT:   %.a33: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet.ed3: %Copy.type = facet_value %ptr.215, (%Copy.lookup_impl_witness.cf6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ed3) [symbolic]
// CHECK:STDOUT:   %.45d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e0, %Copy.facet.ed3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c5d: %.45d = impl_witness_access %Copy.lookup_impl_witness.cf6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5cf: <specific function> = specific_impl_function %impl.elem0.c5d, @Copy.WithSelf.Op(%Copy.facet.ed3) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.6e9: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Class.88b: type = class_type @Class, @Class(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %pattern_type.949: type = pattern_type %Class.88b [concrete]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.973: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a21: %pattern_type.df9 = return_slot_pattern %return.param_patt.973, %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.9d7: type = unbound_element_type %Class.88b, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.9a3: type = fn_type @Class.Get, @Class(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %Class.Get.9c7: %Class.Get.type.9a3 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.ed1: type = fn_type @Class.GetAddr, @Class(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.6a4: %Class.GetAddr.type.ed1 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.fb6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a7b: <witness> = complete_type_witness %struct_type.x.fb6 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet.6e9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.860: %pattern_type.949 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.131: %pattern_type.949 = wrapper_binding_pattern self, %self.param_patt.860 [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.9c7, @Class.Get(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %ptr.dcc: type = ptr_type %Class.88b [concrete]
// CHECK:STDOUT:   %self.param_patt.8cf: %pattern_type.949 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.9bb: %pattern_type.949 = wrapper_binding_pattern self, %self.param_patt.8cf [concrete]
// CHECK:STDOUT:   %ptr.49e: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.2a0: Core.Form = init_form %ptr.49e [concrete]
// CHECK:STDOUT:   %pattern_type.50c: type = pattern_type %ptr.49e [concrete]
// CHECK:STDOUT:   %return.param_patt.c11: %pattern_type.50c = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.9d1: %pattern_type.50c = return_slot_pattern %return.param_patt.c11, %ptr.49e [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.6a4, @Class.GetAddr(%Copy.facet.6e9) [concrete]
// CHECK:STDOUT:   %complete_type.1e6: <witness> = complete_type_witness %ptr.49e [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.dd4: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d83: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.22e: %ptr.as.Copy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %.537: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.759: %Copy.type = facet_value %ptr.49e, (%Copy.impl_witness.dd4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.037: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.759) [concrete]
// CHECK:STDOUT:   %.29f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.037, %Copy.facet.759 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.22e, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Class.bac
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_19.1 [symbolic = %Class.elem (constants.%Class.elem.2e6)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc7_19.1 [symbolic = %require_complete.loc9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc9_16.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_16.5 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.5 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.bac)) -> out %return.param: @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.bac) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.2e6) = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.as_type.loc7_19.1 (%T.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.5 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %.loc9_16.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc9_16.4: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.e2a)]
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc13_28.1 [symbolic = %Class.elem (constants.%Class.elem.2e6)]
// CHECK:STDOUT:   %.loc15_12.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc13_28.1) [symbolic = %.loc15_12.3 (constants.%.a33)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cf6)]
// CHECK:STDOUT:   %Copy.facet.loc15_12.3: %Copy.type = facet_value %ptr.loc13_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc15_12.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.6e0)]
// CHECK:STDOUT:   %.loc15_12.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc15_12.3 [symbolic = %.loc15_12.4 (constants.%.45d)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.4 (%.45d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.c5d)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.WithSelf.Op(%Copy.facet.loc15_12.3) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.5cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class.bac)) -> out %return.param: @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.bac) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.2e6) = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.as_type.loc13_28.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.4 (%.45d) = impl_witness_access constants.%Copy.lookup_impl_witness.cf6, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.c5d)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %Copy.facet.loc15_12.1: %Copy.type = facet_value constants.%ptr.215, (constants.%Copy.lookup_impl_witness.cf6) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %.loc15_12.1: %Copy.type = converted constants.%ptr.215, %Copy.facet.loc15_12.1 [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %Copy.facet.loc15_12.2: %Copy.type = facet_value constants.%ptr.215, (constants.%Copy.lookup_impl_witness.cf6) [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %.loc15_12.2: %Copy.type = converted constants.%ptr.215, %Copy.facet.loc15_12.2 [symbolic = %Copy.facet.loc15_12.3 (constants.%Copy.facet.ed3)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.ed3) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.5cf)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc13_28.1 (%ptr.215) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.88b) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.88b = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.9d7 = name_ref x, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.88b) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.88b = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.9a3 = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.6e9) [concrete = constants.%Class.Get.9c7]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.9a3 = name_ref Get, %.loc28 [concrete = constants.%Class.Get.9c7]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.6e9) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.dcc) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.dcc = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.88b = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.ed1 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.6e9) [concrete = constants.%Class.GetAddr.6a4]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.ed1 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.6a4]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.6e9) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.49e = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.49e = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.49e = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
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
// CHECK:STDOUT:   %Class => constants.%Class.bac
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2e6
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.fce
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.04a
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.031
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.ed6
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ac1
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.b8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class.bac
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.037
// CHECK:STDOUT:   %self.param_patt.loc7_10.2 => constants.%self.param_patt.82d
// CHECK:STDOUT:   %self.patt.loc7_10.2 => constants.%self.patt.d7a
// CHECK:STDOUT:   %T.as_type.loc7_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_19.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %return.param_patt.loc7_19.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc7_16.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class.bac
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.037
// CHECK:STDOUT:   %self.param_patt.loc13_18.2 => constants.%self.param_patt.076
// CHECK:STDOUT:   %self.patt.loc13_18.2 => constants.%self.patt.503
// CHECK:STDOUT:   %T.as_type.loc13_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc13_28.1 => constants.%ptr.215
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.4fb
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.d5b
// CHECK:STDOUT:   %return.param_patt.loc13_28.2 => constants.%return.param_patt.4df
// CHECK:STDOUT:   %return.patt.loc13_24.2 => constants.%return.patt.c21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.6e9) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.6e9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc5_10.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.88b
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.9d7
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.9a3
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.9c7
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.ed1
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.6a4
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.fb6
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.a7b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.6e9) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.6e9
// CHECK:STDOUT:   %Class => constants.%Class.88b
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.949
// CHECK:STDOUT:   %self.param_patt.loc7_10.2 => constants.%self.param_patt.860
// CHECK:STDOUT:   %self.patt.loc7_10.2 => constants.%self.patt.131
// CHECK:STDOUT:   %T.as_type.loc7_19.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_19.2 => constants.%.4e1
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.df9
// CHECK:STDOUT:   %return.param_patt.loc7_19.2 => constants.%return.param_patt.973
// CHECK:STDOUT:   %return.patt.loc7_16.2 => constants.%return.patt.a21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.a7b
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.9d7
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.67d
// CHECK:STDOUT:   %.loc9_16.5 => constants.%.187
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.4c3
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.175
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.6e9) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.6e9
// CHECK:STDOUT:   %Class => constants.%Class.88b
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.949
// CHECK:STDOUT:   %self.param_patt.loc13_18.2 => constants.%self.param_patt.8cf
// CHECK:STDOUT:   %self.patt.loc13_18.2 => constants.%self.patt.9bb
// CHECK:STDOUT:   %T.as_type.loc13_28.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_28.1 => constants.%ptr.49e
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.2a0
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.50c
// CHECK:STDOUT:   %return.param_patt.loc13_28.2 => constants.%return.param_patt.c11
// CHECK:STDOUT:   %return.patt.loc13_24.2 => constants.%return.patt.9d1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_18 => constants.%complete_type.a7b
// CHECK:STDOUT:   %require_complete.loc13_24 => constants.%complete_type.1e6
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.9d7
// CHECK:STDOUT:   %.loc15_12.3 => constants.%.537
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.dd4
// CHECK:STDOUT:   %Copy.facet.loc15_12.3 => constants.%Copy.facet.759
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.037
// CHECK:STDOUT:   %.loc15_12.4 => constants.%.29f
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.22e
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
// CHECK:STDOUT:   %.072: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.347: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.347 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt: %pattern_type.347 = return_slot_pattern %return.param_patt, %Class [symbolic]
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
// CHECK:STDOUT:   %.loc8_56.2 => constants.%.072
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.347
// CHECK:STDOUT:   %return.param_patt.loc8_56.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc8_46.2 => constants.%return.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:

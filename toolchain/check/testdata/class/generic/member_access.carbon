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
// CHECK:STDOUT:   %T.f92: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.8fd: type = class_type @Class, @Class(%T.f92) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.f92 [symbolic]
// CHECK:STDOUT:   %require_complete.91e: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem.aa1: type = unbound_element_type %Class.8fd, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.6e6: type = pattern_type %Class.8fd [symbolic]
// CHECK:STDOUT:   %pattern_type.c769df.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.Get.type.16e: type = fn_type @Class.Get, @Class(%T.f92) [symbolic]
// CHECK:STDOUT:   %Class.Get.ff7: %Class.Get.type.16e = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.2e1: type = ptr_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.95d: type = pattern_type %ptr.2e1 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.14f: type = fn_type @Class.GetAddr, @Class(%T.f92) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.b6b: %Class.GetAddr.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.12c: type = struct_type {.x: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.7f9: <witness> = complete_type_witness %struct_type.x.12c [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.edd: <witness> = lookup_impl_witness %T.f92, @Copy [symbolic]
// CHECK:STDOUT:   %.232: type = fn_type_with_self_type %Copy.Op.type, %T.f92 [symbolic]
// CHECK:STDOUT:   %impl.elem0.8df: %.232 = impl_witness_access %Copy.lookup_impl_witness.edd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5c4: <specific function> = specific_impl_function %impl.elem0.8df, @Copy.Op(%T.f92) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b05: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8b8: %ptr.as.Copy.impl.Op.type.b05 = struct_value () [symbolic]
// CHECK:STDOUT:   %.892: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.29a: <witness> = lookup_impl_witness %ptr.2e1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.6b4: %Copy.type = facet_value %ptr.2e1, (%Copy.lookup_impl_witness.29a) [symbolic]
// CHECK:STDOUT:   %.057: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.6b4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.120: %.057 = impl_witness_access %Copy.lookup_impl_witness.29a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5a3: <specific function> = specific_impl_function %impl.elem0.120, @Copy.Op(%Copy.facet.6b4) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.413: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.2d6: %Int.as.Copy.impl.Op.type.413 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.66f: <witness> = impl_witness imports.%Copy.impl_witness_table.373, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.60b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.60b = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Copy.facet.9cb: %Copy.type = facet_value %i32, (%Copy.impl_witness.66f) [concrete]
// CHECK:STDOUT:   %Class.4d3: type = class_type @Class, @Class(%Copy.facet.9cb) [concrete]
// CHECK:STDOUT:   %pattern_type.373: type = pattern_type %Class.4d3 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.elem.702: type = unbound_element_type %Class.4d3, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.189: type = fn_type @Class.Get, @Class(%Copy.facet.9cb) [concrete]
// CHECK:STDOUT:   %Class.Get.4b6: %Class.Get.type.189 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.dc0: type = fn_type @Class.GetAddr, @Class(%Copy.facet.9cb) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.288: %Class.GetAddr.type.dc0 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %.958: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.9cb [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c85, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.4b6, @Class.Get(%Copy.facet.9cb) [concrete]
// CHECK:STDOUT:   %ptr.047: type = ptr_type %Class.4d3 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.288, @Class.GetAddr(%Copy.facet.9cb) [concrete]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9c7: <witness> = impl_witness imports.%Copy.impl_witness_table.4df, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.082: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.624: %ptr.as.Copy.impl.Op.type.082 = struct_value () [concrete]
// CHECK:STDOUT:   %.ff7: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.fff: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.9c7) [concrete]
// CHECK:STDOUT:   %.fef: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fff [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.624, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b85: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b05) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8b8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4df = impl_witness_table (%Core.import_ref.b85), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.30b: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.413) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.2d6)]
// CHECK:STDOUT:   %Copy.impl_witness_table.373 = impl_witness_table (%Core.import_ref.30b), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Class.8fd
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
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.aa1)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9 (constants.%require_complete.91e)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.edd)]
// CHECK:STDOUT:   %.loc9_16.3: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc9_16.3 (constants.%.232)]
// CHECK:STDOUT:   %impl.elem0.loc9_16.2: @Class.Get.%.loc9_16.3 (%.232) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.8df)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2: <specific function> = specific_impl_function %impl.elem0.loc9_16.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.5c4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.8fd)) -> %return.param: @Class.Get.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.8fd) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.aa1) = name_ref x, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:     %.loc9_16.1: ref @Class.Get.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc9_16.2: @Class.Get.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc9_16.1
// CHECK:STDOUT:     %impl.elem0.loc9_16.1: @Class.Get.%.loc9_16.3 (%.232) = impl_witness_access constants.%Copy.lookup_impl_witness.edd, element0 [symbolic = %impl.elem0.loc9_16.2 (constants.%impl.elem0.8df)]
// CHECK:STDOUT:     %bound_method.loc9_16.1: <bound method> = bound_method %.loc9_16.2, %impl.elem0.loc9_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_16.1: <specific function> = specific_impl_function %impl.elem0.loc9_16.1, @Copy.Op(constants.%T.f92) [symbolic = %specific_impl_fn.loc9_16.2 (constants.%specific_impl_fn.5c4)]
// CHECK:STDOUT:     %bound_method.loc9_16.2: <bound method> = bound_method %.loc9_16.2, %specific_impl_fn.loc9_16.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.Op.call: init @Class.Get.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc9_16.2(%.loc9_16.2) to %.loc7_24
// CHECK:STDOUT:     return %Copy.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.aa1)]
// CHECK:STDOUT:   %.loc15_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc15_12.1 (constants.%.892)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.29a)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.6b4)]
// CHECK:STDOUT:   %.loc15_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc15_12.2 (constants.%.057)]
// CHECK:STDOUT:   %impl.elem0.loc15_12.2: @Class.GetAddr.%.loc15_12.2 (%.057) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.120)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_12.2: <specific function> = specific_impl_function %impl.elem0.loc15_12.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.5a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetAddr.%Class (%Class.8fd)) -> @Class.GetAddr.%ptr.loc13_36.1 (%ptr.2e1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class.8fd) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.aa1) = name_ref x, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:     %.loc15_17: ref @Class.GetAddr.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc13_36.1 (%ptr.2e1) = addr_of %.loc15_17
// CHECK:STDOUT:     %impl.elem0.loc15_12.1: @Class.GetAddr.%.loc15_12.2 (%.057) = impl_witness_access constants.%Copy.lookup_impl_witness.29a, element0 [symbolic = %impl.elem0.loc15_12.2 (constants.%impl.elem0.120)]
// CHECK:STDOUT:     %bound_method.loc15_12.1: <bound method> = bound_method %addr, %impl.elem0.loc15_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_12.1: <specific function> = specific_impl_function %impl.elem0.loc15_12.1, @Copy.Op(constants.%Copy.facet.6b4) [symbolic = %specific_impl_fn.loc15_12.2 (constants.%specific_impl_fn.5a3)]
// CHECK:STDOUT:     %bound_method.loc15_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc15_12.1
// CHECK:STDOUT:     %Copy.Op.call: init @Class.GetAddr.%ptr.loc13_36.1 (%ptr.2e1) = call %bound_method.loc15_12.2(%addr)
// CHECK:STDOUT:     return %Copy.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.4d3) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc22_10: %Class.4d3 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc22_11: %Class.elem.702 = name_ref x, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:   %.loc22_11.1: ref %i32 = class_element_access %x.ref.loc22_10, element0
// CHECK:STDOUT:   %.loc22_11.2: %i32 = acquire_value %.loc22_11.1
// CHECK:STDOUT:   %impl.elem0: %.958 = impl_witness_access constants.%Copy.impl_witness.66f, element0 [concrete = constants.%Int.as.Copy.impl.Op.c85]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %.loc22_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %.loc22_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_11.2(%.loc22_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.4d3) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.4d3 = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %Class.Get.type.189 = specific_constant @Class.%Class.Get.decl, @Class(constants.%Copy.facet.9cb) [concrete = constants.%Class.Get.4b6]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.189 = name_ref Get, %.loc28 [concrete = constants.%Class.Get.4b6]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%Copy.facet.9cb) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   return %Class.Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.047) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.047 = name_ref p, %p
// CHECK:STDOUT:   %.loc34_12.1: ref %Class.4d3 = deref %p.ref
// CHECK:STDOUT:   %.loc34_12.2: %Class.GetAddr.type.dc0 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%Copy.facet.9cb) [concrete = constants.%Class.GetAddr.288]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.dc0 = name_ref GetAddr, %.loc34_12.2 [concrete = constants.%Class.GetAddr.288]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc34_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%Copy.facet.9cb) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_22: <bound method> = bound_method %.loc34_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.235 = call %bound_method.loc34_22(%.loc34_12.1)
// CHECK:STDOUT:   %.loc34_22.1: %ptr.235 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc34_22.2: %ptr.235 = converted %Class.GetAddr.call, %.loc34_22.1
// CHECK:STDOUT:   %.loc34_10.1: ref %i32 = deref %.loc34_22.2
// CHECK:STDOUT:   %.loc34_10.2: %i32 = acquire_value %.loc34_10.1
// CHECK:STDOUT:   %impl.elem0: %.958 = impl_witness_access constants.%Copy.impl_witness.66f, element0 [concrete = constants.%Int.as.Copy.impl.Op.c85]
// CHECK:STDOUT:   %bound_method.loc34_10.1: <bound method> = bound_method %.loc34_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_10.2: <bound method> = bound_method %.loc34_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc34_10.2(%.loc34_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.f92) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.f92
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.91e
// CHECK:STDOUT:   %Class => constants.%Class.8fd
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.aa1
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.16e
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.ff7
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.14f
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.b6b
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.12c
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.7f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T.f92) {
// CHECK:STDOUT:   %T => constants.%T.f92
// CHECK:STDOUT:   %Class => constants.%Class.8fd
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.6e6
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.c769df.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f92) {
// CHECK:STDOUT:   %T => constants.%T.f92
// CHECK:STDOUT:   %Class => constants.%Class.8fd
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.6e6
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.2e1
// CHECK:STDOUT:   %pattern_type.loc13_32 => constants.%pattern_type.95d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%Copy.facet.9cb) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.9cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.4d3
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.702
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.189
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.4b6
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.dc0
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.288
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%Copy.facet.9cb) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.9cb
// CHECK:STDOUT:   %Class => constants.%Class.4d3
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.373
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.702
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.66f
// CHECK:STDOUT:   %.loc9_16.3 => constants.%.958
// CHECK:STDOUT:   %impl.elem0.loc9_16.2 => constants.%Int.as.Copy.impl.Op.c85
// CHECK:STDOUT:   %specific_impl_fn.loc9_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%Copy.facet.9cb) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.9cb
// CHECK:STDOUT:   %Class => constants.%Class.4d3
// CHECK:STDOUT:   %pattern_type.loc13_18 => constants.%pattern_type.373
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %ptr.loc13_36.1 => constants.%ptr.235
// CHECK:STDOUT:   %pattern_type.loc13_32 => constants.%pattern_type.fe8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_36 => constants.%complete_type.3d0
// CHECK:STDOUT:   %require_complete.loc13_22 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.702
// CHECK:STDOUT:   %.loc15_12.1 => constants.%.ff7
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.9c7
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.fff
// CHECK:STDOUT:   %.loc15_12.2 => constants.%.fef
// CHECK:STDOUT:   %impl.elem0.loc15_12.2 => constants.%ptr.as.Copy.impl.Op.624
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
// CHECK:STDOUT:   %pattern_type.466: type = pattern_type %Class [symbolic]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.466
// CHECK:STDOUT: }
// CHECK:STDOUT:

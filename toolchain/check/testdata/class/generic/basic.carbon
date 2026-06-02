// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/basic.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(T:! Core.Copy) {
  fn GetAddr[ref self: Self]() -> T* {
    return &self.k;
  }

  fn GetValue[self: Self]() -> T {
    return self.k;
  }

  var k: T;
}

class Declaration(T:! type);
//@dump-sem-ir-end

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.f49) [symbolic]
// CHECK:STDOUT:   %pattern_type.16a: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f49 [symbolic]
// CHECK:STDOUT:   %ptr.a3f: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.0e7: Core.Form = init_form %ptr.a3f [symbolic]
// CHECK:STDOUT:   %pattern_type.eec: type = pattern_type %ptr.a3f [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.f49) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr: %Class.GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.dedae6.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.c0fdca.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.f49) [symbolic]
// CHECK:STDOUT:   %Class.GetValue: %Class.GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.77e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.28e: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %require_complete.040: <witness> = require_complete_type %ptr.a3f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2dd: <witness> = lookup_impl_witness %T.f49, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e9f10f.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f49) [symbolic]
// CHECK:STDOUT:   %.e16: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e9f10f.2, %T.f49 [symbolic]
// CHECK:STDOUT:   %impl.elem0.41e: %.e16 = impl_witness_access %Copy.lookup_impl_witness.2dd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7ec: <specific function> = specific_impl_function %impl.elem0.41e, @Copy.WithSelf.Op(%T.f49) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.fd9: <witness> = lookup_impl_witness %ptr.a3f, @Copy [symbolic]
// CHECK:STDOUT:   %.7ac: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.a3f, (%Copy.lookup_impl_witness.fd9) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.93e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4e8, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.d58: %.93e = impl_witness_access %Copy.lookup_impl_witness.fd9, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.436: <specific function> = specific_impl_function %impl.elem0.d58, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %Declaration.type: type = generic_class_type @Declaration [concrete]
// CHECK:STDOUT:   %Declaration.generic: %Declaration.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.b72 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f49)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_23.1: type = splice_block %.loc17_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc17_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc17_20.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f49)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.loc5_14.1) [symbolic = %Class.GetAddr.type (constants.%Class.GetAddr.type)]
// CHECK:STDOUT:   %Class.GetAddr: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = struct_value () [symbolic = %Class.GetAddr (constants.%Class.GetAddr)]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.loc5_14.1) [symbolic = %Class.GetValue.type (constants.%Class.GetValue.type)]
// CHECK:STDOUT:   %Class.GetValue: @Class.%Class.GetValue.type (%Class.GetValue.type) = struct_value () [symbolic = %Class.GetValue (constants.%Class.GetValue)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.loc5_14.1 [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.77e)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_14.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.as_type (%T.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.GetAddr.decl: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = fn_decl @Class.GetAddr [symbolic = @Class.%Class.GetAddr (constants.%Class.GetAddr)] {
// CHECK:STDOUT:       %self.param_patt: @Class.GetAddr.%pattern_type.loc6_22 (%pattern_type.16a) = ref_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Class.GetAddr.%pattern_type.loc6_22 (%pattern_type.16a) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetAddr.%pattern_type.loc6_36 (%pattern_type.eec) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetAddr.%pattern_type.loc6_36 (%pattern_type.eec) = return_slot_pattern %return.param_patt, %ptr.loc6_36.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:       %T.as_type.loc6_36.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_36.2: type = converted %T.ref, %T.as_type.loc6_36.2 [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %ptr.loc6_36.2: type = ptr_type %.loc6_36.2 [symbolic = %ptr.loc6_36.1 (constants.%ptr.a3f)]
// CHECK:STDOUT:       %.loc6_36.3: Core.Form = init_form %ptr.loc6_36.2 [symbolic = %.loc6_36.1 (constants.%.0e7)]
// CHECK:STDOUT:       %self.param: ref @Class.GetAddr.%Class (%Class) = ref_param call_param0
// CHECK:STDOUT:       %.loc6_24.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc6_24.2: type = specific_constant constants.%Class, @Class(constants.%T.f49) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_24.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: ref @Class.GetAddr.%Class (%Class) = ref_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetAddr.%ptr.loc6_36.1 (%ptr.a3f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetAddr.%ptr.loc6_36.1 (%ptr.a3f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetValue.decl: @Class.%Class.GetValue.type (%Class.GetValue.type) = fn_decl @Class.GetValue [symbolic = @Class.%Class.GetValue (constants.%Class.GetValue)] {
// CHECK:STDOUT:       %self.param_patt: @Class.GetValue.%pattern_type.loc10_19 (%pattern_type.16a) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Class.GetValue.%pattern_type.loc10_19 (%pattern_type.16a) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetValue.%pattern_type.loc10_32 (%pattern_type.c0fdca.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetValue.%pattern_type.loc10_32 (%pattern_type.c0fdca.2) = return_slot_pattern %return.param_patt, %.loc10_32.3 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:       %T.as_type.loc10_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_32.3: type = converted %T.ref, %T.as_type.loc10_32.2 [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_32.4: Core.Form = init_form %.loc10_32.3 [symbolic = %.loc10_32.2 (constants.%.dedae6.2)]
// CHECK:STDOUT:       %self.param: @Class.GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_21.2: type = specific_constant constants.%Class, @Class(constants.%T.f49) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetValue.%Class (%Class) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc14: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc15_1.1: <witness> = complete_type_witness constants.%struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc15_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .GetAddr = %Class.GetAddr.decl
// CHECK:STDOUT:     .GetValue = %Class.GetValue.decl
// CHECK:STDOUT:     .k = %.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(%T.loc17_20.2: type) {
// CHECK:STDOUT:   %T.loc17_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc17_20.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc6_22: type = pattern_type %Class [symbolic = %pattern_type.loc6_22 (constants.%pattern_type.16a)]
// CHECK:STDOUT:   %T.as_type.loc6_36.1: type = facet_access_type %T [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %ptr.loc6_36.1: type = ptr_type %T.as_type.loc6_36.1 [symbolic = %ptr.loc6_36.1 (constants.%ptr.a3f)]
// CHECK:STDOUT:   %.loc6_36.1: Core.Form = init_form %ptr.loc6_36.1 [symbolic = %.loc6_36.1 (constants.%.0e7)]
// CHECK:STDOUT:   %pattern_type.loc6_36: type = pattern_type %ptr.loc6_36.1 [symbolic = %pattern_type.loc6_36 (constants.%pattern_type.eec)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_22: <witness> = require_complete_type %Class [symbolic = %require_complete.loc6_22 (constants.%require_complete.28e)]
// CHECK:STDOUT:   %require_complete.loc6_32: <witness> = require_complete_type %ptr.loc6_36.1 [symbolic = %require_complete.loc6_32 (constants.%require_complete.040)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc6_36.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %.loc7_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc6_36.1) [symbolic = %.loc7_12.1 (constants.%.7ac)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc6_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.fd9)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc6_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.4e8)]
// CHECK:STDOUT:   %.loc7_12.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc7_12.2 (constants.%.93e)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.GetAddr.%.loc7_12.2 (%.93e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.d58)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.436)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class)) -> out %return.param: @Class.GetAddr.%ptr.loc6_36.1 (%ptr.a3f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc7_17: ref @Class.GetAddr.%T.as_type.loc6_36.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc6_36.1 (%ptr.a3f) = addr_of %.loc7_17
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.GetAddr.%.loc7_12.2 (%.93e) = impl_witness_access constants.%Copy.lookup_impl_witness.fd9, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.d58)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %addr, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.436)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc6_36.1 (%ptr.a3f) = call %bound_method.loc7_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetValue(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_19: type = pattern_type %Class [symbolic = %pattern_type.loc10_19 (constants.%pattern_type.16a)]
// CHECK:STDOUT:   %T.as_type.loc10_32.1: type = facet_access_type %T [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc10_32.2: Core.Form = init_form %T.as_type.loc10_32.1 [symbolic = %.loc10_32.2 (constants.%.dedae6.2)]
// CHECK:STDOUT:   %pattern_type.loc10_32: type = pattern_type %T.as_type.loc10_32.1 [symbolic = %pattern_type.loc10_32 (constants.%pattern_type.c0fdca.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10 (constants.%require_complete.28e)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc10_32.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %T.as_type.loc10_32.1 [symbolic = %require_complete.loc11 (constants.%require_complete.77e)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e9f10f.2)]
// CHECK:STDOUT:   %.loc11_16.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc11_16.3 (constants.%.e16)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2dd)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.GetValue.%.loc11_16.3 (%.e16) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetValue.%Class (%Class)) -> out %return.param: @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = acquire_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.GetValue.%.loc11_16.3 (%.e16) = impl_witness_access constants.%Copy.lookup_impl_witness.2dd, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.WithSelf.Op(constants.%T.f49) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc10_32.1: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) to %.loc10_32.1 = call %bound_method.loc11_16.2(%.loc11_16.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.f49) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.f49
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr
// CHECK:STDOUT:   %Class.GetValue.type => constants.%Class.GetValue.type
// CHECK:STDOUT:   %Class.GetValue => constants.%Class.GetValue
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.77e
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f49) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc6_22 => constants.%pattern_type.16a
// CHECK:STDOUT:   %T.as_type.loc6_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc6_36.1 => constants.%ptr.a3f
// CHECK:STDOUT:   %.loc6_36.1 => constants.%.0e7
// CHECK:STDOUT:   %pattern_type.loc6_36 => constants.%pattern_type.eec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetValue(constants.%T.f49) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_19 => constants.%pattern_type.16a
// CHECK:STDOUT:   %T.as_type.loc10_32.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.dedae6.2
// CHECK:STDOUT:   %pattern_type.loc10_32 => constants.%pattern_type.c0fdca.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc17_20.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:

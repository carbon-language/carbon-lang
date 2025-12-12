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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.035) [symbolic]
// CHECK:STDOUT:   %pattern_type.893: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %ptr.e7d: type = ptr_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.c6d: form = init_form %ptr.e7d, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.65a: type = pattern_type %ptr.e7d [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.035) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr: %Class.GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.075d25.1: form = init_form %T.binding.as_type, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.035) [symbolic]
// CHECK:STDOUT:   %Class.GetValue: %Class.GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.9dc: <witness> = require_complete_type %ptr.e7d [symbolic]
// CHECK:STDOUT:   %require_complete.904: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %.72e: type = fn_type_with_self_type %Copy.Op.type, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.07b: %.72e = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c9: <specific function> = specific_impl_function %impl.elem0.07b, @Copy.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.3a3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e7: <witness> = lookup_impl_witness %ptr.e7d, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e7d, (%Copy.lookup_impl_witness.2e7) [symbolic]
// CHECK:STDOUT:   %.f11: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.a55: %.f11 = impl_witness_access %Copy.lookup_impl_witness.2e7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0df: <specific function> = specific_impl_function %impl.elem0.a55, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.035)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc17_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc17_19.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.035)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.loc5_13.1) [symbolic = %Class.GetAddr.type (constants.%Class.GetAddr.type)]
// CHECK:STDOUT:   %Class.GetAddr: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = struct_value () [symbolic = %Class.GetAddr (constants.%Class.GetAddr)]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.loc5_13.1) [symbolic = %Class.GetValue.type (constants.%Class.GetValue.type)]
// CHECK:STDOUT:   %Class.GetValue: @Class.%Class.GetValue.type (%Class.GetValue.type) = struct_value () [symbolic = %Class.GetValue (constants.%Class.GetValue)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc5_13.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.GetAddr.decl: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = fn_decl @Class.GetAddr [symbolic = @Class.%Class.GetAddr (constants.%Class.GetAddr)] {
// CHECK:STDOUT:       %self.patt: @Class.GetAddr.%pattern_type.loc6_18 (%pattern_type.893) = ref_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetAddr.%pattern_type.loc6_18 (%pattern_type.893) = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetAddr.%pattern_type.loc6_32 (%pattern_type.65a) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetAddr.%pattern_type.loc6_32 (%pattern_type.65a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_36.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %ptr.loc6_36.2: type = ptr_type %.loc6_36.2 [symbolic = %ptr.loc6_36.1 (constants.%ptr.e7d)]
// CHECK:STDOUT:       %.loc6_36.3: form = init_form %ptr.loc6_36.2, call_param1 [symbolic = %.loc6_36.1 (constants.%.c6d)]
// CHECK:STDOUT:       %self.param: ref @Class.GetAddr.%Class (%Class) = ref_param call_param0
// CHECK:STDOUT:       %.loc6_24.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc6_24.2: type = specific_constant constants.%Class, @Class(constants.%T.035) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_24.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: ref @Class.GetAddr.%Class (%Class) = ref_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetAddr.%ptr.loc6_36.1 (%ptr.e7d) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetAddr.%ptr.loc6_36.1 (%ptr.e7d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetValue.decl: @Class.%Class.GetValue.type (%Class.GetValue.type) = fn_decl @Class.GetValue [symbolic = @Class.%Class.GetValue (constants.%Class.GetValue)] {
// CHECK:STDOUT:       %self.patt: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.893) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.893) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetValue.%pattern_type.loc10_29 (%pattern_type.9b9f0c.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetValue.%pattern_type.loc10_29 (%pattern_type.9b9f0c.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc10_32.3: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc10_32.4: form = init_form %.loc10_32.3, call_param1 [symbolic = %.loc10_32.2 (constants.%.075d25.1)]
// CHECK:STDOUT:       %self.param: @Class.GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_21.2: type = specific_constant constants.%Class, @Class(constants.%T.035) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetValue.%Class (%Class) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetValue.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetValue.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc14_10: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc14_8: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc15_1.1: <witness> = complete_type_witness constants.%struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc15_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .GetAddr = %Class.GetAddr.decl
// CHECK:STDOUT:     .GetValue = %Class.GetValue.decl
// CHECK:STDOUT:     .k = %.loc14_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(%T.loc17_19.2: type) {
// CHECK:STDOUT:   %T.loc17_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc17_19.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.035)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %Class [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.893)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %ptr.loc6_36.1: type = ptr_type %T.binding.as_type [symbolic = %ptr.loc6_36.1 (constants.%ptr.e7d)]
// CHECK:STDOUT:   %.loc6_36.1: form = init_form %ptr.loc6_36.1, call_param1 [symbolic = %.loc6_36.1 (constants.%.c6d)]
// CHECK:STDOUT:   %pattern_type.loc6_32: type = pattern_type %ptr.loc6_36.1 [symbolic = %pattern_type.loc6_32 (constants.%pattern_type.65a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_36: <witness> = require_complete_type %ptr.loc6_36.1 [symbolic = %require_complete.loc6_36 (constants.%require_complete.9dc)]
// CHECK:STDOUT:   %require_complete.loc6_22: <witness> = require_complete_type %Class [symbolic = %require_complete.loc6_22 (constants.%require_complete.904)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %.loc7_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc7_12.1 (constants.%.3a3)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc6_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc6_36.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc7_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc7_12.2 (constants.%.f11)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.GetAddr.%.loc7_12.2 (%.f11) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.a55)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.0df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetAddr.%Class (%Class)) -> @Class.GetAddr.%ptr.loc6_36.1 (%ptr.e7d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc7_17: ref @Class.GetAddr.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc6_36.1 (%ptr.e7d) = addr_of %.loc7_17
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.GetAddr.%.loc7_12.2 (%.f11) = impl_witness_access constants.%Copy.lookup_impl_witness.2e7, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.a55)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %addr, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.0df)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %Copy.Op.call: init @Class.GetAddr.%ptr.loc6_36.1 (%ptr.e7d) = call %bound_method.loc7_12.2(%addr)
// CHECK:STDOUT:     return %Copy.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetValue(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.035)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_15: type = pattern_type %Class [symbolic = %pattern_type.loc10_15 (constants.%pattern_type.893)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %.loc10_32.2: form = init_form %T.binding.as_type, call_param1 [symbolic = %.loc10_32.2 (constants.%.075d25.1)]
// CHECK:STDOUT:   %pattern_type.loc10_29: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc10_29 (constants.%pattern_type.9b9f0c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_19: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10_19 (constants.%require_complete.904)]
// CHECK:STDOUT:   %require_complete.loc10_32: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc10_32 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %.loc11_16.3: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc11_16.3 (constants.%.72e)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.GetValue.%.loc11_16.3 (%.72e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetValue.%Class (%Class)) -> %return.param: @Class.GetValue.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.GetValue.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.GetValue.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.GetValue.%.loc11_16.3 (%.72e) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc10_32.1: ref @Class.GetValue.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.Op.call: init @Class.GetValue.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc11_16.2(%.loc11_16.2) to %.loc10_32.1
// CHECK:STDOUT:     return %Copy.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.035) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.035
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr
// CHECK:STDOUT:   %Class.GetValue.type => constants.%Class.GetValue.type
// CHECK:STDOUT:   %Class.GetValue => constants.%Class.GetValue
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.67c
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.035) {
// CHECK:STDOUT:   %T => constants.%T.035
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.893
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %ptr.loc6_36.1 => constants.%ptr.e7d
// CHECK:STDOUT:   %.loc6_36.1 => constants.%.c6d
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.65a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetValue(constants.%T.035) {
// CHECK:STDOUT:   %T => constants.%T.035
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_15 => constants.%pattern_type.893
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.075d25.1
// CHECK:STDOUT:   %pattern_type.loc10_29 => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc17_19.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:

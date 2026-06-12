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
  fn GetAddr(ref self) -> T* {
    return &self.k;
  }

  fn GetValue(self) -> T {
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
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %pattern_type.e18: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %self.param_patt.f0d: %pattern_type.e18 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.282: %pattern_type.e18 = at_binding_pattern self, %self.param_patt.f0d [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %ptr.215: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.4fb: Core.Form = init_form %ptr.215 [symbolic]
// CHECK:STDOUT:   %pattern_type.d5b: type = pattern_type %ptr.215 [symbolic]
// CHECK:STDOUT:   %return.param_patt.4df: %pattern_type.d5b = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c21: %pattern_type.d5b = return_slot_pattern %return.param_patt.4df, %ptr.215 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr: %Class.GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %self.param_patt.aaa: %pattern_type.e18 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d8a: %pattern_type.e18 = at_binding_pattern self, %self.param_patt.aaa [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.GetValue: %Class.GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %k.patt: %pattern_type.2c66b4.2 = ref_binding_pattern k [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.1dd: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %require_complete.94b: <witness> = require_complete_type %ptr.215 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cf6: <witness> = lookup_impl_witness %ptr.215, @Copy [symbolic]
// CHECK:STDOUT:   %.a33: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.215, (%Copy.lookup_impl_witness.cf6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.45d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e0, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.c5d: %.45d = impl_witness_access %Copy.lookup_impl_witness.cf6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5cf: <specific function> = specific_impl_function %impl.elem0.c5d, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt.loc17_20.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_20.2 (constants.%T.patt.d47)]
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
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc5_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.loc5_14.1) [symbolic = %Class.GetAddr.type (constants.%Class.GetAddr.type)]
// CHECK:STDOUT:   %Class.GetAddr: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = struct_value () [symbolic = %Class.GetAddr (constants.%Class.GetAddr)]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.loc5_14.1) [symbolic = %Class.GetValue.type (constants.%Class.GetValue.type)]
// CHECK:STDOUT:   %Class.GetValue: @Class.%Class.GetValue.type (%Class.GetValue.type) = struct_value () [symbolic = %Class.GetValue (constants.%Class.GetValue)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.loc5_14.1 [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %k.patt: @Class.%pattern_type (%pattern_type.2c66b4.2) = ref_binding_pattern k [symbolic = %k.patt (constants.%k.patt)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_14.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.as_type (%T.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.GetAddr.decl: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = fn_decl @Class.GetAddr [symbolic = @Class.%Class.GetAddr (constants.%Class.GetAddr)] {
// CHECK:STDOUT:       %self.param_patt.loc6_18.1: @Class.GetAddr.%pattern_type.loc6_18 (%pattern_type.e18) = ref_param_pattern [symbolic = %self.param_patt.loc6_18.2 (constants.%self.param_patt.f0d)]
// CHECK:STDOUT:       %self.patt.loc6_18.1: @Class.GetAddr.%pattern_type.loc6_18 (%pattern_type.e18) = at_binding_pattern self, %self.param_patt.loc6_18.1 [symbolic = %self.patt.loc6_18.2 (constants.%self.patt.282)]
// CHECK:STDOUT:       %return.param_patt.loc6_28.1: @Class.GetAddr.%pattern_type.loc6_28 (%pattern_type.d5b) = out_param_pattern [symbolic = %return.param_patt.loc6_28.2 (constants.%return.param_patt.4df)]
// CHECK:STDOUT:       %return.patt.loc6_24.1: @Class.GetAddr.%pattern_type.loc6_28 (%pattern_type.d5b) = return_slot_pattern %return.param_patt.loc6_28.1, %ptr.loc6_28.2 [symbolic = %return.patt.loc6_24.2 (constants.%return.patt.c21)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc6_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %T.ref, %T.as_type.loc6_28.2 [symbolic = %T.as_type.loc6_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %ptr.loc6_28.2: type = ptr_type %.loc6_28.2 [symbolic = %ptr.loc6_28.1 (constants.%ptr.215)]
// CHECK:STDOUT:       %.loc6_28.3: Core.Form = init_form %ptr.loc6_28.2 [symbolic = %.loc6_28.1 (constants.%.4fb)]
// CHECK:STDOUT:       %self.param: ref @Class.GetAddr.%Class (%Class) = ref_param call_param0
// CHECK:STDOUT:       %.loc6_18.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc6_18.2: type = specific_constant constants.%Class, @Class(constants.%T.f84) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_18.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: ref @Class.GetAddr.%Class (%Class) = ref_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetAddr.%ptr.loc6_28.1 (%ptr.215) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetAddr.%ptr.loc6_28.1 (%ptr.215) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetValue.decl: @Class.%Class.GetValue.type (%Class.GetValue.type) = fn_decl @Class.GetValue [symbolic = @Class.%Class.GetValue (constants.%Class.GetValue)] {
// CHECK:STDOUT:       %self.param_patt.loc10_15.1: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.e18) = value_param_pattern [symbolic = %self.param_patt.loc10_15.2 (constants.%self.param_patt.aaa)]
// CHECK:STDOUT:       %self.patt.loc10_15.1: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.e18) = at_binding_pattern self, %self.param_patt.loc10_15.1 [symbolic = %self.patt.loc10_15.2 (constants.%self.patt.d8a)]
// CHECK:STDOUT:       %return.param_patt.loc10_24.1: @Class.GetValue.%pattern_type.loc10_24 (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_24.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:       %return.patt.loc10_21.1: @Class.GetValue.%pattern_type.loc10_24 (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_24.1, %.loc10_24.3 [symbolic = %return.patt.loc10_21.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc10_24.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_24.3: type = converted %T.ref, %T.as_type.loc10_24.2 [symbolic = %T.as_type.loc10_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_24.4: Core.Form = init_form %.loc10_24.3 [symbolic = %.loc10_24.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:       %self.param: @Class.GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_15.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_15.2: type = specific_constant constants.%Class, @Class(constants.%T.f84) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_15.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetValue.%Class (%Class) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) = return_slot %return.param
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
// CHECK:STDOUT:   %T.patt.loc17_20.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_20.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc17_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc17_20.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %Class [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.e18)]
// CHECK:STDOUT:   %self.param_patt.loc6_18.2: @Class.GetAddr.%pattern_type.loc6_18 (%pattern_type.e18) = ref_param_pattern [symbolic = %self.param_patt.loc6_18.2 (constants.%self.param_patt.f0d)]
// CHECK:STDOUT:   %self.patt.loc6_18.2: @Class.GetAddr.%pattern_type.loc6_18 (%pattern_type.e18) = at_binding_pattern self, %self.param_patt.loc6_18.2 [symbolic = %self.patt.loc6_18.2 (constants.%self.patt.282)]
// CHECK:STDOUT:   %T.as_type.loc6_28.1: type = facet_access_type %T [symbolic = %T.as_type.loc6_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %ptr.loc6_28.1: type = ptr_type %T.as_type.loc6_28.1 [symbolic = %ptr.loc6_28.1 (constants.%ptr.215)]
// CHECK:STDOUT:   %.loc6_28.1: Core.Form = init_form %ptr.loc6_28.1 [symbolic = %.loc6_28.1 (constants.%.4fb)]
// CHECK:STDOUT:   %pattern_type.loc6_28: type = pattern_type %ptr.loc6_28.1 [symbolic = %pattern_type.loc6_28 (constants.%pattern_type.d5b)]
// CHECK:STDOUT:   %return.param_patt.loc6_28.2: @Class.GetAddr.%pattern_type.loc6_28 (%pattern_type.d5b) = out_param_pattern [symbolic = %return.param_patt.loc6_28.2 (constants.%return.param_patt.4df)]
// CHECK:STDOUT:   %return.patt.loc6_24.2: @Class.GetAddr.%pattern_type.loc6_28 (%pattern_type.d5b) = return_slot_pattern %return.param_patt.loc6_28.2, %ptr.loc6_28.1 [symbolic = %return.patt.loc6_24.2 (constants.%return.patt.c21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_18: <witness> = require_complete_type %Class [symbolic = %require_complete.loc6_18 (constants.%require_complete.1dd)]
// CHECK:STDOUT:   %require_complete.loc6_24: <witness> = require_complete_type %ptr.loc6_28.1 [symbolic = %require_complete.loc6_24 (constants.%require_complete.94b)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc6_28.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %.loc7_12.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc6_28.1) [symbolic = %.loc7_12.3 (constants.%.a33)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc6_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cf6)]
// CHECK:STDOUT:   %Copy.facet.loc7_12.3: %Copy.type = facet_value %ptr.loc6_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc7_12.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc7_12.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.6e0)]
// CHECK:STDOUT:   %.loc7_12.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc7_12.3 [symbolic = %.loc7_12.4 (constants.%.45d)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.GetAddr.%.loc7_12.4 (%.45d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.c5d)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.WithSelf.Op(%Copy.facet.loc7_12.3) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.5cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @Class.GetAddr.%Class (%Class)) -> out %return.param: @Class.GetAddr.%ptr.loc6_28.1 (%ptr.215) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @Class.GetAddr.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc7_17: ref @Class.GetAddr.%T.as_type.loc6_28.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc6_28.1 (%ptr.215) = addr_of %.loc7_17
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.GetAddr.%.loc7_12.4 (%.45d) = impl_witness_access constants.%Copy.lookup_impl_witness.cf6, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.c5d)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %addr, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %Copy.facet.loc7_12.1: %Copy.type = facet_value constants.%ptr.215, (constants.%Copy.lookup_impl_witness.cf6) [symbolic = %Copy.facet.loc7_12.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc7_12.1: %Copy.type = converted constants.%ptr.215, %Copy.facet.loc7_12.1 [symbolic = %Copy.facet.loc7_12.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc7_12.2: %Copy.type = facet_value constants.%ptr.215, (constants.%Copy.lookup_impl_witness.cf6) [symbolic = %Copy.facet.loc7_12.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc7_12.2: %Copy.type = converted constants.%ptr.215, %Copy.facet.loc7_12.2 [symbolic = %Copy.facet.loc7_12.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.5cf)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetAddr.%ptr.loc6_28.1 (%ptr.215) = call %bound_method.loc7_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetValue(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_15: type = pattern_type %Class [symbolic = %pattern_type.loc10_15 (constants.%pattern_type.e18)]
// CHECK:STDOUT:   %self.param_patt.loc10_15.2: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.e18) = value_param_pattern [symbolic = %self.param_patt.loc10_15.2 (constants.%self.param_patt.aaa)]
// CHECK:STDOUT:   %self.patt.loc10_15.2: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.e18) = at_binding_pattern self, %self.param_patt.loc10_15.2 [symbolic = %self.patt.loc10_15.2 (constants.%self.patt.d8a)]
// CHECK:STDOUT:   %T.as_type.loc10_24.1: type = facet_access_type %T [symbolic = %T.as_type.loc10_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc10_24.2: Core.Form = init_form %T.as_type.loc10_24.1 [symbolic = %.loc10_24.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %pattern_type.loc10_24: type = pattern_type %T.as_type.loc10_24.1 [symbolic = %pattern_type.loc10_24 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %return.param_patt.loc10_24.2: @Class.GetValue.%pattern_type.loc10_24 (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_24.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc10_21.2: @Class.GetValue.%pattern_type.loc10_24 (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_24.2, %T.as_type.loc10_24.1 [symbolic = %return.patt.loc10_21.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10 (constants.%require_complete.1dd)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc10_24.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %T.as_type.loc10_24.1 [symbolic = %require_complete.loc11 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc11_16.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc11_16.5 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.GetValue.%.loc11_16.5 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetValue.%Class (%Class)) -> out %return.param: @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) = acquire_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.GetValue.%.loc11_16.5 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %.loc11_16.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc11_16.4: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc10_24.1: ref @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.GetValue.%T.as_type.loc10_24.1 (%T.as_type) to %.loc10_24.1 = call %bound_method.loc11_16.2(%.loc11_16.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.f84
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr
// CHECK:STDOUT:   %Class.GetValue.type => constants.%Class.GetValue.type
// CHECK:STDOUT:   %Class.GetValue => constants.%Class.GetValue
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %k.patt => constants.%k.patt
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.e18
// CHECK:STDOUT:   %self.param_patt.loc6_18.2 => constants.%self.param_patt.f0d
// CHECK:STDOUT:   %self.patt.loc6_18.2 => constants.%self.patt.282
// CHECK:STDOUT:   %T.as_type.loc6_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc6_28.1 => constants.%ptr.215
// CHECK:STDOUT:   %.loc6_28.1 => constants.%.4fb
// CHECK:STDOUT:   %pattern_type.loc6_28 => constants.%pattern_type.d5b
// CHECK:STDOUT:   %return.param_patt.loc6_28.2 => constants.%return.param_patt.4df
// CHECK:STDOUT:   %return.patt.loc6_24.2 => constants.%return.patt.c21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetValue(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_15 => constants.%pattern_type.e18
// CHECK:STDOUT:   %self.param_patt.loc10_15.2 => constants.%self.param_patt.aaa
// CHECK:STDOUT:   %self.patt.loc10_15.2 => constants.%self.patt.d8a
// CHECK:STDOUT:   %T.as_type.loc10_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc10_24.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type.loc10_24 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %return.param_patt.loc10_24.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc10_21.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc17_20.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc17_20.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:

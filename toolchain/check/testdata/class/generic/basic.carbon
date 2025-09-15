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
  fn GetAddr[addr self: Self*]() -> T* {
    return &self->k;
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
// CHECK:STDOUT:   %T.be8: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %ptr.818: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.e8f: type = pattern_type %ptr.818 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.be8 [symbolic]
// CHECK:STDOUT:   %ptr.f3f: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.f74: type = pattern_type %ptr.f3f [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr: %Class.GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.38d: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.965801.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.GetValue: %Class.GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.07c: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.dd9: <witness> = require_complete_type %ptr.f3f [symbolic]
// CHECK:STDOUT:   %require_complete.70e: <witness> = require_complete_type %ptr.818 [symbolic]
// CHECK:STDOUT:   %require_complete.3e3: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15: <witness> = lookup_impl_witness %T.be8, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.021: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.e15) [symbolic]
// CHECK:STDOUT:   %.c50: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.021 [symbolic]
// CHECK:STDOUT:   %impl.elem0.792: %.c50 = impl_witness_access %Copy.lookup_impl_witness.e15, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.547: <specific function> = specific_impl_function %impl.elem0.792, @Copy.Op(%Copy.facet.021) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.046: <witness> = lookup_impl_witness %ptr.f3f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.c30: %Copy.type = facet_value %ptr.f3f, (%Copy.lookup_impl_witness.046) [symbolic]
// CHECK:STDOUT:   %.10a: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c30 [symbolic]
// CHECK:STDOUT:   %impl.elem0.823: %.10a = impl_witness_access %Copy.lookup_impl_witness.046, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.831: <specific function> = specific_impl_function %impl.elem0.823, @Copy.Op(%Copy.facet.c30) [symbolic]
// CHECK:STDOUT:   %Declaration.type: type = generic_class_type @Declaration [concrete]
// CHECK:STDOUT:   %Declaration.generic: %Declaration.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.322 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc17_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_19.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.e8f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.e8f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc5_28.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.818) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc5_28.3: type = specific_constant constants.%Class, @Class(constants.%T.be8) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_28.3 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.818) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.loc5_13.1) [symbolic = %Class.GetAddr.type (constants.%Class.GetAddr.type)]
// CHECK:STDOUT:   %Class.GetAddr: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = struct_value () [symbolic = %Class.GetAddr (constants.%Class.GetAddr)]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.loc5_13.1) [symbolic = %Class.GetValue.type (constants.%Class.GetValue.type)]
// CHECK:STDOUT:   %Class.GetValue: @Class.%Class.GetValue.type (%Class.GetValue.type) = struct_value () [symbolic = %Class.GetValue (constants.%Class.GetValue)]
// CHECK:STDOUT:   %T.as_type.loc14_10.2: type = facet_access_type %T.loc5_13.1 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc14_10.2 [symbolic = %require_complete (constants.%require_complete.07c)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc14_10.2 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.as_type.loc14_10.2 (%T.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.GetAddr.decl: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = fn_decl @Class.GetAddr [symbolic = @Class.%Class.GetAddr (constants.%Class.GetAddr)] {
// CHECK:STDOUT:       %self.patt: @Class.GetAddr.%pattern_type.loc6_19 (%pattern_type.e8f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetAddr.%pattern_type.loc6_19 (%pattern_type.e8f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetAddr.%pattern_type.loc6_34 (%pattern_type.f74) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetAddr.%pattern_type.loc6_34 (%pattern_type.f74) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc6_38.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_38.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_38: type = converted %T.ref, %T.as_type.loc6_38.2 [symbolic = %T.as_type.loc6_38.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %ptr.loc6_38.2: type = ptr_type %.loc6_38 [symbolic = %ptr.loc6_38.1 (constants.%ptr.f3f)]
// CHECK:STDOUT:       %self.param: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.818) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29: type = splice_block %ptr.loc6_29.2 [symbolic = %ptr.loc6_29.1 (constants.%ptr.818)] {
// CHECK:STDOUT:         %.loc6_25: type = specific_constant constants.%Class, @Class(constants.%T.be8) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_25 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %ptr.loc6_29.2: type = ptr_type %Self.ref [symbolic = %ptr.loc6_29.1 (constants.%ptr.818)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.818) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetAddr.%ptr.loc6_38.1 (%ptr.f3f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetAddr.%ptr.loc6_38.1 (%ptr.f3f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetValue.decl: @Class.%Class.GetValue.type (%Class.GetValue.type) = fn_decl @Class.GetValue [symbolic = @Class.%Class.GetValue (constants.%Class.GetValue)] {
// CHECK:STDOUT:       %self.patt: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.38d) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.38d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetValue.%pattern_type.loc10_29 (%pattern_type.965801.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetValue.%pattern_type.loc10_29 (%pattern_type.965801.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc10_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_32: type = converted %T.ref, %T.as_type.loc10_32.2 [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %self.param: @Class.GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_21.2: type = specific_constant constants.%Class, @Class(constants.%T.be8) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetValue.%Class (%Class) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type.loc14_10.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc14_10: type = converted %T.ref, %T.as_type.loc14_10.1 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc14_8: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T.be8) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
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
// CHECK:STDOUT:   %T.loc17_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_19.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr.loc6_29.1: type = ptr_type %Class [symbolic = %ptr.loc6_29.1 (constants.%ptr.818)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %ptr.loc6_29.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.e8f)]
// CHECK:STDOUT:   %T.as_type.loc6_38.1: type = facet_access_type %T [symbolic = %T.as_type.loc6_38.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %ptr.loc6_38.1: type = ptr_type %T.as_type.loc6_38.1 [symbolic = %ptr.loc6_38.1 (constants.%ptr.f3f)]
// CHECK:STDOUT:   %pattern_type.loc6_34: type = pattern_type %ptr.loc6_38.1 [symbolic = %pattern_type.loc6_34 (constants.%pattern_type.f74)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_34: <witness> = require_complete_type %ptr.loc6_38.1 [symbolic = %require_complete.loc6_34 (constants.%require_complete.dd9)]
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %ptr.loc6_29.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.70e)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %Class [symbolic = %require_complete.loc7 (constants.%require_complete.3e3)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc6_38.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc6_38.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.046)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc6_38.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.c30)]
// CHECK:STDOUT:   %.loc7_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc7_12.2 (constants.%.10a)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.GetAddr.%.loc7_12.2 (%.10a) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.823)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.831)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.818)) -> @Class.GetAddr.%ptr.loc6_38.1 (%ptr.f3f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.818) = name_ref self, %self
// CHECK:STDOUT:     %.loc7_17.1: ref @Class.GetAddr.%Class (%Class) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @Class.GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc7_17.2: ref @Class.GetAddr.%T.as_type.loc6_38.1 (%T.as_type) = class_element_access %.loc7_17.1, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc6_38.1 (%ptr.f3f) = addr_of %.loc7_17.2
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.GetAddr.%.loc7_12.2 (%.10a) = impl_witness_access constants.%Copy.lookup_impl_witness.046, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.823)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %addr, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.Op(constants.%Copy.facet.c30) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.831)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %.loc7_12.1: init @Class.GetAddr.%ptr.loc6_38.1 (%ptr.f3f) = call %bound_method.loc7_12.2(%addr)
// CHECK:STDOUT:     return %.loc7_12.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetValue(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_15: type = pattern_type %Class [symbolic = %pattern_type.loc10_15 (constants.%pattern_type.38d)]
// CHECK:STDOUT:   %T.as_type.loc10_32.1: type = facet_access_type %T [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc10_29: type = pattern_type %T.as_type.loc10_32.1 [symbolic = %pattern_type.loc10_29 (constants.%pattern_type.965801.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10 (constants.%require_complete.3e3)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc10_32.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %T.as_type.loc10_32.1 [symbolic = %require_complete.loc11 (constants.%require_complete.07c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc10_32.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.021)]
// CHECK:STDOUT:   %.loc11_16.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc11_16.4 (constants.%.c50)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.GetValue.%.loc11_16.4 (%.c50) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.792)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.547)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetValue.%Class (%Class)) -> %return.param: @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = bind_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.GetValue.%.loc11_16.4 (%.c50) = impl_witness_access constants.%Copy.lookup_impl_witness.e15, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.792)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.Op(constants.%Copy.facet.021) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.547)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc10_29: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = splice_block %return {}
// CHECK:STDOUT:     %.loc11_16.3: init @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = call %bound_method.loc11_16.2(%.loc11_16.2) to %.loc10_29
// CHECK:STDOUT:     return %.loc11_16.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.818)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.e8f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.818)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.be8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr
// CHECK:STDOUT:   %Class.GetValue.type => constants.%Class.GetValue.type
// CHECK:STDOUT:   %Class.GetValue => constants.%Class.GetValue
// CHECK:STDOUT:   %T.as_type.loc14_10.2 => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.07c
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.818
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.e8f
// CHECK:STDOUT:   %T.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc6_38.1 => constants.%ptr.f3f
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.f74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetValue(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_15 => constants.%pattern_type.38d
// CHECK:STDOUT:   %T.as_type.loc10_32.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc10_29 => constants.%pattern_type.965801.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr => constants.%ptr.818
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e8f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc17_19.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:

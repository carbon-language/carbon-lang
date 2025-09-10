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
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.578) [symbolic]
// CHECK:STDOUT:   %ptr.0b3: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.417: type = pattern_type %ptr.0b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.578 [symbolic]
// CHECK:STDOUT:   %ptr.a75: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.b45: type = pattern_type %ptr.a75 [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.578) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr: %Class.GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.47a: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.f8cebc.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.578) [symbolic]
// CHECK:STDOUT:   %Class.GetValue: %Class.GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.ecc: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T.578) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T.578) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.999: <witness> = require_complete_type %ptr.a75 [symbolic]
// CHECK:STDOUT:   %require_complete.27d: <witness> = require_complete_type %ptr.0b3 [symbolic]
// CHECK:STDOUT:   %require_complete.ef5: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3ba: <witness> = lookup_impl_witness %T.578, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.72e: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.72e [symbolic]
// CHECK:STDOUT:   %impl.elem0.56e: %.671 = impl_witness_access %Copy.lookup_impl_witness.3ba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fb7: <specific function> = specific_impl_function %impl.elem0.56e, @Copy.Op(%Copy.facet.72e) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.23e: <witness> = lookup_impl_witness %ptr.a75, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.d7e: %Copy.type = facet_value %ptr.a75, (%Copy.lookup_impl_witness.23e) [symbolic]
// CHECK:STDOUT:   %.30e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.d7e [symbolic]
// CHECK:STDOUT:   %impl.elem0.cb9: %.30e = impl_witness_access %Copy.lookup_impl_witness.23e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e97: <specific function> = specific_impl_function %impl.elem0.cb9, @Copy.Op(%Copy.facet.d7e) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.578)]
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
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.417) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.417) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc5_28.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.0b3) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc5_28.3: type = specific_constant constants.%Class, @Class(constants.%T.578) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_28.3 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.0b3) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.loc5_13.1) [symbolic = %Class.GetAddr.type (constants.%Class.GetAddr.type)]
// CHECK:STDOUT:   %Class.GetAddr: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = struct_value () [symbolic = %Class.GetAddr (constants.%Class.GetAddr)]
// CHECK:STDOUT:   %Class.GetValue.type: type = fn_type @Class.GetValue, @Class(%T.loc5_13.1) [symbolic = %Class.GetValue.type (constants.%Class.GetValue.type)]
// CHECK:STDOUT:   %Class.GetValue: @Class.%Class.GetValue.type (%Class.GetValue.type) = struct_value () [symbolic = %Class.GetValue (constants.%Class.GetValue)]
// CHECK:STDOUT:   %T.as_type.loc14_10.2: type = facet_access_type %T.loc5_13.1 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc14_10.2 [symbolic = %require_complete (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc14_10.2 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.as_type.loc14_10.2 (%T.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.GetAddr.decl: @Class.%Class.GetAddr.type (%Class.GetAddr.type) = fn_decl @Class.GetAddr [symbolic = @Class.%Class.GetAddr (constants.%Class.GetAddr)] {
// CHECK:STDOUT:       %self.patt: @Class.GetAddr.%pattern_type.loc6_19 (%pattern_type.417) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetAddr.%pattern_type.loc6_19 (%pattern_type.417) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetAddr.%pattern_type.loc6_34 (%pattern_type.b45) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetAddr.%pattern_type.loc6_34 (%pattern_type.b45) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc6_38.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_38.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_38: type = converted %T.ref, %T.as_type.loc6_38.2 [symbolic = %T.as_type.loc6_38.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %ptr.loc6_38.2: type = ptr_type %.loc6_38 [symbolic = %ptr.loc6_38.1 (constants.%ptr.a75)]
// CHECK:STDOUT:       %self.param: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.0b3) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29: type = splice_block %ptr.loc6_29.2 [symbolic = %ptr.loc6_29.1 (constants.%ptr.0b3)] {
// CHECK:STDOUT:         %.loc6_25: type = specific_constant constants.%Class, @Class(constants.%T.578) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_25 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %ptr.loc6_29.2: type = ptr_type %Self.ref [symbolic = %ptr.loc6_29.1 (constants.%ptr.0b3)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.0b3) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetAddr.%ptr.loc6_38.1 (%ptr.a75) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetAddr.%ptr.loc6_38.1 (%ptr.a75) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetValue.decl: @Class.%Class.GetValue.type (%Class.GetValue.type) = fn_decl @Class.GetValue [symbolic = @Class.%Class.GetValue (constants.%Class.GetValue)] {
// CHECK:STDOUT:       %self.patt: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.47a) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetValue.%pattern_type.loc10_15 (%pattern_type.47a) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetValue.%pattern_type.loc10_29 (%pattern_type.f8cebc.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetValue.%pattern_type.loc10_29 (%pattern_type.f8cebc.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc10_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_32: type = converted %T.ref, %T.as_type.loc10_32.2 [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %self.param: @Class.GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_21.2: type = specific_constant constants.%Class, @Class(constants.%T.578) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetValue.%Class (%Class) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc14_10.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc14_10: type = converted %T.ref, %T.as_type.loc14_10.1 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc14_8: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T.578) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
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
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr.loc6_29.1: type = ptr_type %Class [symbolic = %ptr.loc6_29.1 (constants.%ptr.0b3)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %ptr.loc6_29.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.417)]
// CHECK:STDOUT:   %T.as_type.loc6_38.1: type = facet_access_type %T [symbolic = %T.as_type.loc6_38.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %ptr.loc6_38.1: type = ptr_type %T.as_type.loc6_38.1 [symbolic = %ptr.loc6_38.1 (constants.%ptr.a75)]
// CHECK:STDOUT:   %pattern_type.loc6_34: type = pattern_type %ptr.loc6_38.1 [symbolic = %pattern_type.loc6_34 (constants.%pattern_type.b45)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_34: <witness> = require_complete_type %ptr.loc6_38.1 [symbolic = %require_complete.loc6_34 (constants.%require_complete.999)]
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %ptr.loc6_29.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.27d)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %Class [symbolic = %require_complete.loc7 (constants.%require_complete.ef5)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc6_38.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc6_38.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.23e)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc6_38.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d7e)]
// CHECK:STDOUT:   %.loc7_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc7_12.2 (constants.%.30e)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.GetAddr.%.loc7_12.2 (%.30e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.cb9)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.e97)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.0b3)) -> @Class.GetAddr.%ptr.loc6_38.1 (%ptr.a75) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetAddr.%ptr.loc6_29.1 (%ptr.0b3) = name_ref self, %self
// CHECK:STDOUT:     %.loc7_17.1: ref @Class.GetAddr.%Class (%Class) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @Class.GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc7_17.2: ref @Class.GetAddr.%T.as_type.loc6_38.1 (%T.as_type) = class_element_access %.loc7_17.1, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc6_38.1 (%ptr.a75) = addr_of %.loc7_17.2
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.GetAddr.%.loc7_12.2 (%.30e) = impl_witness_access constants.%Copy.lookup_impl_witness.23e, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.cb9)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %addr, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.Op(constants.%Copy.facet.d7e) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.e97)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %.loc7_12.1: init @Class.GetAddr.%ptr.loc6_38.1 (%ptr.a75) = call %bound_method.loc7_12.2(%addr)
// CHECK:STDOUT:     return %.loc7_12.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetValue(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_15: type = pattern_type %Class [symbolic = %pattern_type.loc10_15 (constants.%pattern_type.47a)]
// CHECK:STDOUT:   %T.as_type.loc10_32.1: type = facet_access_type %T [symbolic = %T.as_type.loc10_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc10_29: type = pattern_type %T.as_type.loc10_32.1 [symbolic = %pattern_type.loc10_29 (constants.%pattern_type.f8cebc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10 (constants.%require_complete.ef5)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc10_32.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %T.as_type.loc10_32.1 [symbolic = %require_complete.loc11 (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3ba)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc10_32.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.72e)]
// CHECK:STDOUT:   %.loc11_16.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc11_16.4 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.GetValue.%.loc11_16.4 (%.671) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetValue.%Class (%Class)) -> @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @Class.GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = bind_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.GetValue.%.loc11_16.4 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc11_16.3: init @Class.GetValue.%T.as_type.loc10_32.1 (%T.as_type) = call %bound_method.loc11_16.2(%.loc11_16.2)
// CHECK:STDOUT:     return %.loc11_16.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.0b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.417)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.0b3)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.578) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.578
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr
// CHECK:STDOUT:   %Class.GetValue.type => constants.%Class.GetValue.type
// CHECK:STDOUT:   %Class.GetValue => constants.%Class.GetValue
// CHECK:STDOUT:   %T.as_type.loc14_10.2 => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ecc
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T.578) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.0b3
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.417
// CHECK:STDOUT:   %T.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc6_38.1 => constants.%ptr.a75
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.b45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetValue(constants.%T.578) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_15 => constants.%pattern_type.47a
// CHECK:STDOUT:   %T.as_type.loc10_32.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc10_29 => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T.578) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T.578) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr => constants.%ptr.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.417
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc17_19.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:

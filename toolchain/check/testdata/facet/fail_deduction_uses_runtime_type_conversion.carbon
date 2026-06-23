// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_deduction_uses_runtime_type_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_deduction_uses_runtime_type_conversion.carbon

// Uses a tuple to allow using the inner type as a type that isn't directly
// deducible, without deducing through HoldsType.
class HoldsType(T:! (type, )) {}

class RuntimeConvertFrom {}
class RuntimeConvertTo {}

impl RuntimeConvertFrom as Core.ImplicitAs(RuntimeConvertTo) {
  // Runtime conversion function
  fn Convert(unused self) -> RuntimeConvertTo { return {}; }
}

fn F[T:! (type, )](unused A:! T.0, unused x: HoldsType(T)) {}

fn G(holds_to: HoldsType((RuntimeConvertTo, )), from:! RuntimeConvertFrom) {
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE+10]]:3: error: compile-time value requires runtime conversion, constructing value of type `RuntimeConvertTo` [RuntimeConversionDuringCompTimeDeduction]
  // CHECK:STDERR:   F(from, holds_to);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! (type, )](unused A:! T.0, unused x: HoldsType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! (type, )](unused A:! T.0, unused x: HoldsType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(from, holds_to);
}


// CHECK:STDOUT: --- fail_deduction_uses_runtime_type_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.0d2: %tuple.type = tuple_value (type) [concrete]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.a83: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom: type = class_type @RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo: type = class_type @RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.286: type = facet_type <@ImplicitAs, @ImplicitAs(%RuntimeConvertTo)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @RuntimeConvertFrom.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.598: type = pattern_type %RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %self.param_patt.c41: %pattern_type.598 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.65f: %pattern_type.598 = at_binding_pattern self, %self.param_patt.c41 [concrete]
// CHECK:STDOUT:   %.aac: Core.Form = init_form %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %return.param_patt.183: %pattern_type.c10 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.b01: %pattern_type.c10 = return_slot_pattern %return.param_patt.183, %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type: type = fn_type @RuntimeConvertFrom.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert: %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.286 = facet_value %RuntimeConvertFrom, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f8a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%RuntimeConvertTo, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo.val: %RuntimeConvertTo = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e66: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %A.patt.aee: %pattern_type.e66 = symbolic_binding_pattern A, 1 [symbolic]
// CHECK:STDOUT:   %A: %tuple.elem0 = symbolic_binding A, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b2f: type = pattern_type %HoldsType.a83 [symbolic]
// CHECK:STDOUT:   %x.param_patt.321: %pattern_type.b2f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.a80: %pattern_type.b2f = at_binding_pattern x, %x.param_patt.321 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.a83 [symbolic]
// CHECK:STDOUT:   %tuple.162: %tuple.type = tuple_value (%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %HoldsType.9f2: type = class_type @HoldsType, @HoldsType(%tuple.162) [concrete]
// CHECK:STDOUT:   %pattern_type.905: type = pattern_type %HoldsType.9f2 [concrete]
// CHECK:STDOUT:   %holds_to.param_patt: %pattern_type.905 = value_param_pattern [concrete]
// CHECK:STDOUT:   %holds_to.patt: %pattern_type.905 = at_binding_pattern holds_to, %holds_to.param_patt [concrete]
// CHECK:STDOUT:   %from.patt: %pattern_type.598 = symbolic_binding_pattern from, 0 [symbolic]
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.820: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f8a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %from, %RuntimeConvertFrom.as.ImplicitAs.impl.Convert [symbolic]
// CHECK:STDOUT:   %A.patt.797: %pattern_type.c10 = symbolic_binding_pattern A, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.fbd: %pattern_type.905 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.32c: %pattern_type.905 = at_binding_pattern x, %x.param_patt.fbd [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc40_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HoldsType = %HoldsType.decl
// CHECK:STDOUT:     .RuntimeConvertFrom = %RuntimeConvertFrom.decl
// CHECK:STDOUT:     .RuntimeConvertTo = %RuntimeConvertTo.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HoldsType.decl: %HoldsType.type = class_decl @HoldsType [concrete = constants.%HoldsType.generic] {
// CHECK:STDOUT:     %T.patt.loc17_18.1: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_28.1: type = splice_block %.loc17_28.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc17_22: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc17_28.2: %tuple.type = tuple_literal (%.loc17_22) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc17_28.3: type = converted %.loc17_28.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_18.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc17_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeConvertFrom.decl: type = class_decl @RuntimeConvertFrom [concrete = constants.%RuntimeConvertFrom] {} {}
// CHECK:STDOUT:   %RuntimeConvertTo.decl: type = class_decl @RuntimeConvertTo [concrete = constants.%RuntimeConvertTo] {} {}
// CHECK:STDOUT:   impl_decl @RuntimeConvertFrom.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%RuntimeConvertTo)> [concrete = constants.%ImplicitAs.type.286]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc27_7.1: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %A.patt.loc27_28.1: @F.%pattern_type.loc27_28 (%pattern_type.e66) = symbolic_binding_pattern A, 1 [symbolic = %A.patt.loc27_28.2 (constants.%A.patt.aee)]
// CHECK:STDOUT:     %x.param_patt.loc27_44.1: @F.%pattern_type.loc27_44 (%pattern_type.b2f) = value_param_pattern [symbolic = %x.param_patt.loc27_44.2 (constants.%x.param_patt.321)]
// CHECK:STDOUT:     %x.patt.loc27_44.1: @F.%pattern_type.loc27_44 (%pattern_type.b2f) = at_binding_pattern x, %x.param_patt.loc27_44.1 [symbolic = %x.patt.loc27_44.2 (constants.%x.patt.a80)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_17.1: type = splice_block %.loc27_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.loc27_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc27_11: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc27_17.2: %tuple.type = tuple_literal (%.loc27_11) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc27_17.3: type = converted %.loc27_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_7.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc27_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc27_32: type = splice_block %tuple.elem0.loc27_32.2 [symbolic = %tuple.elem0.loc27_32.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %.Self.loc27_28: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref.loc27_31: %tuple.type = name_ref T, %T.loc27_7.2 [symbolic = %T.loc27_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc27_32.2: type = tuple_access %T.ref.loc27_31, element0 [symbolic = %tuple.elem0.loc27_32.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc27_28.2: @F.%tuple.elem0.loc27_32.1 (%tuple.elem0) = symbolic_binding A, 1 [symbolic = %A.loc27_28.1 (constants.%A)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc27_57.1 (%HoldsType.a83) = value_param call_param0
// CHECK:STDOUT:     %.loc27_57: type = splice_block %HoldsType.loc27_57.2 [symbolic = %HoldsType.loc27_57.1 (constants.%HoldsType.a83)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc27_56: %tuple.type = name_ref T, %T.loc27_7.2 [symbolic = %T.loc27_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc27_57.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc27_57.1 (constants.%HoldsType.a83)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc27_57.1 (%HoldsType.a83) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %holds_to.param_patt: %pattern_type.905 = value_param_pattern [concrete = constants.%holds_to.param_patt]
// CHECK:STDOUT:     %holds_to.patt: %pattern_type.905 = at_binding_pattern holds_to, %holds_to.param_patt [concrete = constants.%holds_to.patt]
// CHECK:STDOUT:     %from.patt.loc29_53.1: %pattern_type.598 = symbolic_binding_pattern from, 0 [symbolic = %from.patt.loc29_53.2 (constants.%from.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %holds_to.param: %HoldsType.9f2 = value_param call_param0
// CHECK:STDOUT:     %.loc29_46.1: type = splice_block %HoldsType [concrete = constants.%HoldsType.9f2] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:       %.loc29_45: %tuple.type = tuple_literal (%RuntimeConvertTo.ref) [concrete = constants.%tuple.162]
// CHECK:STDOUT:       %tuple.loc29: %tuple.type = tuple_value (%RuntimeConvertTo.ref) [concrete = constants.%tuple.162]
// CHECK:STDOUT:       %.loc29_46.2: %tuple.type = converted %.loc29_45, %tuple.loc29 [concrete = constants.%tuple.162]
// CHECK:STDOUT:       %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.162) [concrete = constants.%HoldsType.9f2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %holds_to: %HoldsType.9f2 = wrapper_binding holds_to, %holds_to.param
// CHECK:STDOUT:     %.loc29_56: type = splice_block %RuntimeConvertFrom.ref [concrete = constants.%RuntimeConvertFrom] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %from.loc29_53.2: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic = %from.loc29_53.1 (constants.%from)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @RuntimeConvertFrom.as.ImplicitAs.impl: %RuntimeConvertFrom.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl: %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type = fn_decl @RuntimeConvertFrom.as.ImplicitAs.impl.Convert [concrete = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.598 = value_param_pattern [concrete = constants.%self.param_patt.c41]
// CHECK:STDOUT:     %self.patt: %pattern_type.598 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.65f]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c10 = out_param_pattern [concrete = constants.%return.param_patt.183]
// CHECK:STDOUT:     %return.patt: %pattern_type.c10 = return_slot_pattern %return.param_patt, %RuntimeConvertTo.ref [concrete = constants.%return.patt.b01]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %.loc24_30: Core.Form = init_form %RuntimeConvertTo.ref [concrete = constants.%.aac]
// CHECK:STDOUT:     %self.param: %RuntimeConvertFrom = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @RuntimeConvertFrom.as.ImplicitAs.impl.%RuntimeConvertFrom.ref [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %self: %RuntimeConvertFrom = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %RuntimeConvertTo = out_param call_param1
// CHECK:STDOUT:     %return: ref %RuntimeConvertTo = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl), @RuntimeConvertFrom.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .RuntimeConvertTo = <poisoned>
// CHECK:STDOUT:   .Convert = %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc17_18.2: %tuple.type) {
// CHECK:STDOUT:   %T.patt.loc17_18.2: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc17_18.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc17_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.a83
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @RuntimeConvertFrom {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%RuntimeConvertFrom
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @RuntimeConvertTo {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%RuntimeConvertTo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeConvertFrom.as.ImplicitAs.impl.Convert(%self.param: %RuntimeConvertFrom) -> out %return.param: %RuntimeConvertTo {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_57.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_57.2: init %RuntimeConvertTo to %return.param = class_init () [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   %.loc24_58: init %RuntimeConvertTo = converted %.loc24_57.1, %.loc24_57.2 [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   return %.loc24_58 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc27_7.2: %tuple.type, %A.loc27_28.2: @F.%tuple.elem0.loc27_32.1 (%tuple.elem0)) {
// CHECK:STDOUT:   %T.patt.loc27_7.2: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc27_7.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc27_7.1 (constants.%T)]
// CHECK:STDOUT:   %tuple.elem0.loc27_32.1: type = tuple_access %T.loc27_7.1, element0 [symbolic = %tuple.elem0.loc27_32.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc27_28: type = pattern_type %tuple.elem0.loc27_32.1 [symbolic = %pattern_type.loc27_28 (constants.%pattern_type.e66)]
// CHECK:STDOUT:   %A.patt.loc27_28.2: @F.%pattern_type.loc27_28 (%pattern_type.e66) = symbolic_binding_pattern A, 1 [symbolic = %A.patt.loc27_28.2 (constants.%A.patt.aee)]
// CHECK:STDOUT:   %A.loc27_28.1: @F.%tuple.elem0.loc27_32.1 (%tuple.elem0) = symbolic_binding A, 1 [symbolic = %A.loc27_28.1 (constants.%A)]
// CHECK:STDOUT:   %HoldsType.loc27_57.1: type = class_type @HoldsType, @HoldsType(%T.loc27_7.1) [symbolic = %HoldsType.loc27_57.1 (constants.%HoldsType.a83)]
// CHECK:STDOUT:   %pattern_type.loc27_44: type = pattern_type %HoldsType.loc27_57.1 [symbolic = %pattern_type.loc27_44 (constants.%pattern_type.b2f)]
// CHECK:STDOUT:   %x.param_patt.loc27_44.2: @F.%pattern_type.loc27_44 (%pattern_type.b2f) = value_param_pattern [symbolic = %x.param_patt.loc27_44.2 (constants.%x.param_patt.321)]
// CHECK:STDOUT:   %x.patt.loc27_44.2: @F.%pattern_type.loc27_44 (%pattern_type.b2f) = at_binding_pattern x, %x.param_patt.loc27_44.2 [symbolic = %x.patt.loc27_44.2 (constants.%x.patt.a80)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc27_57.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc27_57.1 (%HoldsType.a83)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%from.loc29_53.2: %RuntimeConvertFrom) {
// CHECK:STDOUT:   %from.patt.loc29_53.2: %pattern_type.598 = symbolic_binding_pattern from, 0 [symbolic = %from.patt.loc29_53.2 (constants.%from.patt)]
// CHECK:STDOUT:   %from.loc29_53.1: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic = %from.loc29_53.1 (constants.%from)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %from.loc29_53.1, constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert [symbolic = %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound (constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%holds_to.param: %HoldsType.9f2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %from.ref: %RuntimeConvertFrom = name_ref from, %from.loc29_53.2 [symbolic = %from.loc29_53.1 (constants.%from)]
// CHECK:STDOUT:     %holds_to.ref: %HoldsType.9f2 = name_ref holds_to, %holds_to
// CHECK:STDOUT:     %impl.elem0: %.820 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method constants.%from, %impl.elem0 [symbolic = %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound (constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %.loc40_19.1: ref %RuntimeConvertTo = temporary_storage
// CHECK:STDOUT:     %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.call: init %RuntimeConvertTo to %.loc40_19.1 = call %bound_method(constants.%from)
// CHECK:STDOUT:     %.loc40_19.2: init %RuntimeConvertTo = converted constants.%from, %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:     %.loc40_19.3: ref %RuntimeConvertTo = temporary %.loc40_19.1, %.loc40_19.2
// CHECK:STDOUT:     %.loc40_19.4: %RuntimeConvertTo = acquire_value %.loc40_19.3
// CHECK:STDOUT:     %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.162, <error>) [concrete = <error>]
// CHECK:STDOUT:     %.loc40_19.5: %empty_tuple.type = call %F.specific_fn(%holds_to.ref)
// CHECK:STDOUT:     %tuple.loc40: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc40_19.6: %empty_tuple.type = converted %.loc40_19.5, %tuple.loc40 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc40_19.3, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc40_19.3)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc40_19.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc40_19.2(%self.param: ref %RuntimeConvertTo) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc17_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%A) {
// CHECK:STDOUT:   %T.patt.loc27_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc27_7.1 => constants.%T
// CHECK:STDOUT:   %tuple.elem0.loc27_32.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc27_28 => constants.%pattern_type.e66
// CHECK:STDOUT:   %A.patt.loc27_28.2 => constants.%A.patt.aee
// CHECK:STDOUT:   %A.loc27_28.1 => constants.%A
// CHECK:STDOUT:   %HoldsType.loc27_57.1 => constants.%HoldsType.a83
// CHECK:STDOUT:   %pattern_type.loc27_44 => constants.%pattern_type.b2f
// CHECK:STDOUT:   %x.param_patt.loc27_44.2 => constants.%x.param_patt.321
// CHECK:STDOUT:   %x.patt.loc27_44.2 => constants.%x.patt.a80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.162) {
// CHECK:STDOUT:   %T.patt.loc17_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_18.1 => constants.%tuple.162
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%from) {
// CHECK:STDOUT:   %from.patt.loc29_53.2 => constants.%from.patt
// CHECK:STDOUT:   %from.loc29_53.1 => constants.%from
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.162, <error>) {
// CHECK:STDOUT:   %T.patt.loc27_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc27_7.1 => constants.%tuple.162
// CHECK:STDOUT:   %tuple.elem0.loc27_32.1 => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %pattern_type.loc27_28 => constants.%pattern_type.c10
// CHECK:STDOUT:   %A.patt.loc27_28.2 => constants.%A.patt.797
// CHECK:STDOUT:   %A.loc27_28.1 => <error>
// CHECK:STDOUT:   %HoldsType.loc27_57.1 => constants.%HoldsType.9f2
// CHECK:STDOUT:   %pattern_type.loc27_44 => constants.%pattern_type.905
// CHECK:STDOUT:   %x.param_patt.loc27_44.2 => constants.%x.param_patt.fbd
// CHECK:STDOUT:   %x.patt.loc27_44.2 => constants.%x.patt.32c
// CHECK:STDOUT: }
// CHECK:STDOUT:

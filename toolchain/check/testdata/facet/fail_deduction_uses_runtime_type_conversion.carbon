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
class HoldsType(T: (type, )) {}

class RuntimeConvertFrom {}
class RuntimeConvertTo {}

impl RuntimeConvertFrom as Core.ImplicitAs(RuntimeConvertTo) {
  // Runtime conversion function
  fn Convert(unused self) -> RuntimeConvertTo { return {}; }
}

fn F[T: (type, )](unused generic A: T.0, unused x: HoldsType(T)) {}

fn G(holds_to: HoldsType((RuntimeConvertTo, )), generic from: RuntimeConvertFrom) {
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE+4]]:3-19: error: compile-time value requires runtime conversion, constructing value of type `RuntimeConvertTo` [RuntimeConversionDuringCompTimeDeduction]
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE-4]]:1-66: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE-5]]:1-66: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  F(from, holds_to);
}


// CHECK:STDOUT: --- fail_deduction_uses_runtime_type_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type = tuple_value (type) [concrete]
// CHECK:STDOUT:   %pattern_type.ddf: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %T.patt.d26: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.198: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = parameterized_class_name_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.7c6: type = class_type @HoldsType, @HoldsType(%T.198) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom: type = class_type @RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo: type = class_type @RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.47c: type = facet_type <@ImplicitAs, @ImplicitAs(%RuntimeConvertTo)> [concrete]
// CHECK:STDOUT:   %.692: <witness> = impl_self_witness %RuntimeConvertFrom, @ImplicitAs, @ImplicitAs(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @RuntimeConvertFrom.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.598: type = pattern_type %RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %self.param_patt.c41: %pattern_type.598 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.61f: %pattern_type.598 = wrapper_binding_pattern self, %self.param_patt.c41 [concrete]
// CHECK:STDOUT:   %.aac: Core.Form = init_form %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %return.param_patt.183: %pattern_type.c10 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.b01: %pattern_type.c10 = return_slot_pattern %return.param_patt.183, %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type: type = fn_type @RuntimeConvertFrom.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert: %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.47c = facet_value %RuntimeConvertFrom, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e8f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%RuntimeConvertTo, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo.val: %RuntimeConvertTo = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T.198, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.251: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %A.patt.ee4: %pattern_type.251 = symbolic_binding_pattern A, 1 [symbolic]
// CHECK:STDOUT:   %A: %tuple.elem0 = symbolic_binding A, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %HoldsType.7c6 [symbolic]
// CHECK:STDOUT:   %x.param_patt.c68: %pattern_type.831 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.efd: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt.c68 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1ff: <witness> = require_complete_type %HoldsType.7c6 [symbolic]
// CHECK:STDOUT:   %tuple.eb2: %tuple.type = tuple_value (%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %HoldsType.30a: type = class_type @HoldsType, @HoldsType(%tuple.eb2) [concrete]
// CHECK:STDOUT:   %pattern_type.aaa: type = pattern_type %HoldsType.30a [concrete]
// CHECK:STDOUT:   %holds_to.param_patt: %pattern_type.aaa = value_param_pattern [concrete]
// CHECK:STDOUT:   %holds_to.patt: %pattern_type.aaa = wrapper_binding_pattern holds_to, %holds_to.param_patt [concrete]
// CHECK:STDOUT:   %from.patt: %pattern_type.598 = symbolic_binding_pattern from, 0 [symbolic]
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.0aa: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e8f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %from, %RuntimeConvertFrom.as.ImplicitAs.impl.Convert [symbolic]
// CHECK:STDOUT:   %A.patt.797: %pattern_type.c10 = symbolic_binding_pattern A, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.32e: %pattern_type.aaa = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c12: %pattern_type.aaa = wrapper_binding_pattern x, %x.param_patt.32e [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.790: %SubobjectDestroy.type = facet_value %RuntimeConvertTo, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.18d: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.790) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %RuntimeConvertTo, (%DestroyImpl.impl_witness.18d, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc17_18.1: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_18.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_27.1: type = splice_block %.loc17_27.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc17_21: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc17_27.2: %tuple.type = tuple_literal (%.loc17_21) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %.loc17_27.3: type = converted %.loc17_27.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_18.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc17_18.1 (constants.%T.198)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeConvertFrom.decl: type = class_decl @RuntimeConvertFrom [concrete = constants.%RuntimeConvertFrom] {} {}
// CHECK:STDOUT:   %RuntimeConvertTo.decl: type = class_decl @RuntimeConvertTo [concrete = constants.%RuntimeConvertTo] {} {}
// CHECK:STDOUT:   impl_decl @RuntimeConvertFrom.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%RuntimeConvertTo)> [concrete = constants.%ImplicitAs.type.47c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = impl_self_witness @RuntimeConvertFrom.as.ImplicitAs.impl.%RuntimeConvertFrom.ref, @ImplicitAs, @ImplicitAs(constants.%RuntimeConvertTo) [concrete = constants.%.692]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc27_7.1: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_7.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:     %A.patt.loc27_35.1: @F.%pattern_type.loc27_35 (%pattern_type.251) = symbolic_binding_pattern A, 1 [symbolic = %A.patt.loc27_35.2 (constants.%A.patt.ee4)]
// CHECK:STDOUT:     %x.param_patt.loc27_50.1: @F.%pattern_type.loc27_50 (%pattern_type.831) = value_param_pattern [symbolic = %x.param_patt.loc27_50.2 (constants.%x.param_patt.c68)]
// CHECK:STDOUT:     %x.patt.loc27_50.1: @F.%pattern_type.loc27_50 (%pattern_type.831) = wrapper_binding_pattern x, %x.param_patt.loc27_50.1 [symbolic = %x.patt.loc27_50.2 (constants.%x.patt.efd)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_16.1: type = splice_block %.loc27_16.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.frozen.loc27_7: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc27_10: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc27_16.2: %tuple.type = tuple_literal (%.loc27_10) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %.loc27_16.3: type = converted %.loc27_16.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_7.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc27_7.1 (constants.%T.198)]
// CHECK:STDOUT:     %.loc27_38: type = splice_block %tuple.elem0.loc27_38.2 [symbolic = %tuple.elem0.loc27_38.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %.Self.frozen.loc27_35: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %T.ref.loc27_37: %tuple.type = name_ref T, %T.loc27_7.2 [symbolic = %T.loc27_7.1 (constants.%T.198)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc27_38.2: type = tuple_access %T.ref.loc27_37, element0 [symbolic = %tuple.elem0.loc27_38.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc27_35.2: @F.%tuple.elem0.loc27_38.1 (%tuple.elem0) = symbolic_binding A, 1 [symbolic = %A.loc27_35.1 (constants.%A)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc27_63.1 (%HoldsType.7c6) = value_param call_param0
// CHECK:STDOUT:     %.loc27_63: type = splice_block %HoldsType.loc27_63.2 [symbolic = %HoldsType.loc27_63.1 (constants.%HoldsType.7c6)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc27_62: %tuple.type = name_ref T, %T.loc27_7.2 [symbolic = %T.loc27_7.1 (constants.%T.198)]
// CHECK:STDOUT:       %HoldsType.loc27_63.2: type = class_type @HoldsType, @HoldsType(constants.%T.198) [symbolic = %HoldsType.loc27_63.1 (constants.%HoldsType.7c6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc27_63.1 (%HoldsType.7c6) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %holds_to.param_patt: %pattern_type.aaa = value_param_pattern [concrete = constants.%holds_to.param_patt]
// CHECK:STDOUT:     %holds_to.patt: %pattern_type.aaa = wrapper_binding_pattern holds_to, %holds_to.param_patt [concrete = constants.%holds_to.patt]
// CHECK:STDOUT:     %from.patt.loc29_61.1: %pattern_type.598 = symbolic_binding_pattern from, 0 [symbolic = %from.patt.loc29_61.2 (constants.%from.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %holds_to.param: %HoldsType.30a = value_param call_param0
// CHECK:STDOUT:     %.loc29_46.1: type = splice_block %HoldsType [concrete = constants.%HoldsType.30a] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:       %.loc29_45: %tuple.type = tuple_literal (%RuntimeConvertTo.ref) [concrete = constants.%tuple.eb2]
// CHECK:STDOUT:       %tuple: %tuple.type = tuple_value (%RuntimeConvertTo.ref) [concrete = constants.%tuple.eb2]
// CHECK:STDOUT:       %.loc29_46.2: %tuple.type = converted %.loc29_45, %tuple [concrete = constants.%tuple.eb2]
// CHECK:STDOUT:       %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.eb2) [concrete = constants.%HoldsType.30a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %holds_to: %HoldsType.30a = wrapper_binding holds_to, %holds_to.param
// CHECK:STDOUT:     %.loc29_63: type = splice_block %RuntimeConvertFrom.ref [concrete = constants.%RuntimeConvertFrom] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %from.loc29_61.2: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic = %from.loc29_61.1 (constants.%from)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @RuntimeConvertFrom.as.ImplicitAs.impl: %RuntimeConvertFrom.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl: %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type = fn_decl @RuntimeConvertFrom.as.ImplicitAs.impl.Convert [concrete = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.598 = value_param_pattern [concrete = constants.%self.param_patt.c41]
// CHECK:STDOUT:     %self.patt: %pattern_type.598 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.61f]
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
// CHECK:STDOUT:   %T.patt.loc17_18.2: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_18.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:   %T.loc17_18.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc17_18.1 (constants.%T.198)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.7c6
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
// CHECK:STDOUT: generic fn @F(%T.loc27_7.2: %tuple.type, %A.loc27_35.2: @F.%tuple.elem0.loc27_38.1 (%tuple.elem0)) {
// CHECK:STDOUT:   %T.patt.loc27_7.2: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_7.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:   %T.loc27_7.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc27_7.1 (constants.%T.198)]
// CHECK:STDOUT:   %tuple.elem0.loc27_38.1: type = tuple_access %T.loc27_7.1, element0 [symbolic = %tuple.elem0.loc27_38.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc27_35: type = pattern_type %tuple.elem0.loc27_38.1 [symbolic = %pattern_type.loc27_35 (constants.%pattern_type.251)]
// CHECK:STDOUT:   %A.patt.loc27_35.2: @F.%pattern_type.loc27_35 (%pattern_type.251) = symbolic_binding_pattern A, 1 [symbolic = %A.patt.loc27_35.2 (constants.%A.patt.ee4)]
// CHECK:STDOUT:   %A.loc27_35.1: @F.%tuple.elem0.loc27_38.1 (%tuple.elem0) = symbolic_binding A, 1 [symbolic = %A.loc27_35.1 (constants.%A)]
// CHECK:STDOUT:   %HoldsType.loc27_63.1: type = class_type @HoldsType, @HoldsType(%T.loc27_7.1) [symbolic = %HoldsType.loc27_63.1 (constants.%HoldsType.7c6)]
// CHECK:STDOUT:   %pattern_type.loc27_50: type = pattern_type %HoldsType.loc27_63.1 [symbolic = %pattern_type.loc27_50 (constants.%pattern_type.831)]
// CHECK:STDOUT:   %x.param_patt.loc27_50.2: @F.%pattern_type.loc27_50 (%pattern_type.831) = value_param_pattern [symbolic = %x.param_patt.loc27_50.2 (constants.%x.param_patt.c68)]
// CHECK:STDOUT:   %x.patt.loc27_50.2: @F.%pattern_type.loc27_50 (%pattern_type.831) = wrapper_binding_pattern x, %x.param_patt.loc27_50.2 [symbolic = %x.patt.loc27_50.2 (constants.%x.patt.efd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc27_63.1 [symbolic = %require_complete (constants.%require_complete.1ff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc27_63.1 (%HoldsType.7c6)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%from.loc29_61.2: %RuntimeConvertFrom) {
// CHECK:STDOUT:   %from.patt.loc29_61.2: %pattern_type.598 = symbolic_binding_pattern from, 0 [symbolic = %from.patt.loc29_61.2 (constants.%from.patt)]
// CHECK:STDOUT:   %from.loc29_61.1: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic = %from.loc29_61.1 (constants.%from)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %from.loc29_61.1, constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert [symbolic = %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound (constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%holds_to.param: %HoldsType.30a) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %from.ref: %RuntimeConvertFrom = name_ref from, %from.loc29_61.2 [symbolic = %from.loc29_61.1 (constants.%from)]
// CHECK:STDOUT:     %holds_to.ref: %HoldsType.30a = name_ref holds_to, %holds_to
// CHECK:STDOUT:     %impl.elem0: %.0aa = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method constants.%from, %impl.elem0 [symbolic = %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound (constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:     %.loc34_19.1: ref %RuntimeConvertTo = temporary_storage
// CHECK:STDOUT:     %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.call: init %RuntimeConvertTo to %.loc34_19.1 = call %bound_method(constants.%from)
// CHECK:STDOUT:     %.loc34_19.2: init %RuntimeConvertTo = converted constants.%from, %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:     %.loc34_19.3: ref %RuntimeConvertTo = temporary %.loc34_19.1, %.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.4: %RuntimeConvertTo = acquire_value %.loc34_19.3
// CHECK:STDOUT:     %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.eb2, <error>) [concrete = <error>]
// CHECK:STDOUT:     %.loc34_19.5: %empty_tuple.type = call %F.specific_fn(%holds_to.ref)
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc34_19.6: %empty_tuple.type = converted %.loc34_19.5, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %facet_value.loc34_19.1: %facet_type = facet_value constants.%RuntimeConvertTo, (constants.%DestroyImpl.impl_witness.18d, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc34_19.7: %facet_type = converted constants.%RuntimeConvertTo, %facet_value.loc34_19.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:     %facet_value.loc34_19.2: %facet_type = facet_value constants.%RuntimeConvertTo, (constants.%DestroyImpl.impl_witness.18d, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc34_19.8: %facet_type = converted constants.%RuntimeConvertTo, %facet_value.loc34_19.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:     %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc34_19.3)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.2(%self.param: ref %RuntimeConvertTo) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.198) {
// CHECK:STDOUT:   %T.patt.loc17_18.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc17_18.1 => constants.%T.198
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.198, constants.%A) {
// CHECK:STDOUT:   %T.patt.loc27_7.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc27_7.1 => constants.%T.198
// CHECK:STDOUT:   %tuple.elem0.loc27_38.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc27_35 => constants.%pattern_type.251
// CHECK:STDOUT:   %A.patt.loc27_35.2 => constants.%A.patt.ee4
// CHECK:STDOUT:   %A.loc27_35.1 => constants.%A
// CHECK:STDOUT:   %HoldsType.loc27_63.1 => constants.%HoldsType.7c6
// CHECK:STDOUT:   %pattern_type.loc27_50 => constants.%pattern_type.831
// CHECK:STDOUT:   %x.param_patt.loc27_50.2 => constants.%x.param_patt.c68
// CHECK:STDOUT:   %x.patt.loc27_50.2 => constants.%x.patt.efd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.eb2) {
// CHECK:STDOUT:   %T.patt.loc17_18.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc17_18.1 => constants.%tuple.eb2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%from) {
// CHECK:STDOUT:   %from.patt.loc29_61.2 => constants.%from.patt
// CHECK:STDOUT:   %from.loc29_61.1 => constants.%from
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.eb2, <error>) {
// CHECK:STDOUT:   %T.patt.loc27_7.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc27_7.1 => constants.%tuple.eb2
// CHECK:STDOUT:   %tuple.elem0.loc27_38.1 => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %pattern_type.loc27_35 => constants.%pattern_type.c10
// CHECK:STDOUT:   %A.patt.loc27_35.2 => constants.%A.patt.797
// CHECK:STDOUT:   %A.loc27_35.1 => <error>
// CHECK:STDOUT:   %HoldsType.loc27_63.1 => constants.%HoldsType.30a
// CHECK:STDOUT:   %pattern_type.loc27_50 => constants.%pattern_type.aaa
// CHECK:STDOUT:   %x.param_patt.loc27_50.2 => constants.%x.param_patt.32e
// CHECK:STDOUT:   %x.patt.loc27_50.2 => constants.%x.patt.c12
// CHECK:STDOUT: }
// CHECK:STDOUT:

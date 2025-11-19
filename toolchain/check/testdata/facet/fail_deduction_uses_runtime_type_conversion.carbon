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
  fn Convert[self: Self]() -> RuntimeConvertTo { return {}; }
}

fn F[T:! (type, )](A:! T.0, x: HoldsType(T)) {}

fn G(holds_to: HoldsType((RuntimeConvertTo, ))) {
  let from:! RuntimeConvertFrom = {} as RuntimeConvertFrom;
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE+10]]:3: error: compile-time value requires runtime conversion, constructing value of type `RuntimeConvertTo` [RuntimeConversionDuringCompTimeDeduction]
  // CHECK:STDERR:   F(from, holds_to);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! (type, )](A:! T.0, x: HoldsType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduction_uses_runtime_type_conversion.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! (type, )](A:! T.0, x: HoldsType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %T: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.239: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom: type = class_type @RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo: type = class_type @RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b66: type = facet_type <@ImplicitAs, @ImplicitAs(%RuntimeConvertTo)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.50a: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f64: type = pattern_type %RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %pattern_type.109: type = pattern_type %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type: type = fn_type @RuntimeConvertFrom.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert: %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b66 = facet_value %RuntimeConvertFrom, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo.val: %RuntimeConvertTo = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %A: %tuple.elem0 = symbolic_binding A, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c8: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %pattern_type.af4: type = pattern_type %HoldsType.239 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b27: <witness> = require_complete_type %HoldsType.239 [symbolic]
// CHECK:STDOUT:   %tuple.c68: %tuple.type = tuple_value (%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %HoldsType.066: type = class_type @HoldsType, @HoldsType(%tuple.c68) [concrete]
// CHECK:STDOUT:   %pattern_type.a13: type = pattern_type %HoldsType.066 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = symbolic_binding from, 0 [symbolic]
// CHECK:STDOUT:   %RuntimeConvertFrom.val: %RuntimeConvertFrom = struct_value () [concrete]
// CHECK:STDOUT:   %.838: type = fn_type_with_self_type %ImplicitAs.Convert.type.50a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %from, %RuntimeConvertFrom.as.ImplicitAs.impl.Convert [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.d7d: %type_where = facet_value %RuntimeConvertTo, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.96b: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d7d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.310: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.96b = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.748: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.310, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.d7d) [concrete]
// CHECK:STDOUT:   %facet_value.631: %type_where = facet_value %RuntimeConvertFrom, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0dd: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.631) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a5f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0dd = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.f4e: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.a5f, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.631) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_28.1: type = splice_block %.loc17_28.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc17_28.2: %tuple.type = tuple_literal (type) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc17_28.3: type = converted %.loc17_28.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_17.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc17_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeConvertFrom.decl: type = class_decl @RuntimeConvertFrom [concrete = constants.%RuntimeConvertFrom] {} {}
// CHECK:STDOUT:   %RuntimeConvertTo.decl: type = class_decl @RuntimeConvertTo [concrete = constants.%RuntimeConvertTo] {} {}
// CHECK:STDOUT:   impl_decl @RuntimeConvertFrom.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%RuntimeConvertTo)> [concrete = constants.%ImplicitAs.type.b66]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (@RuntimeConvertFrom.as.ImplicitAs.impl.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl), @RuntimeConvertFrom.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %A.patt: @F.%pattern_type.loc27_20 (%pattern_type.2c8) = symbolic_binding_pattern A, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc27_29 (%pattern_type.af4) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc27_29 (%pattern_type.af4) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_17.1: type = splice_block %.loc27_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc27_17.2: %tuple.type = tuple_literal (type) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc27_17.3: type = converted %.loc27_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_6.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc27_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc27_25: type = splice_block %tuple.elem0.loc27_25.2 [symbolic = %tuple.elem0.loc27_25.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref.loc27_24: %tuple.type = name_ref T, %T.loc27_6.2 [symbolic = %T.loc27_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc27_25.2: type = tuple_access %T.ref.loc27_24, element0 [symbolic = %tuple.elem0.loc27_25.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc27_20.2: @F.%tuple.elem0.loc27_25.1 (%tuple.elem0) = symbolic_binding A, 1 [symbolic = %A.loc27_20.1 (constants.%A)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc27_43.1 (%HoldsType.239) = value_param call_param0
// CHECK:STDOUT:     %.loc27_43: type = splice_block %HoldsType.loc27_43.2 [symbolic = %HoldsType.loc27_43.1 (constants.%HoldsType.239)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc27_42: %tuple.type = name_ref T, %T.loc27_6.2 [symbolic = %T.loc27_6.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc27_43.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc27_43.1 (constants.%HoldsType.239)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc27_43.1 (%HoldsType.239) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %holds_to.patt: %pattern_type.a13 = value_binding_pattern holds_to [concrete]
// CHECK:STDOUT:     %holds_to.param_patt: %pattern_type.a13 = value_param_pattern %holds_to.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %holds_to.param: %HoldsType.066 = value_param call_param0
// CHECK:STDOUT:     %.loc29_46.1: type = splice_block %HoldsType [concrete = constants.%HoldsType.066] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:       %.loc29_45: %tuple.type = tuple_literal (%RuntimeConvertTo.ref) [concrete = constants.%tuple.c68]
// CHECK:STDOUT:       %tuple: %tuple.type = tuple_value (%RuntimeConvertTo.ref) [concrete = constants.%tuple.c68]
// CHECK:STDOUT:       %.loc29_46.2: %tuple.type = converted %.loc29_45, %tuple [concrete = constants.%tuple.c68]
// CHECK:STDOUT:       %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.c68) [concrete = constants.%HoldsType.066]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %holds_to: %HoldsType.066 = value_binding holds_to, %holds_to.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @RuntimeConvertFrom.as.ImplicitAs.impl: %RuntimeConvertFrom.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl: %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.type = fn_decl @RuntimeConvertFrom.as.ImplicitAs.impl.Convert [concrete = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f64 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f64 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.109 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.109 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %self.param: %RuntimeConvertFrom = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @RuntimeConvertFrom.as.ImplicitAs.impl.%RuntimeConvertFrom.ref [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %self: %RuntimeConvertFrom = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %RuntimeConvertTo = out_param call_param1
// CHECK:STDOUT:     %return: ref %RuntimeConvertTo = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .RuntimeConvertTo = <poisoned>
// CHECK:STDOUT:   .Convert = %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc17_17.2: %tuple.type) {
// CHECK:STDOUT:   %T.loc17_17.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc17_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.239
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
// CHECK:STDOUT: fn @RuntimeConvertFrom.as.ImplicitAs.impl.Convert(%self.param: %RuntimeConvertFrom) -> %return.param: %RuntimeConvertTo {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_58.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_58.2: init %RuntimeConvertTo = class_init (), %return [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   %.loc24_59: init %RuntimeConvertTo = converted %.loc24_58.1, %.loc24_58.2 [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   return %.loc24_59 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc27_6.2: %tuple.type, %A.loc27_20.2: @F.%tuple.elem0.loc27_25.1 (%tuple.elem0)) {
// CHECK:STDOUT:   %T.loc27_6.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc27_6.1 (constants.%T)]
// CHECK:STDOUT:   %tuple.elem0.loc27_25.1: type = tuple_access %T.loc27_6.1, element0 [symbolic = %tuple.elem0.loc27_25.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %A.loc27_20.1: @F.%tuple.elem0.loc27_25.1 (%tuple.elem0) = symbolic_binding A, 1 [symbolic = %A.loc27_20.1 (constants.%A)]
// CHECK:STDOUT:   %pattern_type.loc27_20: type = pattern_type %tuple.elem0.loc27_25.1 [symbolic = %pattern_type.loc27_20 (constants.%pattern_type.2c8)]
// CHECK:STDOUT:   %HoldsType.loc27_43.1: type = class_type @HoldsType, @HoldsType(%T.loc27_6.1) [symbolic = %HoldsType.loc27_43.1 (constants.%HoldsType.239)]
// CHECK:STDOUT:   %pattern_type.loc27_29: type = pattern_type %HoldsType.loc27_43.1 [symbolic = %pattern_type.loc27_29 (constants.%pattern_type.af4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc27_43.1 [symbolic = %require_complete (constants.%require_complete.b27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc27_43.1 (%HoldsType.239)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%holds_to.param: %HoldsType.066) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %from.patt: %pattern_type.f64 = symbolic_binding_pattern from, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30_36.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %RuntimeConvertFrom.ref.loc30_41: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:   %.loc30_36.2: ref %RuntimeConvertFrom = temporary_storage
// CHECK:STDOUT:   %.loc30_36.3: init %RuntimeConvertFrom = class_init (), %.loc30_36.2 [concrete = constants.%RuntimeConvertFrom.val]
// CHECK:STDOUT:   %.loc30_36.4: ref %RuntimeConvertFrom = temporary %.loc30_36.2, %.loc30_36.3
// CHECK:STDOUT:   %.loc30_38.1: ref %RuntimeConvertFrom = converted %.loc30_36.1, %.loc30_36.4
// CHECK:STDOUT:   %.loc30_14: type = splice_block %RuntimeConvertFrom.ref.loc30_14 [concrete = constants.%RuntimeConvertFrom] {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %RuntimeConvertFrom.ref.loc30_14: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30_38.2: %RuntimeConvertFrom = acquire_value %.loc30_38.1
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = symbolic_binding from, 0, %.loc30_38.2 [symbolic = constants.%from]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %from.ref: %RuntimeConvertFrom = name_ref from, %from [symbolic = constants.%from]
// CHECK:STDOUT:   %holds_to.ref: %HoldsType.066 = name_ref holds_to, %holds_to
// CHECK:STDOUT:   %impl.elem0: %.838 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc41_19.1: <bound method> = bound_method constants.%from, %impl.elem0 [symbolic = constants.%RuntimeConvertFrom.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %.loc41_19.1: ref %RuntimeConvertTo = temporary_storage
// CHECK:STDOUT:   %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.call: init %RuntimeConvertTo = call %bound_method.loc41_19.1(constants.%from) to %.loc41_19.1
// CHECK:STDOUT:   %.loc41_19.2: init %RuntimeConvertTo = converted constants.%from, %RuntimeConvertFrom.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc41_19.3: ref %RuntimeConvertTo = temporary %.loc41_19.1, %.loc41_19.2
// CHECK:STDOUT:   %.loc41_19.4: %RuntimeConvertTo = acquire_value %.loc41_19.3
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.c68, <error>) [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%holds_to.ref)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc41: <bound method> = bound_method %.loc41_19.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.310
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.310, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.d7d) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.748]
// CHECK:STDOUT:   %bound_method.loc41_19.2: <bound method> = bound_method %.loc41_19.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc41: init %empty_tuple.type = call %bound_method.loc41_19.2(%.loc41_19.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc30: <bound method> = bound_method %.loc30_36.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a5f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a5f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.631) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.f4e]
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %.loc30_36.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc30: init %empty_tuple.type = call %bound_method.loc30(%.loc30_36.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.loc17_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%A) {
// CHECK:STDOUT:   %T.loc27_6.1 => constants.%T
// CHECK:STDOUT:   %tuple.elem0.loc27_25.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %A.loc27_20.1 => constants.%A
// CHECK:STDOUT:   %pattern_type.loc27_20 => constants.%pattern_type.2c8
// CHECK:STDOUT:   %HoldsType.loc27_43.1 => constants.%HoldsType.239
// CHECK:STDOUT:   %pattern_type.loc27_29 => constants.%pattern_type.af4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.c68) {
// CHECK:STDOUT:   %T.loc17_17.1 => constants.%tuple.c68
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.c68, <error>) {
// CHECK:STDOUT:   %T.loc27_6.1 => constants.%tuple.c68
// CHECK:STDOUT:   %tuple.elem0.loc27_25.1 => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %A.loc27_20.1 => <error>
// CHECK:STDOUT:   %pattern_type.loc27_20 => constants.%pattern_type.109
// CHECK:STDOUT:   %HoldsType.loc27_43.1 => constants.%HoldsType.066
// CHECK:STDOUT:   %pattern_type.loc27_29 => constants.%pattern_type.a13
// CHECK:STDOUT: }
// CHECK:STDOUT:

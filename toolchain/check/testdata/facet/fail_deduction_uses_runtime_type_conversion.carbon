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
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %T.6eb: %tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.cc9: type = class_type @HoldsType, @HoldsType(%T.6eb) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom: type = class_type @RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo: type = class_type @RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.580: type = facet_type <@ImplicitAs, @ImplicitAs(%RuntimeConvertTo)> [concrete]
// CHECK:STDOUT:   %Convert.type.50a: type = fn_type @Convert.1, @ImplicitAs(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f64: type = pattern_type %RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %pattern_type.109: type = pattern_type %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %Convert.type.e8b: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.e81: %Convert.type.e8b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.580 = facet_value %RuntimeConvertFrom, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo.val: %RuntimeConvertTo = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T.6eb, element0 [symbolic]
// CHECK:STDOUT:   %A: %tuple.elem0 = bind_symbolic_name A, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.08e: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ec6: type = pattern_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.514: <witness> = require_complete_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %HoldsType.066: type = class_type @HoldsType, @HoldsType(%tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.a13: type = pattern_type %HoldsType.066 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = bind_symbolic_name from, 0 [symbolic]
// CHECK:STDOUT:   %RuntimeConvertFrom.val: %RuntimeConvertFrom = struct_value () [concrete]
// CHECK:STDOUT:   %.f4e: type = fn_type_with_self_type %Convert.type.50a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %from, %Convert.e81 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.3c9: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %Op.type.139: type = fn_type @Op.2, @impl.49c(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %Op.d2d: %Op.type.139 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.339: type = ptr_type %RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %Destroy.facet.31d: %Destroy.type = facet_value %RuntimeConvertTo, (%Destroy.impl_witness.3c9) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.31d [concrete]
// CHECK:STDOUT:   %Op.specific_fn.373: <specific function> = specific_function %Op.d2d, @Op.2(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.fab: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%RuntimeConvertFrom) [concrete]
// CHECK:STDOUT:   %Op.type.147: type = fn_type @Op.2, @impl.49c(%RuntimeConvertFrom) [concrete]
// CHECK:STDOUT:   %Op.3f3: %Op.type.147 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.415: type = ptr_type %RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %Destroy.facet.71e: %Destroy.type = facet_value %RuntimeConvertFrom, (%Destroy.impl_witness.fab) [concrete]
// CHECK:STDOUT:   %.a98: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.71e [concrete]
// CHECK:STDOUT:   %Op.specific_fn.5db: <specific function> = specific_function %Op.3f3, @Op.2(%RuntimeConvertFrom) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:       %.loc17_28.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc17_28.3: type = converted %.loc17_28.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_17.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_17.2 (constants.%T.6eb)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeConvertFrom.decl: type = class_decl @RuntimeConvertFrom [concrete = constants.%RuntimeConvertFrom] {} {}
// CHECK:STDOUT:   %RuntimeConvertTo.decl: type = class_decl @RuntimeConvertTo [concrete = constants.%RuntimeConvertTo] {} {}
// CHECK:STDOUT:   impl_decl @impl.423 [concrete] {} {
// CHECK:STDOUT:     %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%RuntimeConvertTo)> [concrete = constants.%ImplicitAs.type.580]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (@impl.423.%Convert.decl), @impl.423 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %A.patt: @F.%pattern_type.loc27_20 (%pattern_type.08e) = symbolic_binding_pattern A, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc27_29 (%pattern_type.ec6) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc27_29 (%pattern_type.ec6) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_17.1: type = splice_block %.loc27_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.loc27_17.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc27_17.3: type = converted %.loc27_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_6.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc27_6.2 (constants.%T.6eb)]
// CHECK:STDOUT:     %.loc27_25: type = splice_block %tuple.elem0.loc27_25.1 [symbolic = %tuple.elem0.loc27_25.2 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc27_24: %tuple.type = name_ref T, %T.loc27_6.1 [symbolic = %T.loc27_6.2 (constants.%T.6eb)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc27_25.1: type = tuple_access %T.ref.loc27_24, element0 [symbolic = %tuple.elem0.loc27_25.2 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc27_20.1: @F.%tuple.elem0.loc27_25.2 (%tuple.elem0) = bind_symbolic_name A, 1 [symbolic = %A.loc27_20.2 (constants.%A)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc27_43.2 (%HoldsType.cc9) = value_param call_param0
// CHECK:STDOUT:     %.loc27_43: type = splice_block %HoldsType.loc27_43.1 [symbolic = %HoldsType.loc27_43.2 (constants.%HoldsType.cc9)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc27_42: %tuple.type = name_ref T, %T.loc27_6.1 [symbolic = %T.loc27_6.2 (constants.%T.6eb)]
// CHECK:STDOUT:       %HoldsType.loc27_43.1: type = class_type @HoldsType, @HoldsType(constants.%T.6eb) [symbolic = %HoldsType.loc27_43.2 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc27_43.2 (%HoldsType.cc9) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %holds_to.patt: %pattern_type.a13 = binding_pattern holds_to [concrete]
// CHECK:STDOUT:     %holds_to.param_patt: %pattern_type.a13 = value_param_pattern %holds_to.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %holds_to.param: %HoldsType.066 = value_param call_param0
// CHECK:STDOUT:     %.loc29_46.1: type = splice_block %HoldsType [concrete = constants.%HoldsType.066] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:       %.loc29_45: %tuple.type = tuple_literal (%RuntimeConvertTo.ref)
// CHECK:STDOUT:       %tuple: %tuple.type = tuple_value (%RuntimeConvertTo.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:       %.loc29_46.2: %tuple.type = converted %.loc29_45, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:       %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple) [concrete = constants.%HoldsType.066]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %holds_to: %HoldsType.066 = bind_name holds_to, %holds_to.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.423: %RuntimeConvertFrom.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e8b = fn_decl @Convert.2 [concrete = constants.%Convert.e81] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f64 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f64 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.109 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.109 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %self.param: %RuntimeConvertFrom = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.423.%RuntimeConvertFrom.ref [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %self: %RuntimeConvertFrom = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %RuntimeConvertTo = out_param call_param1
// CHECK:STDOUT:     %return: ref %RuntimeConvertTo = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .RuntimeConvertTo = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc17_17.1: %tuple.type) {
// CHECK:STDOUT:   %T.loc17_17.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_17.2 (constants.%T.6eb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.cc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @RuntimeConvertFrom {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%RuntimeConvertFrom
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @RuntimeConvertTo {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%RuntimeConvertTo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2(%self.param: %RuntimeConvertFrom) -> %return.param: %RuntimeConvertTo {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_58.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_58.2: init %RuntimeConvertTo = class_init (), %return [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   %.loc24_59: init %RuntimeConvertTo = converted %.loc24_58.1, %.loc24_58.2 [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   return %.loc24_59 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc27_6.1: %tuple.type, %A.loc27_20.1: @F.%tuple.elem0.loc27_25.2 (%tuple.elem0)) {
// CHECK:STDOUT:   %T.loc27_6.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc27_6.2 (constants.%T.6eb)]
// CHECK:STDOUT:   %tuple.elem0.loc27_25.2: type = tuple_access %T.loc27_6.2, element0 [symbolic = %tuple.elem0.loc27_25.2 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %A.loc27_20.2: @F.%tuple.elem0.loc27_25.2 (%tuple.elem0) = bind_symbolic_name A, 1 [symbolic = %A.loc27_20.2 (constants.%A)]
// CHECK:STDOUT:   %pattern_type.loc27_20: type = pattern_type %tuple.elem0.loc27_25.2 [symbolic = %pattern_type.loc27_20 (constants.%pattern_type.08e)]
// CHECK:STDOUT:   %HoldsType.loc27_43.2: type = class_type @HoldsType, @HoldsType(%T.loc27_6.2) [symbolic = %HoldsType.loc27_43.2 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:   %pattern_type.loc27_29: type = pattern_type %HoldsType.loc27_43.2 [symbolic = %pattern_type.loc27_29 (constants.%pattern_type.ec6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc27_43.2 [symbolic = %require_complete (constants.%require_complete.514)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc27_43.2 (%HoldsType.cc9)) {
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
// CHECK:STDOUT:   %.loc30_36.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %RuntimeConvertFrom.ref.loc30_41: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:   %.loc30_36.2: ref %RuntimeConvertFrom = temporary_storage
// CHECK:STDOUT:   %.loc30_36.3: init %RuntimeConvertFrom = class_init (), %.loc30_36.2 [concrete = constants.%RuntimeConvertFrom.val]
// CHECK:STDOUT:   %.loc30_36.4: ref %RuntimeConvertFrom = temporary %.loc30_36.2, %.loc30_36.3
// CHECK:STDOUT:   %.loc30_38.1: ref %RuntimeConvertFrom = converted %.loc30_36.1, %.loc30_36.4
// CHECK:STDOUT:   %RuntimeConvertFrom.ref.loc30_14: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:   %.loc30_38.2: %RuntimeConvertFrom = bind_value %.loc30_38.1
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = bind_symbolic_name from, 0, %.loc30_38.2 [symbolic = constants.%from]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %from.ref: %RuntimeConvertFrom = name_ref from, %from [symbolic = constants.%from]
// CHECK:STDOUT:   %holds_to.ref: %HoldsType.066 = name_ref holds_to, %holds_to
// CHECK:STDOUT:   %impl.elem0.loc41_19.1: %.f4e = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%Convert.e81]
// CHECK:STDOUT:   %bound_method.loc41_19.1: <bound method> = bound_method constants.%from, %impl.elem0.loc41_19.1 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %.loc41_19.1: ref %RuntimeConvertTo = temporary_storage
// CHECK:STDOUT:   %Convert.call: init %RuntimeConvertTo = call %bound_method.loc41_19.1(constants.%from) to %.loc41_19.1
// CHECK:STDOUT:   %.loc41_19.2: init %RuntimeConvertTo = converted constants.%from, %Convert.call
// CHECK:STDOUT:   %.loc41_19.3: ref %RuntimeConvertTo = temporary %.loc41_19.1, %.loc41_19.2
// CHECK:STDOUT:   %.loc41_19.4: %RuntimeConvertTo = bind_value %.loc41_19.3
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple, <error>) [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%holds_to.ref) [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc41_19.2: %.79d = impl_witness_access constants.%Destroy.impl_witness.3c9, element0 [concrete = constants.%Op.d2d]
// CHECK:STDOUT:   %bound_method.loc41_19.2: <bound method> = bound_method %.loc41_19.1, %impl.elem0.loc41_19.2
// CHECK:STDOUT:   %specific_fn.loc41: <specific function> = specific_function %impl.elem0.loc41_19.2, @Op.2(constants.%RuntimeConvertTo) [concrete = constants.%Op.specific_fn.373]
// CHECK:STDOUT:   %bound_method.loc41_19.3: <bound method> = bound_method %.loc41_19.1, %specific_fn.loc41
// CHECK:STDOUT:   %addr.loc41: %ptr.339 = addr_of %.loc41_19.1
// CHECK:STDOUT:   %no_op.loc41: init %empty_tuple.type = call %bound_method.loc41_19.3(%addr.loc41)
// CHECK:STDOUT:   %impl.elem0.loc30: %.a98 = impl_witness_access constants.%Destroy.impl_witness.fab, element0 [concrete = constants.%Op.3f3]
// CHECK:STDOUT:   %bound_method.loc30_36.1: <bound method> = bound_method %.loc30_36.2, %impl.elem0.loc30
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Op.2(constants.%RuntimeConvertFrom) [concrete = constants.%Op.specific_fn.5db]
// CHECK:STDOUT:   %bound_method.loc30_36.2: <bound method> = bound_method %.loc30_36.2, %specific_fn.loc30
// CHECK:STDOUT:   %addr.loc30: %ptr.415 = addr_of %.loc30_36.2
// CHECK:STDOUT:   %no_op.loc30: init %empty_tuple.type = call %bound_method.loc30_36.2(%addr.loc30)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.6eb) {
// CHECK:STDOUT:   %T.loc17_17.2 => constants.%T.6eb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.6eb, constants.%A) {
// CHECK:STDOUT:   %T.loc27_6.2 => constants.%T.6eb
// CHECK:STDOUT:   %tuple.elem0.loc27_25.2 => constants.%tuple.elem0
// CHECK:STDOUT:   %A.loc27_20.2 => constants.%A
// CHECK:STDOUT:   %pattern_type.loc27_20 => constants.%pattern_type.08e
// CHECK:STDOUT:   %HoldsType.loc27_43.2 => constants.%HoldsType.cc9
// CHECK:STDOUT:   %pattern_type.loc27_29 => constants.%pattern_type.ec6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple) {
// CHECK:STDOUT:   %T.loc17_17.2 => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple, <error>) {
// CHECK:STDOUT:   %T.loc27_6.2 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0.loc27_25.2 => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %A.loc27_20.2 => <error>
// CHECK:STDOUT:   %pattern_type.loc27_20 => constants.%pattern_type.109
// CHECK:STDOUT:   %HoldsType.loc27_43.2 => constants.%HoldsType.066
// CHECK:STDOUT:   %pattern_type.loc27_29 => constants.%pattern_type.a13
// CHECK:STDOUT: }
// CHECK:STDOUT:

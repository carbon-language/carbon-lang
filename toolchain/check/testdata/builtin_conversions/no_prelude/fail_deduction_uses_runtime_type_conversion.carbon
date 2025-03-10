// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/fail_deduction_uses_runtime_type_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/fail_deduction_uses_runtime_type_conversion.carbon

// --- core.carbon

package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- fail_deduction_uses_runtime_type_conversion.carbon

import Core;

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

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc4_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc4_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc4_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc4_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc4_22.1: type) {
// CHECK:STDOUT:   %Dest.loc4_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc4_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc4_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc4_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc4_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest.loc4_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc4_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_20.1: type = splice_block %.loc5_20.3 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_20.2: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc5_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_20.3: type = converted %Self.ref, %Self.as_type.loc5_20.2 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc5_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%Dest.loc4_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc4_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc4_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc4_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduction_uses_runtime_type_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %T: %tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %tuple.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.cc9: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %RuntimeConvertFrom: type = class_type @RuntimeConvertFrom [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo: type = class_type @RuntimeConvertTo [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.580: type = facet_type <@ImplicitAs, @ImplicitAs(%RuntimeConvertTo)> [concrete]
// CHECK:STDOUT:   %Convert.type.50a: type = fn_type @Convert.1, @ImplicitAs(%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %Convert.993: %Convert.type.50a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.46b: type = assoc_entity_type %ImplicitAs.type.580 [concrete]
// CHECK:STDOUT:   %assoc0.322: %ImplicitAs.assoc_type.46b = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.e8b: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.e81: %Convert.type.e8b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.580 = facet_value %RuntimeConvertFrom, (%impl_witness) [concrete]
// CHECK:STDOUT:   %RuntimeConvertTo.val: %RuntimeConvertTo = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %A: %tuple.elem0 = bind_symbolic_name A, 1 [symbolic]
// CHECK:STDOUT:   %A.patt: %tuple.elem0 = symbolic_binding_pattern A, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%RuntimeConvertTo) [concrete]
// CHECK:STDOUT:   %HoldsType.066: type = class_type @HoldsType, @HoldsType(%tuple) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = bind_symbolic_name from, 0 [symbolic]
// CHECK:STDOUT:   %from.patt: %RuntimeConvertFrom = symbolic_binding_pattern from, 0 [symbolic]
// CHECK:STDOUT:   %RuntimeConvertFrom.val: %RuntimeConvertFrom = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %.31e: type = fn_type_with_self_type %Convert.type.50a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %from, %Convert.e81 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc5_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst24 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc5_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
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
// CHECK:STDOUT:     %T.patt.loc6_17.1: %tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.loc6_28.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc6_28.3: type = converted %.loc6_28.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_17.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeConvertFrom.decl: type = class_decl @RuntimeConvertFrom [concrete = constants.%RuntimeConvertFrom] {} {}
// CHECK:STDOUT:   %RuntimeConvertTo.decl: type = class_decl @RuntimeConvertTo [concrete = constants.%RuntimeConvertTo] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %RuntimeConvertFrom.ref: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%RuntimeConvertTo)> [concrete = constants.%ImplicitAs.type.580]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc16_6.1: %tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %A.patt.loc16_20.1: @F.%tuple.elem0.loc16_25.2 (%tuple.elem0) = symbolic_binding_pattern A, 1 [symbolic = %A.patt.loc16_20.2 (constants.%A.patt)]
// CHECK:STDOUT:     %x.patt: @F.%HoldsType.loc16_43.2 (%HoldsType.cc9) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%HoldsType.loc16_43.2 (%HoldsType.cc9) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_17.1: type = splice_block %.loc16_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.loc16_17.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc16_17.3: type = converted %.loc16_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_6.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc16_25: type = splice_block %tuple.elem0.loc16_25.1 [symbolic = %tuple.elem0.loc16_25.2 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc16_24: %tuple.type = name_ref T, %T.loc16_6.1 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc16_25.1: type = tuple_access %T.ref.loc16_24, element0 [symbolic = %tuple.elem0.loc16_25.2 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc16_20.1: @F.%tuple.elem0.loc16_25.2 (%tuple.elem0) = bind_symbolic_name A, 1 [symbolic = %A.loc16_20.2 (constants.%A)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc16_43.2 (%HoldsType.cc9) = value_param call_param0
// CHECK:STDOUT:     %.loc16_43: type = splice_block %HoldsType.loc16_43.1 [symbolic = %HoldsType.loc16_43.2 (constants.%HoldsType.cc9)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc16_42: %tuple.type = name_ref T, %T.loc16_6.1 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc16_43.1: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc16_43.2 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc16_43.2 (%HoldsType.cc9) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %holds_to.patt: %HoldsType.066 = binding_pattern holds_to
// CHECK:STDOUT:     %holds_to.param_patt: %HoldsType.066 = value_param_pattern %holds_to.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %holds_to.param: %HoldsType.066 = value_param call_param0
// CHECK:STDOUT:     %.loc18_46.1: type = splice_block %HoldsType [concrete = constants.%HoldsType.066] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:       %.loc18_45: %tuple.type = tuple_literal (%RuntimeConvertTo.ref)
// CHECK:STDOUT:       %tuple: %tuple.type = tuple_value (%RuntimeConvertTo.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:       %.loc18_46.2: %tuple.type = converted %.loc18_45, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:       %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple) [concrete = constants.%HoldsType.066]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %holds_to: %HoldsType.066 = bind_name holds_to, %holds_to.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %RuntimeConvertFrom.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e8b = fn_decl @Convert.2 [concrete = constants.%Convert.e81] {
// CHECK:STDOUT:     %self.patt: %RuntimeConvertFrom = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %RuntimeConvertFrom = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %RuntimeConvertTo = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %RuntimeConvertTo = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %RuntimeConvertTo.ref: type = name_ref RuntimeConvertTo, file.%RuntimeConvertTo.decl [concrete = constants.%RuntimeConvertTo]
// CHECK:STDOUT:     %self.param: %RuntimeConvertFrom = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%RuntimeConvertFrom.ref [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:     %self: %RuntimeConvertFrom = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %RuntimeConvertTo = out_param call_param1
// CHECK:STDOUT:     %return: ref %RuntimeConvertTo = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .RuntimeConvertTo = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc6_17.1: %tuple.type) {
// CHECK:STDOUT:   %T.loc6_17.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_17.2: %tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.cc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @RuntimeConvertFrom {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%RuntimeConvertFrom
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @RuntimeConvertTo {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%RuntimeConvertTo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: %RuntimeConvertFrom]() -> %return.param_patt: %RuntimeConvertTo {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_58.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_58.2: init %RuntimeConvertTo = class_init (), %return [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   %.loc13_59: init %RuntimeConvertTo = converted %.loc13_58.1, %.loc13_58.2 [concrete = constants.%RuntimeConvertTo.val]
// CHECK:STDOUT:   return %.loc13_59 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc16_6.1: %tuple.type, %A.loc16_20.1: @F.%tuple.elem0.loc16_25.2 (%tuple.elem0)) {
// CHECK:STDOUT:   %T.loc16_6.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc16_6.2: %tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %tuple.elem0.loc16_25.2: type = tuple_access %T.loc16_6.2, element0 [symbolic = %tuple.elem0.loc16_25.2 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %A.loc16_20.2: %tuple.elem0 = bind_symbolic_name A, 1 [symbolic = %A.loc16_20.2 (constants.%A)]
// CHECK:STDOUT:   %A.patt.loc16_20.2: %tuple.elem0 = symbolic_binding_pattern A, 1 [symbolic = %A.patt.loc16_20.2 (constants.%A.patt)]
// CHECK:STDOUT:   %HoldsType.loc16_43.2: type = class_type @HoldsType, @HoldsType(%T.loc16_6.2) [symbolic = %HoldsType.loc16_43.2 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%HoldsType.loc16_43.2 (%HoldsType.cc9) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc16_6.1: %tuple.type](%A.patt.loc16_20.1: @F.%tuple.elem0.loc16_25.2 (%tuple.elem0), %x.param_patt: @F.%HoldsType.loc16_43.2 (%HoldsType.cc9)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%holds_to.param_patt: %HoldsType.066) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %from.patt: %RuntimeConvertFrom = symbolic_binding_pattern from, 0 [symbolic = constants.%from.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_36.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %RuntimeConvertFrom.ref.loc19_41: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:   %.loc19_36.2: ref %RuntimeConvertFrom = temporary_storage
// CHECK:STDOUT:   %.loc19_36.3: init %RuntimeConvertFrom = class_init (), %.loc19_36.2 [concrete = constants.%RuntimeConvertFrom.val]
// CHECK:STDOUT:   %.loc19_36.4: ref %RuntimeConvertFrom = temporary %.loc19_36.2, %.loc19_36.3
// CHECK:STDOUT:   %.loc19_38: ref %RuntimeConvertFrom = converted %.loc19_36.1, %.loc19_36.4
// CHECK:STDOUT:   %RuntimeConvertFrom.ref.loc19_14: type = name_ref RuntimeConvertFrom, file.%RuntimeConvertFrom.decl [concrete = constants.%RuntimeConvertFrom]
// CHECK:STDOUT:   %from: %RuntimeConvertFrom = bind_symbolic_name from, 0, %.loc19_38 [symbolic = constants.%from]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %from.ref: %RuntimeConvertFrom = name_ref from, %from [symbolic = constants.%from]
// CHECK:STDOUT:   %holds_to.ref: %HoldsType.066 = name_ref holds_to, %holds_to
// CHECK:STDOUT:   %impl.elem0: %.31e = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Convert.e81]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method constants.%from, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %.loc30_19.1: ref %RuntimeConvertTo = temporary_storage
// CHECK:STDOUT:   %Convert.call: init %RuntimeConvertTo = call %bound_method(constants.%from) to %.loc30_19.1
// CHECK:STDOUT:   %.loc30_19.2: init %RuntimeConvertTo = converted constants.%from, %Convert.call
// CHECK:STDOUT:   %.loc30_19.3: ref %RuntimeConvertTo = temporary %.loc30_19.1, %.loc30_19.2
// CHECK:STDOUT:   %.loc30_19.4: %RuntimeConvertTo = bind_value %.loc30_19.3
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple, <error>) [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%holds_to.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.loc6_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%RuntimeConvertTo) {
// CHECK:STDOUT:   %Dest => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %Dest.patt => constants.%RuntimeConvertTo
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.580
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.50a
// CHECK:STDOUT:   %Convert => constants.%Convert.993
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.46b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%RuntimeConvertTo, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %Dest => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.580
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet
// CHECK:STDOUT:   %Self.as_type => constants.%RuntimeConvertFrom
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%A) {
// CHECK:STDOUT:   %T.loc16_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc16_6.2 => constants.%T
// CHECK:STDOUT:   %tuple.elem0.loc16_25.2 => constants.%tuple.elem0
// CHECK:STDOUT:   %A.loc16_20.2 => constants.%A
// CHECK:STDOUT:   %A.patt.loc16_20.2 => constants.%A
// CHECK:STDOUT:   %HoldsType.loc16_43.2 => constants.%HoldsType.cc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(@F.%T.loc16_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple) {
// CHECK:STDOUT:   %T.loc6_17.2 => constants.%tuple
// CHECK:STDOUT:   %T.patt.loc6_17.2 => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple, <error>) {
// CHECK:STDOUT:   %T.loc16_6.2 => constants.%tuple
// CHECK:STDOUT:   %T.patt.loc16_6.2 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0.loc16_25.2 => constants.%RuntimeConvertTo
// CHECK:STDOUT:   %A.loc16_20.2 => <error>
// CHECK:STDOUT:   %A.patt.loc16_20.2 => <error>
// CHECK:STDOUT:   %HoldsType.loc16_43.2 => constants.%HoldsType.066
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_value_to_generic_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtin_conversions/no_prelude/convert_class_value_to_generic_facet_value_value.carbon

// --- core.carbon

package Core;

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

// --- convert_class_value_to_generic_facet_value_value.carbon

library "[[@TEST_NAME]]";

import Core;

interface Generic(Scalar:! type) {
  fn F();
}

class GenericParam {}

class ImplsGeneric {}
impl ImplsGeneric as Generic(GenericParam) {
  fn F() {}
}

fn CallGenericMethod[T:! type, U:! Generic(T)](a: U, s: T) {
  U.F();
}

fn G() {
  CallGenericMethod({} as ImplsGeneric, {} as GenericParam);
}

// --- multiple_generic_params_one_fixed_one_deduced.carbon
library "[[@TEST_NAME]]";

import Core;

interface I(V:! type, W:! type) {}

class C {}

impl forall [T:! type] C as I(T, ()) {}

fn A[T:! I({}, ())](t: T) {}

fn B() {
  A({} as C);
}

// --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

import Core;

interface I(V:! type, W:! type) {}

class C {}

impl forall [T:! type] C as I(T, ()) {}

fn A[T:! I({}, {})](t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+10]]:3: error: cannot implicitly convert value of type `type` to `I({}, {})` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   A({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(I({}, {}))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   A({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_constraint_with_fixed_specific.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I({}, {})](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C);
}

// --- fail_mismatch_impl_self_with_fixed_specific.carbon
library "[[@TEST_NAME]]";

import Core;

interface I {}

class C(V:! type, W:! type) {}

impl forall [T:! type] C(T, ()) as I {}

fn A[T:! I](t: T) {}

fn B() {
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+10]]:3: error: cannot implicitly convert value of type `type` to `I` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   A({} as C({}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   A({} as C({}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mismatch_impl_self_with_fixed_specific.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! I](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({} as C({}, {}));
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc4_22.1: type) {
// CHECK:STDOUT:   %T.loc4_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc4_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T.loc4_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_32.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc4_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_20.1: type = splice_block %.loc5_20.3 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_20.2: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc5_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_20.3: type = converted %Self.ref, %Self.as_type.loc5_20.2 [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc5_32.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc5_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%T.loc4_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type.loc5_20.1 (%Self.as_type)]() -> @Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc4_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc4_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_class_value_to_generic_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Scalar: type = bind_symbolic_name Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Scalar.patt: type = symbolic_binding_pattern Scalar, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.c21: type = generic_interface_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.c21 = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.91ccba.1: type = facet_type <@Generic, @Generic(%Scalar)> [symbolic]
// CHECK:STDOUT:   %Self: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.1: type = fn_type @F.1, @Generic(%Scalar) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.1: %F.type.f439a9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.de973d.1: type = assoc_entity_type %Generic.type.91ccba.1 [symbolic]
// CHECK:STDOUT:   %assoc0.29ce53.1: %Generic.assoc_type.de973d.1 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %GenericParam: type = class_type @GenericParam [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplsGeneric: type = class_type @ImplsGeneric [concrete]
// CHECK:STDOUT:   %Generic.type.769: type = facet_type <@Generic, @Generic(%GenericParam)> [concrete]
// CHECK:STDOUT:   %F.type.4cf: type = fn_type @F.1, @Generic(%GenericParam) [concrete]
// CHECK:STDOUT:   %F.118: %F.type.4cf = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.assoc_type.9f1: type = assoc_entity_type %Generic.type.769 [concrete]
// CHECK:STDOUT:   %assoc0.9b7: %Generic.assoc_type.9f1 = assoc_entity element0, @Generic.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.17b: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.a56: %F.type.17b = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.facet.8bd: %Generic.type.769 = facet_value %ImplsGeneric, (%impl_witness) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type.91ccba.2: type = facet_type <@Generic, @Generic(%T)> [symbolic]
// CHECK:STDOUT:   %U: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [concrete]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.7b2: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.type.f439a9.2: type = fn_type @F.1, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %F.8a2d67.2: %F.type.f439a9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Generic.assoc_type.de973d.2: type = assoc_entity_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %assoc0.29ce53.2: %Generic.assoc_type.de973d.2 = assoc_entity element0, @Generic.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.02a: <witness> = require_complete_type %Generic.type.91ccba.2 [symbolic]
// CHECK:STDOUT:   %U.as_wit.iface0: <witness> = facet_access_witness %U, element0 [symbolic]
// CHECK:STDOUT:   %Generic.facet.0aa: %Generic.type.91ccba.2 = facet_value %U.as_type, (%U.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.10b: type = fn_type_with_self_type %F.type.f439a9.2, %Generic.facet.0aa [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.10b = impl_witness_access %U.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.1(%T, %Generic.facet.0aa) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsGeneric.val: %ImplsGeneric = struct_value () [concrete]
// CHECK:STDOUT:   %GenericParam.val: %GenericParam = struct_value () [concrete]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod, @CallGenericMethod(%GenericParam, %Generic.facet.8bd) [concrete]
// CHECK:STDOUT:   %complete_type.997: <witness> = complete_type_witness %Generic.type.769 [concrete]
// CHECK:STDOUT:   %.1eb: type = fn_type_with_self_type %F.type.4cf, %Generic.facet.8bd [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.a56, @F.1(%GenericParam, %Generic.facet.8bd) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:     .GenericParam = %GenericParam.decl
// CHECK:STDOUT:     .ImplsGeneric = %ImplsGeneric.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type.c21 = interface_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %Scalar.patt.loc6_19.1: type = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc6_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Scalar.loc6_19.1: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc6_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericParam.decl: type = class_decl @GenericParam [concrete = constants.%GenericParam] {} {}
// CHECK:STDOUT:   %ImplsGeneric.decl: type = class_decl @ImplsGeneric [concrete = constants.%ImplsGeneric] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:     %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:     %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:     %Generic.type: type = facet_type <@Generic, @Generic(constants.%GenericParam)> [concrete = constants.%Generic.type.769]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [concrete = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %T.patt.loc17_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc17_32.1: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc17_32.2 (constants.%U.patt)]
// CHECK:STDOUT:     %a.patt: @CallGenericMethod.%U.as_type.loc17_51.2 (%U.as_type) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @CallGenericMethod.%U.as_type.loc17_51.2 (%U.as_type) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %s.patt: @CallGenericMethod.%T.loc17_22.2 (%T) = binding_pattern s
// CHECK:STDOUT:     %s.param_patt: @CallGenericMethod.%T.loc17_22.2 (%T) = value_param_pattern %s.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc17_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_22.2 (constants.%T)]
// CHECK:STDOUT:     %.loc17_45: type = splice_block %Generic.type.loc17_45.1 [symbolic = %Generic.type.loc17_45.2 (constants.%Generic.type.91ccba.2)] {
// CHECK:STDOUT:       %Generic.ref: %Generic.type.c21 = name_ref Generic, file.%Generic.decl [concrete = constants.%Generic.generic]
// CHECK:STDOUT:       %T.ref.loc17_44: type = name_ref T, %T.loc17_22.1 [symbolic = %T.loc17_22.2 (constants.%T)]
// CHECK:STDOUT:       %Generic.type.loc17_45.1: type = facet_type <@Generic, @Generic(constants.%T)> [symbolic = %Generic.type.loc17_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc17_32.1: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) = bind_symbolic_name U, 1 [symbolic = %U.loc17_32.2 (constants.%U)]
// CHECK:STDOUT:     %a.param: @CallGenericMethod.%U.as_type.loc17_51.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_51.1: type = splice_block %.loc17_51.2 [symbolic = %U.as_type.loc17_51.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref.loc17: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc17_32.1 [symbolic = %U.loc17_32.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc17_51.1: type = facet_access_type %U.ref.loc17 [symbolic = %U.as_type.loc17_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc17_51.2: type = converted %U.ref.loc17, %U.as_type.loc17_51.1 [symbolic = %U.as_type.loc17_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @CallGenericMethod.%U.as_type.loc17_51.2 (%U.as_type) = bind_name a, %a.param
// CHECK:STDOUT:     %s.param: @CallGenericMethod.%T.loc17_22.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc17_57: type = name_ref T, %T.loc17_22.1 [symbolic = %T.loc17_22.2 (constants.%T)]
// CHECK:STDOUT:     %s: @CallGenericMethod.%T.loc17_22.2 (%T) = bind_name s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic(%Scalar.loc6_19.1: type) {
// CHECK:STDOUT:   %Scalar.loc6_19.2: type = bind_symbolic_name Scalar, 0 [symbolic = %Scalar.loc6_19.2 (constants.%Scalar)]
// CHECK:STDOUT:   %Scalar.patt.loc6_19.2: type = symbolic_binding_pattern Scalar, 0 [symbolic = %Scalar.patt.loc6_19.2 (constants.%Scalar.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type: type = facet_type <@Generic, @Generic(%Scalar.loc6_19.2)> [symbolic = %Generic.type (constants.%Generic.type.91ccba.1)]
// CHECK:STDOUT:   %Self.2: %Generic.type.91ccba.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%Scalar.loc6_19.2) [symbolic = %F.type (constants.%F.type.f439a9.1)]
// CHECK:STDOUT:   %F: @Generic.%F.type (%F.type.f439a9.1) = struct_value () [symbolic = %F (constants.%F.8a2d67.1)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @Generic.%Generic.type (%Generic.type.91ccba.1) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.de973d.1)]
// CHECK:STDOUT:   %assoc0.loc7_9.2: @Generic.%Generic.assoc_type (%Generic.assoc_type.de973d.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc7_9.2 (constants.%assoc0.29ce53.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @Generic.%F.type (%F.type.f439a9.1) = fn_decl @F.1 [symbolic = @Generic.%F (constants.%F.8a2d67.1)] {} {}
// CHECK:STDOUT:     %assoc0.loc7_9.1: @Generic.%Generic.assoc_type (%Generic.assoc_type.de973d.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc7_9.2 (constants.%assoc0.29ce53.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc7_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %ImplsGeneric.ref as %Generic.type {
// CHECK:STDOUT:   %F.decl: %F.type.17b = fn_decl @F.2 [concrete = constants.%F.a56] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @GenericParam {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%GenericParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ImplsGeneric {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Generic.%Scalar.loc6_19.1: type, @Generic.%Self.1: @Generic.%Generic.type (%Generic.type.91ccba.1)) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericMethod(%T.loc17_22.1: type, %U.loc17_32.1: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2)) {
// CHECK:STDOUT:   %T.loc17_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc17_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Generic.type.loc17_45.2: type = facet_type <@Generic, @Generic(%T.loc17_22.2)> [symbolic = %Generic.type.loc17_45.2 (constants.%Generic.type.91ccba.2)]
// CHECK:STDOUT:   %U.loc17_32.2: %Generic.type.91ccba.2 = bind_symbolic_name U, 1 [symbolic = %U.loc17_32.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc17_32.2: %Generic.type.91ccba.2 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc17_32.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.as_type.loc17_51.2: type = facet_access_type %U.loc17_32.2 [symbolic = %U.as_type.loc17_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_49: <witness> = require_complete_type @CallGenericMethod.%U.as_type.loc17_51.2 (%U.as_type) [symbolic = %require_complete.loc17_49 (constants.%require_complete.7b2)]
// CHECK:STDOUT:   %require_complete.loc17_55: <witness> = require_complete_type @CallGenericMethod.%T.loc17_22.2 (%T) [symbolic = %require_complete.loc17_55 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) [symbolic = %require_complete.loc18 (constants.%require_complete.02a)]
// CHECK:STDOUT:   %Generic.assoc_type: type = assoc_entity_type @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) [symbolic = %Generic.assoc_type (constants.%Generic.assoc_type.de973d.2)]
// CHECK:STDOUT:   %assoc0: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.de973d.2) = assoc_entity element0, @Generic.%F.decl [symbolic = %assoc0 (constants.%assoc0.29ce53.2)]
// CHECK:STDOUT:   %U.as_wit.iface0.loc18_4.2: <witness> = facet_access_witness %U.loc17_32.2, element0 [symbolic = %U.as_wit.iface0.loc18_4.2 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Generic(%T.loc17_22.2) [symbolic = %F.type (constants.%F.type.f439a9.2)]
// CHECK:STDOUT:   %Generic.facet: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) = facet_value %U.as_type.loc17_51.2, (%U.as_wit.iface0.loc18_4.2) [symbolic = %Generic.facet (constants.%Generic.facet.0aa)]
// CHECK:STDOUT:   %.loc18_4.3: type = fn_type_with_self_type %F.type, %Generic.facet [symbolic = %.loc18_4.3 (constants.%.10b)]
// CHECK:STDOUT:   %impl.elem0.loc18_4.2: @CallGenericMethod.%.loc18_4.3 (%.10b) = impl_witness_access %U.as_wit.iface0.loc18_4.2, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc18_4.2: <specific function> = specific_function %impl.elem0.loc18_4.2, @F.1(%T.loc17_22.2, %Generic.facet) [symbolic = %specific_fn.loc18_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc17_22.1: type, %U.patt.loc17_32.1: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2)](%a.param_patt: @CallGenericMethod.%U.as_type.loc17_51.2 (%U.as_type), %s.param_patt: @CallGenericMethod.%T.loc17_22.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref.loc18: @CallGenericMethod.%Generic.type.loc17_45.2 (%Generic.type.91ccba.2) = name_ref U, %U.loc17_32.1 [symbolic = %U.loc17_32.2 (constants.%U)]
// CHECK:STDOUT:     %.loc18_4.1: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.de973d.2) = specific_constant @Generic.%assoc0.loc7_9.1, @Generic(constants.%T) [symbolic = %assoc0 (constants.%assoc0.29ce53.2)]
// CHECK:STDOUT:     %F.ref: @CallGenericMethod.%Generic.assoc_type (%Generic.assoc_type.de973d.2) = name_ref F, %.loc18_4.1 [symbolic = %assoc0 (constants.%assoc0.29ce53.2)]
// CHECK:STDOUT:     %U.as_type.loc18: type = facet_access_type %U.ref.loc18 [symbolic = %U.as_type.loc17_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc18_4.2: type = converted %U.ref.loc18, %U.as_type.loc18 [symbolic = %U.as_type.loc17_51.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %U.as_wit.iface0.loc18_4.1: <witness> = facet_access_witness %U.ref.loc18, element0 [symbolic = %U.as_wit.iface0.loc18_4.2 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc18_4.1: @CallGenericMethod.%.loc18_4.3 (%.10b) = impl_witness_access %U.as_wit.iface0.loc18_4.1, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc18_4.1: <specific function> = specific_function %impl.elem0.loc18_4.1, @F.1(constants.%T, constants.%Generic.facet.0aa) [symbolic = %specific_fn.loc18_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %specific_fn.loc18_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGenericMethod.ref: %CallGenericMethod.type = name_ref CallGenericMethod, file.%CallGenericMethod.decl [concrete = constants.%CallGenericMethod]
// CHECK:STDOUT:   %.loc22_22.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %ImplsGeneric.ref: type = name_ref ImplsGeneric, file.%ImplsGeneric.decl [concrete = constants.%ImplsGeneric]
// CHECK:STDOUT:   %.loc22_22.2: ref %ImplsGeneric = temporary_storage
// CHECK:STDOUT:   %.loc22_22.3: init %ImplsGeneric = class_init (), %.loc22_22.2 [concrete = constants.%ImplsGeneric.val]
// CHECK:STDOUT:   %.loc22_22.4: ref %ImplsGeneric = temporary %.loc22_22.2, %.loc22_22.3
// CHECK:STDOUT:   %.loc22_24.1: ref %ImplsGeneric = converted %.loc22_22.1, %.loc22_22.4
// CHECK:STDOUT:   %.loc22_42.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %GenericParam.ref: type = name_ref GenericParam, file.%GenericParam.decl [concrete = constants.%GenericParam]
// CHECK:STDOUT:   %.loc22_42.2: ref %GenericParam = temporary_storage
// CHECK:STDOUT:   %.loc22_42.3: init %GenericParam = class_init (), %.loc22_42.2 [concrete = constants.%GenericParam.val]
// CHECK:STDOUT:   %.loc22_42.4: ref %GenericParam = temporary %.loc22_42.2, %.loc22_42.3
// CHECK:STDOUT:   %.loc22_44.1: ref %GenericParam = converted %.loc22_42.1, %.loc22_42.4
// CHECK:STDOUT:   %Generic.facet: %Generic.type.769 = facet_value constants.%ImplsGeneric, (constants.%impl_witness) [concrete = constants.%Generic.facet.8bd]
// CHECK:STDOUT:   %.loc22_59: %Generic.type.769 = converted constants.%ImplsGeneric, %Generic.facet [concrete = constants.%Generic.facet.8bd]
// CHECK:STDOUT:   %CallGenericMethod.specific_fn: <specific function> = specific_function %CallGenericMethod.ref, @CallGenericMethod(constants.%GenericParam, %.loc22_59) [concrete = constants.%CallGenericMethod.specific_fn]
// CHECK:STDOUT:   %.loc22_24.2: %ImplsGeneric = bind_value %.loc22_24.1
// CHECK:STDOUT:   %.loc22_44.2: %GenericParam = bind_value %.loc22_44.1
// CHECK:STDOUT:   %CallGenericMethod.call: init %empty_tuple.type = call %CallGenericMethod.specific_fn(%.loc22_24.2, %.loc22_44.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%Scalar) {
// CHECK:STDOUT:   %Scalar.loc6_19.2 => constants.%Scalar
// CHECK:STDOUT:   %Scalar.patt.loc6_19.2 => constants.%Scalar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Scalar, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%Scalar.loc6_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%GenericParam) {
// CHECK:STDOUT:   %Scalar.loc6_19.2 => constants.%GenericParam
// CHECK:STDOUT:   %Scalar.patt.loc6_19.2 => constants.%GenericParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.769
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %F => constants.%F.118
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.9f1
// CHECK:STDOUT:   %assoc0.loc7_9.2 => constants.%assoc0.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%GenericParam, constants.%Generic.facet.8bd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %Scalar.loc6_19.2 => constants.%T
// CHECK:STDOUT:   %Scalar.patt.loc6_19.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.type => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.f439a9.2
// CHECK:STDOUT:   %F => constants.%F.8a2d67.2
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.de973d.2
// CHECK:STDOUT:   %assoc0.loc7_9.2 => constants.%assoc0.29ce53.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc17_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc17_22.2 => constants.%T
// CHECK:STDOUT:   %Generic.type.loc17_45.2 => constants.%Generic.type.91ccba.2
// CHECK:STDOUT:   %U.loc17_32.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc17_32.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc17_51.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@CallGenericMethod.%T.loc17_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Generic.facet.0aa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(@CallGenericMethod.%T.loc17_22.2, @CallGenericMethod.%Generic.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, @G.%.loc22_59) {
// CHECK:STDOUT:   %T.loc17_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc17_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc17_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc17_32.2 => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %U.patt.loc17_32.2 => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %U.as_type.loc17_51.2 => constants.%ImplsGeneric
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_49 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc17_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc18 => constants.%complete_type.997
// CHECK:STDOUT:   %Generic.assoc_type => constants.%Generic.assoc_type.9f1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9b7
// CHECK:STDOUT:   %U.as_wit.iface0.loc18_4.2 => constants.%impl_witness
// CHECK:STDOUT:   %F.type => constants.%F.type.4cf
// CHECK:STDOUT:   %Generic.facet => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %.loc18_4.3 => constants.%.1eb
// CHECK:STDOUT:   %impl.elem0.loc18_4.2 => constants.%F.a56
// CHECK:STDOUT:   %specific_fn.loc18_4.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericMethod(constants.%GenericParam, constants.%Generic.facet.8bd) {
// CHECK:STDOUT:   %T.loc17_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %T.patt.loc17_22.2 => constants.%GenericParam
// CHECK:STDOUT:   %Generic.type.loc17_45.2 => constants.%Generic.type.769
// CHECK:STDOUT:   %U.loc17_32.2 => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %U.patt.loc17_32.2 => constants.%Generic.facet.8bd
// CHECK:STDOUT:   %U.as_type.loc17_51.2 => constants.%ImplsGeneric
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_generic_params_one_fixed_one_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f76: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.bea: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %require_complete.5ce: <witness> = require_complete_type %I.type.bea [symbolic]
// CHECK:STDOUT:   %impl_witness.298: <witness> = impl_witness (), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.202: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %T.e96: %I.type.202 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.5b9: %I.type.202 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.e96 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c8b: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.3d9: <witness> = complete_type_witness %I.type.202 [concrete]
// CHECK:STDOUT:   %impl_witness.e3d: <witness> = impl_witness (), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.202 = facet_value %C, (%impl_witness.e3d) [concrete]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A, @A(%I.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt.loc5_13.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_13.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc5_23.1: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc5_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_13.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc5_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_23.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc9_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_36: type = converted %.loc9_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc9_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc9_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T.8b3) [symbolic = @impl.%impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: %I.type.202 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt.5b9)]
// CHECK:STDOUT:     %t.patt: @A.%T.as_type.loc11_24.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @A.%T.as_type.loc11_24.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_18.1: type = splice_block %I.type [concrete = constants.%I.type.202] {
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc11_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc11_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_18.2: type = converted %.loc11_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc11_18.3: type = converted %.loc11_17, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_tuple.type)> [concrete = constants.%I.type.202]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_6.1: %I.type.202 = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T.e96)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc11_24.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_24.1: type = splice_block %.loc11_24.2 [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.202 = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T.e96)]
// CHECK:STDOUT:       %T.as_type.loc11_24.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_24.2: type = converted %T.ref, %T.as_type.loc11_24.1 [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc11_24.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc5_13.1: type, %W.loc5_23.1: type) {
// CHECK:STDOUT:   %V.loc5_13.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_13.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc5_13.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_13.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.loc5_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_23.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc5_23.2: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_23.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc5_13.2, %W.loc5_23.2)> [symbolic = %I.type (constants.%I.type.f76)]
// CHECK:STDOUT:   %Self.2: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.f76) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %I.type.loc9_36.2: type = facet_type <@I, @I(%T.loc9_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc9_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc9_36.2 (%I.type.bea) [symbolic = %require_complete (constants.%require_complete.5ce)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc9_14.2) [symbolic = %impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc9_36.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc11_6.1: %I.type.202) {
// CHECK:STDOUT:   %T.loc11_6.2: %I.type.202 = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T.e96)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: %I.type.202 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt.5b9)]
// CHECK:STDOUT:   %T.as_type.loc11_24.2: type = facet_access_type %T.loc11_6.2 [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.as_type.loc11_24.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.c8b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc11_6.1: %I.type.202](%t.param_patt: @A.%T.as_type.loc11_24.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc14_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc14_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_6.3: init %C = class_init (), %.loc14_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14_6.4: ref %C = temporary %.loc14_6.2, %.loc14_6.3
// CHECK:STDOUT:   %.loc14_8.1: ref %C = converted %.loc14_6.1, %.loc14_6.4
// CHECK:STDOUT:   %I.facet.loc14_12.1: %I.type.202 = facet_value constants.%C, (constants.%impl_witness.e3d) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc14_12.1: %I.type.202 = converted constants.%C, %I.facet.loc14_12.1 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %I.facet.loc14_12.2: %I.type.202 = facet_value constants.%C, (constants.%impl_witness.e3d) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc14_12.2: %I.type.202 = converted constants.%C, %I.facet.loc14_12.2 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %A.specific_fn: <specific function> = specific_function %A.ref, @A(constants.%I.facet) [concrete = constants.%A.specific_fn]
// CHECK:STDOUT:   %.loc14_8.2: %C = bind_value %.loc14_8.1
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.specific_fn(%.loc14_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%V
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%V.loc5_13.2, %W.loc5_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bea
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc9_36.2 => constants.%I.type.bea
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ce
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%T.loc9_14.2, constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.202
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.e96) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T.e96
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.e96
// CHECK:STDOUT:   %T.as_type.loc11_24.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc9_36.2 => constants.%I.type.202
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.e3d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%I.facet) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%I.facet
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%I.facet
// CHECK:STDOUT:   %T.as_type.loc11_24.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_constraint_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f76: type = facet_type <@I, @I(%V, %W)> [symbolic]
// CHECK:STDOUT:   %Self.770: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.bea: type = facet_type <@I, @I(%T.8b3, %empty_tuple.type)> [symbolic]
// CHECK:STDOUT:   %require_complete.5ce: <witness> = require_complete_type %I.type.bea [symbolic]
// CHECK:STDOUT:   %impl_witness.298: <witness> = impl_witness (), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %I.type.906: type = facet_type <@I, @I(%empty_struct_type, %empty_struct_type)> [concrete]
// CHECK:STDOUT:   %T.4b2: %I.type.906 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6fc: %I.type.906 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.4b2 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d5b: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.202: type = facet_type <@I, @I(%empty_struct_type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %complete_type.3d9: <witness> = complete_type_witness %I.type.202 [concrete]
// CHECK:STDOUT:   %impl_witness.e3d: <witness> = impl_witness (), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.442: type = facet_type <@ImplicitAs, @ImplicitAs(%I.type.906)> [concrete]
// CHECK:STDOUT:   %Convert.type.b43: type = fn_type @Convert, @ImplicitAs(%I.type.906) [concrete]
// CHECK:STDOUT:   %Convert.2fc: %Convert.type.b43 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.128: type = assoc_entity_type %ImplicitAs.type.442 [concrete]
// CHECK:STDOUT:   %assoc0.5ba: %ImplicitAs.assoc_type.128 = assoc_entity element0, imports.%Core.import_ref.207961.1 [concrete]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc5_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst24 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc5_32, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %V.patt.loc5_13.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_13.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc5_23.1: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_23.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc5_13.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_13.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc5_23.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_23.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc9_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_36: type = converted %.loc9_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.type.loc9_36.1: type = facet_type <@I, @I(constants.%T.8b3, constants.%empty_tuple.type)> [symbolic = %I.type.loc9_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T.8b3) [symbolic = @impl.%impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: %I.type.906 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt.6fc)]
// CHECK:STDOUT:     %t.patt: @A.%T.as_type.loc11_24.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @A.%T.as_type.loc11_24.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_18.1: type = splice_block %I.type [concrete = constants.%I.type.906] {
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.loc11_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc11_17: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc11_18.2: type = converted %.loc11_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc11_18.3: type = converted %.loc11_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%empty_struct_type, constants.%empty_struct_type)> [concrete = constants.%I.type.906]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_6.1: %I.type.906 = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc11_24.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_24.1: type = splice_block %.loc11_24.2 [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.906 = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:       %T.as_type.loc11_24.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_24.2: type = converted %T.ref, %T.as_type.loc11_24.1 [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc11_24.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%V.loc5_13.1: type, %W.loc5_23.1: type) {
// CHECK:STDOUT:   %V.loc5_13.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc5_13.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc5_13.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc5_13.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.loc5_23.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc5_23.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc5_23.2: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc5_23.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%V.loc5_13.2, %W.loc5_23.2)> [symbolic = %I.type (constants.%I.type.f76)]
// CHECK:STDOUT:   %Self.2: %I.type.f76 = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.f76) = bind_symbolic_name Self, 2 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %I.type.loc9_36.2: type = facet_type <@I, @I(%T.loc9_14.2, constants.%empty_tuple.type)> [symbolic = %I.type.loc9_36.2 (constants.%I.type.bea)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc9_36.2 (%I.type.bea) [symbolic = %require_complete (constants.%require_complete.5ce)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc9_14.2) [symbolic = %impl_witness (constants.%impl_witness.298)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc9_36.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc11_6.1: %I.type.906) {
// CHECK:STDOUT:   %T.loc11_6.2: %I.type.906 = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T.4b2)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: %I.type.906 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt.6fc)]
// CHECK:STDOUT:   %T.as_type.loc11_24.2: type = facet_access_type %T.loc11_6.2 [symbolic = %T.as_type.loc11_24.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.as_type.loc11_24.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.d5b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc11_6.1: %I.type.906](%t.param_patt: @A.%T.as_type.loc11_24.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc24_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc24_6.3: init %C = class_init (), %.loc24_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_6.4: ref %C = temporary %.loc24_6.2, %.loc24_6.3
// CHECK:STDOUT:   %.loc24_8: ref %C = converted %.loc24_6.1, %.loc24_6.4
// CHECK:STDOUT:   %.loc24_12: %I.type.906 = converted constants.%C, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%T (%T.8b3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%V
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%V.loc5_13.2, %W.loc5_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bea
// CHECK:STDOUT:   %Self.2 => constants.%Self.770
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %I.type.loc9_36.2 => constants.%I.type.bea
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ce
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%T.loc9_14.2, constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.906
// CHECK:STDOUT:   %Self.2 => constants.%Self.770
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.4b2) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T.4b2
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.4b2
// CHECK:STDOUT:   %T.as_type.loc11_24.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type.loc9_36.2 => constants.%I.type.202
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.e3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc5_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc5_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc5_23.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.202
// CHECK:STDOUT:   %Self.2 => constants.%Self.770
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T.8b3, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%I.type.906) {
// CHECK:STDOUT:   %T => constants.%I.type.906
// CHECK:STDOUT:   %T.patt => constants.%I.type.906
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.442
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.b43
// CHECK:STDOUT:   %Convert => constants.%Convert.2fc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.128
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_impl_self_with_fixed_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c6b: type = class_type @C, @C(%V, %W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.463: type = class_type @C, @C(%T.8b3, %empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %impl_witness.1a1: <witness> = impl_witness (), @impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.826: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3ad: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.826 [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c74: type = class_type @C, @C(%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.c74 = struct_value () [concrete]
// CHECK:STDOUT:   %C.83b: type = class_type @C, @C(%empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %impl_witness.366: <witness> = impl_witness (), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c42: type = facet_type <@ImplicitAs, @ImplicitAs(%I.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.7ef: type = fn_type @Convert, @ImplicitAs(%I.type) [concrete]
// CHECK:STDOUT:   %Convert.c77: %Convert.type.7ef = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.167: type = assoc_entity_type %ImplicitAs.type.c42 [concrete]
// CHECK:STDOUT:   %assoc0.ae9: %ImplicitAs.assoc_type.167 = assoc_entity element0, imports.%Core.import_ref.207961.1 [concrete]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc5_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc4_22, loaded [symbolic = @ImplicitAs.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst24 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc5_32, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %V.patt.loc7_9.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc7_9.2 (constants.%V.patt)]
// CHECK:STDOUT:     %W.patt.loc7_19.1: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc7_19.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.loc7_9.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc7_9.2 (constants.%V)]
// CHECK:STDOUT:     %W.loc7_19.1: type = bind_symbolic_name W, 1 [symbolic = %W.loc7_19.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc9_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_31: type = converted %.loc9_30, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.loc9_31.1: type = class_type @C, @C(constants.%T.8b3, constants.%empty_tuple.type) [symbolic = %C.loc9_31.2 (constants.%C.463)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%T.8b3) [symbolic = @impl.%impl_witness (constants.%impl_witness.1a1)]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:     %t.patt: @A.%T.as_type.loc11_16.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @A.%T.as_type.loc11_16.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc11_6.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T.826)]
// CHECK:STDOUT:     %t.param: @A.%T.as_type.loc11_16.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_16.1: type = splice_block %.loc11_16.2 [symbolic = %T.as_type.loc11_16.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T.826)]
// CHECK:STDOUT:       %T.as_type.loc11_16.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_16.2: type = converted %T.ref, %T.as_type.loc11_16.1 [symbolic = %T.as_type.loc11_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @A.%T.as_type.loc11_16.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %C.loc9_31.2: type = class_type @C, @C(%T.loc9_14.2, constants.%empty_tuple.type) [symbolic = %C.loc9_31.2 (constants.%C.463)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%T.loc9_14.2) [symbolic = %impl_witness (constants.%impl_witness.1a1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc9_31.1 as %I.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc7_9.1: type, %W.loc7_19.1: type) {
// CHECK:STDOUT:   %V.loc7_9.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc7_9.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc7_9.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc7_9.2 (constants.%V.patt)]
// CHECK:STDOUT:   %W.loc7_19.2: type = bind_symbolic_name W, 1 [symbolic = %W.loc7_19.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc7_19.2: type = symbolic_binding_pattern W, 1 [symbolic = %W.patt.loc7_19.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.c6b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc11_6.1: %I.type) {
// CHECK:STDOUT:   %T.loc11_6.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T.826)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   %T.as_type.loc11_16.2: type = facet_access_type %T.loc11_6.2 [symbolic = %T.as_type.loc11_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.as_type.loc11_16.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc11_6.1: %I.type](%t.param_patt: @A.%T.as_type.loc11_16.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc24_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc24_14: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_19.1: type = converted %.loc24_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc24_19.2: type = converted %.loc24_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_struct_type, constants.%empty_struct_type) [concrete = constants.%C.c74]
// CHECK:STDOUT:   %.loc24_6.2: ref %C.c74 = temporary_storage
// CHECK:STDOUT:   %.loc24_6.3: init %C.c74 = class_init (), %.loc24_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_6.4: ref %C.c74 = temporary %.loc24_6.2, %.loc24_6.3
// CHECK:STDOUT:   %.loc24_8: ref %C.c74 = converted %.loc24_6.1, %.loc24_6.4
// CHECK:STDOUT:   %.loc24_20: %I.type = converted constants.%C.c74, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%T (%T.8b3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc7_9.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc7_9.2 => constants.%V
// CHECK:STDOUT:   %W.loc7_19.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc7_19.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc7_9.2 => constants.%T.8b3
// CHECK:STDOUT:   %V.patt.loc7_9.2 => constants.%T.8b3
// CHECK:STDOUT:   %W.loc7_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc7_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %C.loc9_31.2 => constants.%C.463
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc9_14.2, constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.826) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T.826
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.826
// CHECK:STDOUT:   %T.as_type.loc11_16.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc7_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc7_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc7_19.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.patt.loc7_19.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc9_31.2 => constants.%C.83b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %V.loc7_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc7_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %W.loc7_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %W.patt.loc7_19.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T.8b3, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%I.type) {
// CHECK:STDOUT:   %T => constants.%I.type
// CHECK:STDOUT:   %T.patt => constants.%I.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c42
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.7ef
// CHECK:STDOUT:   %Convert => constants.%Convert.c77
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.167
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ae9
// CHECK:STDOUT: }
// CHECK:STDOUT:

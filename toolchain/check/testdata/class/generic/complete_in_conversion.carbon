// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon

// --- fail_derived_to_base.carbon

fn Int(N: Core.IntLiteral()) -> type = "int.make_type_signed";

base class B {}

class A(N:! i32) {
  extend base: B;

  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:10: error: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:          ^~~~~~
  var n: Int(N);
}

fn F(a: A(0)*) {
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:15: note: in `A(0)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   let b: B* = a;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  let b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type.913: type = fn_type @Int.loc2 [concrete]
// CHECK:STDOUT:   %Int.779: %Int.type.913 = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40d: <witness> = impl_witness @B.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type.878: type = generic_class_type @Int.1 [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type.878 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int.1, @Int.1(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.dd3: type = class_type @A, @A(%N.51e) [symbolic]
// CHECK:STDOUT:   %A.elem.500: type = unbound_element_type %A.dd3, %B [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.708: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.c68: %Int.as.ImplicitAs.impl.Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.4ad: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.960: %Int.as.ImplicitAs.impl.Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f49: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.f49 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.588: <bound method> = bound_method %N.51e, %Int.as.ImplicitAs.impl.Convert.960 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.960, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f9c: <bound method> = bound_method %N.51e, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.f9c(%N.51e) [symbolic]
// CHECK:STDOUT:   %iN.builtin.8fe: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call [symbolic]
// CHECK:STDOUT:   %require_complete.e34: <witness> = require_complete_type %iN.builtin.8fe [symbolic]
// CHECK:STDOUT:   %A.elem.07f: type = unbound_element_type %A.dd3, %iN.builtin.8fe [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.3b2: <witness> = impl_witness @A.%Destroy.impl_witness_table, @A.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %ptr.72a: type = ptr_type %A.dd3 [symbolic]
// CHECK:STDOUT:   %pattern_type.b2f: type = pattern_type %ptr.72a [symbolic]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op, @A.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.8fe} [symbolic]
// CHECK:STDOUT:   %complete_type.beb: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %A.6fc: type = class_type @A, @A(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ptr.b65: type = ptr_type %A.6fc [concrete]
// CHECK:STDOUT:   %pattern_type.213: type = pattern_type %ptr.b65 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem.d81: type = unbound_element_type %A.6fc, %B [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.0fd: <bound method> = bound_method %int_0.6a9, %Int.as.ImplicitAs.impl.Convert.960 [concrete]
// CHECK:STDOUT:   %bound_method.4b5: <bound method> = bound_method %int_0.6a9, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type.878 = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.85c: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.708) = import_ref Core//prelude/parts/int, loc20_44, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type.913 = fn_decl @Int.loc2 [concrete = constants.%Int.779] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = binding_pattern N [concrete]
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern %N.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc2_27.1: type = splice_block %.loc2_27.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.3: type = converted %IntLiteral.call, %.loc2_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int.1, @Int.1(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.1 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.213 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.213 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.b65 = value_param call_param0
// CHECK:STDOUT:     %.loc15_13: type = splice_block %ptr.loc15 [concrete = constants.%ptr.b65] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc15_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc15_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_12.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc15_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc15_12.2: %i32 = converted %int_0, %.loc15_12.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.6a9) [concrete = constants.%A.6fc]
// CHECK:STDOUT:       %ptr.loc15: type = ptr_type %A [concrete = constants.%ptr.b65]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.b65 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: @B.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.decl: %B.as.Destroy.impl.Op.type = fn_decl @B.as.Destroy.impl.Op [concrete = constants.%B.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @B.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.Destroy.impl(@A.%N.loc6_9.2: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N) [symbolic = %A (constants.%A.dd3)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @A.%Destroy.impl_witness_table, @A.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.3b2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op, @A.as.Destroy.impl(%N) [symbolic = %A.as.Destroy.impl.Op.type (constants.%A.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: @A.as.Destroy.impl.%A.as.Destroy.impl.Op.type (%A.as.Destroy.impl.Op.type) = struct_value () [symbolic = %A.as.Destroy.impl.Op (constants.%A.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @A.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %A.as.Destroy.impl.Op.decl: @A.as.Destroy.impl.%A.as.Destroy.impl.Op.type (%A.as.Destroy.impl.Op.type) = fn_decl @A.as.Destroy.impl.Op [symbolic = @A.as.Destroy.impl.%A.as.Destroy.impl.Op (constants.%A.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @A.as.Destroy.impl.Op.%pattern_type (%pattern_type.b2f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @A.as.Destroy.impl.Op.%pattern_type (%pattern_type.b2f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_18.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @A.as.Destroy.impl.Op.%ptr (%ptr.72a) = value_param call_param0
// CHECK:STDOUT:       %.loc6_18.2: type = splice_block %Self.ref [symbolic = %A (constants.%A.dd3)] {
// CHECK:STDOUT:         %.loc6_18.3: type = specific_constant constants.%A.dd3, @A(constants.%N.51e) [symbolic = %A (constants.%A.dd3)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_18.3 [symbolic = %A (constants.%A.dd3)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @A.as.Destroy.impl.Op.%ptr (%ptr.72a) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @A.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:   impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.40d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .N = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_9.2: %i32) {
// CHECK:STDOUT:   %N.loc6_9.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.1 (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_9.1) [symbolic = %A (constants.%A.dd3)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type %A, constants.%B [symbolic = %A.elem.loc7 (constants.%A.elem.500)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_9.1, constants.%Int.as.ImplicitAs.impl.Convert.960 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.588)]
// CHECK:STDOUT:   %bound_method.loc12_14.3: <bound method> = bound_method %N.loc6_9.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc12_14.3 (constants.%bound_method.f9c)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2: init Core.IntLiteral = call %bound_method.loc12_14.3(%N.loc6_9.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 [symbolic = %iN.builtin (constants.%iN.builtin.8fe)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.e34)]
// CHECK:STDOUT:   %A.elem.loc12: type = unbound_element_type %A, %iN.builtin [symbolic = %A.elem.loc12 (constants.%A.elem.07f)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.8fe)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.beb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.500) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %Int.ref: %Int.type.913 = name_ref Int, file.%Int.decl [concrete = constants.%Int.779]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc6_9.2 [symbolic = %N.loc6_9.1 (constants.%N.51e)]
// CHECK:STDOUT:     %impl.elem0: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.960]
// CHECK:STDOUT:     %bound_method.loc12_14.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.588)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_14.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc12_14.3 (constants.%bound_method.f9c)]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call.loc12_14.1: init Core.IntLiteral = call %bound_method.loc12_14.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc12_14.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc12_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc12_14.2: Core.IntLiteral = converted %N.ref, %.loc12_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Int.call: init type = call %Int.ref(%.loc12_14.2) [symbolic = %iN.builtin (constants.%iN.builtin.8fe)]
// CHECK:STDOUT:     %.loc12_15.1: type = value_of_initializer %Int.call [symbolic = %iN.builtin (constants.%iN.builtin.8fe)]
// CHECK:STDOUT:     %.loc12_15.2: type = converted %Int.call, %.loc12_15.1 [symbolic = %iN.builtin (constants.%iN.builtin.8fe)]
// CHECK:STDOUT:     %.loc12_8: @A.%A.elem.loc12 (%A.elem.07f) = field_decl n, element1 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A.dd3 [symbolic = @A.as.Destroy.impl.%A (constants.%A.dd3)]
// CHECK:STDOUT:     impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @A.as.Destroy.impl(constants.%N.51e) [symbolic = @A.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.3b2)]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness constants.%struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.beb)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.dd3
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .n = %.loc12_8
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.loc2(%N.param: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op(%self.param: %ptr.e79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.Destroy.impl.Op(@A.%N.loc6_9.2: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N) [symbolic = %A (constants.%A.dd3)]
// CHECK:STDOUT:   %ptr: type = ptr_type %A [symbolic = %ptr (constants.%ptr.72a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.b2f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.as.Destroy.impl.Op.%ptr (%ptr.72a)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.b65) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.960 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %ptr.b65 = name_ref a, %a
// CHECK:STDOUT:   %.loc20_11: type = splice_block %ptr.loc20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_15.1: ref %A.6fc = deref %a.ref
// CHECK:STDOUT:   %.loc20_15.2: ref %B = class_element_access %.loc20_15.1, element0
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc20_15.2
// CHECK:STDOUT:   %.loc20_15.3: %ptr.e79 = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.e79 = bind_name b, %.loc20_15.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc6_9.1 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.Destroy.impl(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %A => constants.%A.dd3
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.3b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.Destroy.impl.Op(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %A => constants.%A.dd3
// CHECK:STDOUT:   %ptr => constants.%ptr.72a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc6_9.1 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.6fc
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.d81
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.0fd
// CHECK:STDOUT:   %bound_method.loc12_14.3 => constants.%bound_method.4b5
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 => constants.%int_0.5c6
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc12 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc13_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

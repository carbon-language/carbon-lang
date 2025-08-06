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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon

// --- int_match.carbon

library "[[@TEST_NAME]]";

class C(a:! i32) {}
class D(b:! C(1_000));
class D(b:! C(1_000)) {}

// --- fail_int_mismatch.carbon

library "[[@TEST_NAME]]";

class C(a:! i32) {}
class D(b:! C(1000));
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class D(b:! C(1_000)) {}
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class D(b:! C(1000));
// CHECK:STDERR:               ^~~~
// CHECK:STDERR:
class D(b:! C(1_000)) {}

// CHECK:STDOUT: --- int_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.1ba: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%a) [symbolic]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %C.506 [symbolic]
// CHECK:STDOUT:   %pattern_type.d64: type = pattern_type %ptr.128 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%a) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1000.ff9: Core.IntLiteral = int_value 1000 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1000.ff9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1000.ff9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1000.1b6: %i32 = int_value 1000 [concrete]
// CHECK:STDOUT:   %C.262: type = class_type @C, @C(%int_1000.1b6) [concrete]
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dfb: type = pattern_type %C.262 [concrete]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D, @D(%b) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.5ad: <witness> = impl_witness @D.%Destroy.impl_witness_table, @D.as.Destroy.impl(%b) [symbolic]
// CHECK:STDOUT:   %ptr.779: type = ptr_type %D [symbolic]
// CHECK:STDOUT:   %pattern_type.bf6: type = pattern_type %ptr.779 [symbolic]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%b) [symbolic]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4_9.2: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc5: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt: %pattern_type.dfb = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_20.1: type = splice_block %C.loc5 [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref.loc5: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000.loc5: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0.loc5: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc5_20.1: <bound method> = bound_method %int_1000.loc5, %impl.elem0.loc5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc5_20.2: <bound method> = bound_method %int_1000.loc5, %specific_fn.loc5 [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %i32 = call %bound_method.loc5_20.2(%int_1000.loc5) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_20.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_20.3: %i32 = converted %int_1000.loc5, %.loc5_20.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C.loc5: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc5_9.2: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc6: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt: %pattern_type.dfb = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_20.1: type = splice_block %C.loc6 [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref.loc6: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000.loc6: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0.loc6: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc6_20.1: <bound method> = bound_method %int_1000.loc6, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc6_20.2: <bound method> = bound_method %int_1000.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_20.2(%int_1000.loc6) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc6_20.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc6_20.3: %i32 = converted %int_1000.loc6, %.loc6_20.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C.loc6: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc6: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%a.loc4_9.2: %i32) {
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%a) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.1ba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%a) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%C.506 as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_18.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = value_param call_param0
// CHECK:STDOUT:       %.loc4_18.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.506)] {
// CHECK:STDOUT:         %.loc4_18.3: type = specific_constant constants.%C.506, @C(constants.%a) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_18.3 [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.Destroy.impl(@D.%b.loc6: %C.262) {
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic = %b (constants.%b)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @D.%Destroy.impl_witness_table, @D.as.Destroy.impl(%b) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.5ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%b) [symbolic = %D.as.Destroy.impl.Op.type (constants.%D.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: @D.as.Destroy.impl.%D.as.Destroy.impl.Op.type (%D.as.Destroy.impl.Op.type) = struct_value () [symbolic = %D.as.Destroy.impl.Op (constants.%D.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%D as constants.%Destroy.type {
// CHECK:STDOUT:     %D.as.Destroy.impl.Op.decl: @D.as.Destroy.impl.%D.as.Destroy.impl.Op.type (%D.as.Destroy.impl.Op.type) = fn_decl @D.as.Destroy.impl.Op [symbolic = @D.as.Destroy.impl.%D.as.Destroy.impl.Op (constants.%D.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @D.as.Destroy.impl.Op.%pattern_type (%pattern_type.bf6) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @D.as.Destroy.impl.Op.%pattern_type (%pattern_type.bf6) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @D.as.Destroy.impl.Op.%ptr (%ptr.779) = value_param call_param0
// CHECK:STDOUT:       %.loc6_23.2: type = splice_block %Self.ref [symbolic = %D (constants.%D)] {
// CHECK:STDOUT:         %.loc6_23.3: type = specific_constant constants.%D, @D(constants.%b) [symbolic = %D (constants.%D)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_23.3 [symbolic = %D (constants.%D)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @D.as.Destroy.impl.Op.%ptr (%ptr.779) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @D.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc4_9.2: %i32) {
// CHECK:STDOUT:   %a.loc4_9.1: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%a) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.1ba)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc5_9.2: %C.262) {
// CHECK:STDOUT:   %b.loc5_9.1: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @D.as.Destroy.impl(constants.%b) [symbolic = @D.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.5ad)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%a.loc4_9.2: %i32) {
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%a) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.128)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.as.Destroy.impl.Op(@D.%b.loc6: %C.262) {
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic = %b (constants.%b)]
// CHECK:STDOUT:   %D: type = class_type @D, @D(%b) [symbolic = %D (constants.%D)]
// CHECK:STDOUT:   %ptr: type = ptr_type %D [symbolic = %ptr (constants.%ptr.779)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.bf6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @D.as.Destroy.impl.Op.%ptr (%ptr.779)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc4_9.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.1ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %C => constants.%C.506
// CHECK:STDOUT:   %ptr => constants.%ptr.128
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.1b6) {
// CHECK:STDOUT:   %a.loc4_9.1 => constants.%int_1000.1b6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%b) {
// CHECK:STDOUT:   %b.loc5_9.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.Destroy.impl(constants.%b) {
// CHECK:STDOUT:   %b => constants.%b
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.5ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.Destroy.impl.Op(constants.%b) {
// CHECK:STDOUT:   %b => constants.%b
// CHECK:STDOUT:   %D => constants.%D
// CHECK:STDOUT:   %ptr => constants.%ptr.779
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bf6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.1ba: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%a) [symbolic]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %C.506 [symbolic]
// CHECK:STDOUT:   %pattern_type.d64: type = pattern_type %ptr.128 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%a) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1000.ff9: Core.IntLiteral = int_value 1000 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1000.ff9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1000.ff9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1000.1b6: %i32 = int_value 1000 [concrete]
// CHECK:STDOUT:   %C.262: type = class_type @C, @C(%int_1000.1b6) [concrete]
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dfb: type = pattern_type %C.262 [concrete]
// CHECK:STDOUT:   %D.type.bbd080.1: type = generic_class_type @D.loc5 [concrete]
// CHECK:STDOUT:   %D.generic.4e2319.1: %D.type.bbd080.1 = struct_value () [concrete]
// CHECK:STDOUT:   %D.type.bbd080.2: type = generic_class_type @D.loc13 [concrete]
// CHECK:STDOUT:   %D.generic.4e2319.2: %D.type.bbd080.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.126b2d.2: type = class_type @D.loc13, @D.loc13(%b) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.5ad: <witness> = impl_witness @D.loc13.%Destroy.impl_witness_table, @D.as.Destroy.impl(%b) [symbolic]
// CHECK:STDOUT:   %ptr.779: type = ptr_type %D.126b2d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.bf6: type = pattern_type %ptr.779 [symbolic]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%b) [symbolic]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4_9.2: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc5: %D.type.bbd080.1 = class_decl @D.loc5 [concrete = constants.%D.generic.4e2319.1] {
// CHECK:STDOUT:     %b.patt: %pattern_type.dfb = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %C [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc5_19.1: <bound method> = bound_method %int_1000, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc5_19.2: <bound method> = bound_method %int_1000, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_19.2(%int_1000) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_19.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_19.3: %i32 = converted %int_1000, %.loc5_19.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc5_9.2: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc13: %D.type.bbd080.2 = class_decl @D.loc13 [concrete = constants.%D.generic.4e2319.2] {
// CHECK:STDOUT:     %b.patt: %pattern_type.dfb = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_20.1: type = splice_block %C [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc13_20.1: <bound method> = bound_method %int_1000, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc13_20.2: <bound method> = bound_method %int_1000, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_20.2(%int_1000) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc13_20.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc13_20.3: %i32 = converted %int_1000, %.loc13_20.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc13_9.2: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc13_9.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%a.loc4_9.2: %i32) {
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%a) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.1ba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%a) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%C.506 as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_18.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = value_param call_param0
// CHECK:STDOUT:       %.loc4_18.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.506)] {
// CHECK:STDOUT:         %.loc4_18.3: type = specific_constant constants.%C.506, @C(constants.%a) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_18.3 [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.Destroy.impl(@D.loc13.%b.loc13_9.2: %C.262) {
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic = %b (constants.%b)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @D.loc13.%Destroy.impl_witness_table, @D.as.Destroy.impl(%b) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.5ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%b) [symbolic = %D.as.Destroy.impl.Op.type (constants.%D.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: @D.as.Destroy.impl.%D.as.Destroy.impl.Op.type (%D.as.Destroy.impl.Op.type) = struct_value () [symbolic = %D.as.Destroy.impl.Op (constants.%D.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%D.126b2d.2 as constants.%Destroy.type {
// CHECK:STDOUT:     %D.as.Destroy.impl.Op.decl: @D.as.Destroy.impl.%D.as.Destroy.impl.Op.type (%D.as.Destroy.impl.Op.type) = fn_decl @D.as.Destroy.impl.Op [symbolic = @D.as.Destroy.impl.%D.as.Destroy.impl.Op (constants.%D.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @D.as.Destroy.impl.Op.%pattern_type (%pattern_type.bf6) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @D.as.Destroy.impl.Op.%pattern_type (%pattern_type.bf6) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc13_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @D.as.Destroy.impl.Op.%ptr (%ptr.779) = value_param call_param0
// CHECK:STDOUT:       %.loc13_23.2: type = splice_block %Self.ref [symbolic = %D (constants.%D.126b2d.2)] {
// CHECK:STDOUT:         %.loc13_23.3: type = specific_constant constants.%D.126b2d.2, @D.loc13(constants.%b) [symbolic = %D (constants.%D.126b2d.2)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc13_23.3 [symbolic = %D (constants.%D.126b2d.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @D.as.Destroy.impl.Op.%ptr (%ptr.779) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @D.loc13.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc4_9.2: %i32) {
// CHECK:STDOUT:   %a.loc4_9.1: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%a) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.1ba)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc5(%b.loc5_9.2: %C.262) {
// CHECK:STDOUT:   %b.loc5_9.1: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc13(%b.loc13_9.2: %C.262) {
// CHECK:STDOUT:   %b.loc13_9.1: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc13_9.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @D.as.Destroy.impl(constants.%b) [symbolic = @D.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.5ad)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.126b2d.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%a.loc4_9.2: %i32) {
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%a) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.128)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.as.Destroy.impl.Op(@D.loc13.%b.loc13_9.2: %C.262) {
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic = %b (constants.%b)]
// CHECK:STDOUT:   %D: type = class_type @D.loc13, @D.loc13(%b) [symbolic = %D (constants.%D.126b2d.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %D [symbolic = %ptr (constants.%ptr.779)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.bf6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @D.as.Destroy.impl.Op.%ptr (%ptr.779)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc4_9.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.1ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %C => constants.%C.506
// CHECK:STDOUT:   %ptr => constants.%ptr.128
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.1b6) {
// CHECK:STDOUT:   %a.loc4_9.1 => constants.%int_1000.1b6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc5(constants.%b) {
// CHECK:STDOUT:   %b.loc5_9.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc13(constants.%b) {
// CHECK:STDOUT:   %b.loc13_9.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.Destroy.impl(constants.%b) {
// CHECK:STDOUT:   %b => constants.%b
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.5ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.Destroy.impl.Op(constants.%b) {
// CHECK:STDOUT:   %b => constants.%b
// CHECK:STDOUT:   %D => constants.%D.126b2d.2
// CHECK:STDOUT:   %ptr => constants.%ptr.779
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bf6
// CHECK:STDOUT: }
// CHECK:STDOUT:

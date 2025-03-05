// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/tuple.carbon

// --- tuple_type.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[T:! type, U:! type](pair: (T, U)) -> U { return F(pair); }

fn G(pair: (C, D)) -> D {
  return F(pair);
}

// --- tuple_value.carbon

library "[[@TEST_NAME]]";

class HasPair(Pair:! (i32, i32)) {}

fn F[A:! i32, B:! i32](h: HasPair((A, B))) -> i32 { return B; }

fn G(h: HasPair((1, 2))) -> i32 {
  return F(h);
}

// --- fail_inconsistent.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[T:! type](pair: (T, T)) -> T;

fn G(pair: (C, D)) -> D {
  // CHECK:STDERR: fail_inconsistent.carbon:[[@LINE+7]]:10: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   return F(pair);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_inconsistent.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! type](pair: (T, T)) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(pair);
}

// CHECK:STDOUT: --- tuple_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %F.specific_fn.dd9: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.e8a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.4a7: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.53b: <witness> = complete_type_witness %tuple.type.e8a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc7_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc7_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %pair.patt: @F.%tuple.type (%tuple.type.30b) = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: @F.%tuple.type (%tuple.type.30b) = value_param_pattern %pair.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%U.loc7_16.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%U.loc7_16.2 (%U) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_43: type = name_ref U, %U.loc7_16.1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc7_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.30b) = value_param call_param0
// CHECK:STDOUT:     %.loc7_37.1: type = splice_block %.loc7_37.3 [symbolic = %tuple.type (constants.%tuple.type.30b)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc7_36: type = name_ref U, %U.loc7_16.1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:       %.loc7_37.2: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc7_36)
// CHECK:STDOUT:       %.loc7_37.3: type = converted %.loc7_37.2, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.30b) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc7_16.2 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc7_16.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.patt: %tuple.type.e8a = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: %tuple.type.e8a = value_param_pattern %pair.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %pair.param: %tuple.type.e8a = value_param call_param0
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type.e8a] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc9_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc9_16)
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type.e8a [concrete = constants.%tuple.type.e8a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.e8a = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type, %U.loc7_16.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc7_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc7_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc7_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%T.loc7_6.2 (%T), @F.%U.loc7_16.2 (%U)) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_40: <witness> = require_complete_type @F.%U.loc7_16.2 (%U) [symbolic = %require_complete.loc7_40 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %require_complete.loc7_30: <witness> = require_complete_type @F.%tuple.type (%tuple.type.30b) [symbolic = %require_complete.loc7_30 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2, %U.loc7_16.2) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.dd9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc7_6.1: type, %U.patt.loc7_16.1: type](%pair.param_patt: @F.%tuple.type (%tuple.type.30b)) -> @F.%U.loc7_16.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %pair.ref: @F.%tuple.type (%tuple.type.30b) = name_ref pair, %pair
// CHECK:STDOUT:     %F.specific_fn.loc7_54.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.dd9)]
// CHECK:STDOUT:     %F.call: init @F.%U.loc7_16.2 (%U) = call %F.specific_fn.loc7_54.1(%pair.ref)
// CHECK:STDOUT:     %.loc7_61.1: @F.%U.loc7_16.2 (%U) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_61.2: @F.%U.loc7_16.2 (%U) = converted %F.call, %.loc7_61.1
// CHECK:STDOUT:     return %.loc7_61.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param_patt: %tuple.type.e8a) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.e8a = name_ref pair, %pair
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.4a7]
// CHECK:STDOUT:   %.loc9_20: ref %D = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %D = call %F.specific_fn(%pair.ref) to %.loc9_20
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc7_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_16.2 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_40 => constants.%require_complete.b54
// CHECK:STDOUT:   %require_complete.loc7_30 => constants.%require_complete.fe1
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.dd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc7_6.2, %U.loc7_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %U.loc7_16.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc7_16.2 => constants.%D
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e8a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc7_30 => constants.%complete_type.53b
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.4a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Pair: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic]
// CHECK:STDOUT:   %Pair.patt: %tuple.type.d07 = symbolic_binding_pattern Pair, 0 [symbolic]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [concrete]
// CHECK:STDOUT:   %HasPair.generic: %HasPair.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasPair.920: type = class_type @HasPair, @HasPair(%Pair) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A: %i32 = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %A.patt: %i32 = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %B: %i32 = bind_symbolic_name B, 1 [symbolic]
// CHECK:STDOUT:   %B.patt: %i32 = symbolic_binding_pattern B, 1 [symbolic]
// CHECK:STDOUT:   %tuple.159: %tuple.type.d07 = tuple_value (%A, %B) [symbolic]
// CHECK:STDOUT:   %HasPair.568: type = class_type @HasPair, @HasPair(%tuple.159) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6bc: <witness> = require_complete_type %HasPair.568 [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %HasPair.369: type = class_type @HasPair, @HasPair(%tuple.21c) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasPair = %HasPair.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasPair.decl: %HasPair.type = class_decl @HasPair [concrete = constants.%HasPair.generic] {
// CHECK:STDOUT:     %Pair.patt.loc4_15.1: %tuple.type.d07 = symbolic_binding_pattern Pair, 0 [symbolic = %Pair.patt.loc4_15.2 (constants.%Pair.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_31.1: type = splice_block %.loc4_31.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:       %int_32.loc4_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc4_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_31.2: %tuple.type.24b = tuple_literal (%i32.loc4_23, %i32.loc4_28)
// CHECK:STDOUT:       %.loc4_31.3: type = converted %.loc4_31.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Pair.loc4_15.1: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic = %Pair.loc4_15.2 (constants.%Pair)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt.loc6_6.1: %i32 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc6_6.2 (constants.%A.patt)]
// CHECK:STDOUT:     %B.patt.loc6_15.1: %i32 = symbolic_binding_pattern B, 1 [symbolic = %B.patt.loc6_15.2 (constants.%B.patt)]
// CHECK:STDOUT:     %h.patt: @F.%HasPair.loc6_41.2 (%HasPair.568) = binding_pattern h
// CHECK:STDOUT:     %h.param_patt: @F.%HasPair.loc6_41.2 (%HasPair.568) = value_param_pattern %h.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_47: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_47: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc6_6.1: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc6_6.2 (constants.%A)]
// CHECK:STDOUT:     %.loc6_19: type = splice_block %i32.loc6_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_15.1: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc6_41.2 (%HasPair.568) = value_param call_param0
// CHECK:STDOUT:     %.loc6_41.1: type = splice_block %HasPair.loc6_41.1 [symbolic = %HasPair.loc6_41.2 (constants.%HasPair.568)] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %A.ref: %i32 = name_ref A, %A.loc6_6.1 [symbolic = %A.loc6_6.2 (constants.%A)]
// CHECK:STDOUT:       %B.ref.loc6_39: %i32 = name_ref B, %B.loc6_15.1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:       %.loc6_40: %tuple.type.d07 = tuple_literal (%A.ref, %B.ref.loc6_39)
// CHECK:STDOUT:       %tuple.loc6_40.1: %tuple.type.d07 = tuple_value (%A.ref, %B.ref.loc6_39) [symbolic = %tuple.loc6_40.2 (constants.%tuple.159)]
// CHECK:STDOUT:       %.loc6_41.2: %tuple.type.d07 = converted %.loc6_40, %tuple.loc6_40.1 [symbolic = %tuple.loc6_40.2 (constants.%tuple.159)]
// CHECK:STDOUT:       %HasPair.loc6_41.1: type = class_type @HasPair, @HasPair(constants.%tuple.159) [symbolic = %HasPair.loc6_41.2 (constants.%HasPair.568)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: @F.%HasPair.loc6_41.2 (%HasPair.568) = bind_name h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %h.patt: %HasPair.369 = binding_pattern h
// CHECK:STDOUT:     %h.param_patt: %HasPair.369 = value_param_pattern %h.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %h.param: %HasPair.369 = value_param call_param0
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %HasPair [concrete = constants.%HasPair.369] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:       %.loc8_22.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:       %impl.elem0.loc8_22.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc8_22.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_22.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:       %specific_fn.loc8_22.1: <specific function> = specific_function %impl.elem0.loc8_22.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.2: <bound method> = bound_method %int_1, %specific_fn.loc8_22.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:       %int.convert_checked.loc8_22.1: init %i32 = call %bound_method.loc8_22.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %.loc8_22.2: %i32 = value_of_initializer %int.convert_checked.loc8_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %.loc8_22.3: %i32 = converted %int_1, %.loc8_22.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %impl.elem0.loc8_22.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc8_22.3: <bound method> = bound_method %int_2, %impl.elem0.loc8_22.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:       %specific_fn.loc8_22.2: <specific function> = specific_function %impl.elem0.loc8_22.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.4: <bound method> = bound_method %int_2, %specific_fn.loc8_22.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:       %int.convert_checked.loc8_22.2: init %i32 = call %bound_method.loc8_22.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %.loc8_22.4: %i32 = value_of_initializer %int.convert_checked.loc8_22.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %.loc8_22.5: %i32 = converted %int_2, %.loc8_22.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %tuple: %tuple.type.d07 = tuple_value (%.loc8_22.3, %.loc8_22.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:       %.loc8_23.2: %tuple.type.d07 = converted %.loc8_22.1, %tuple [concrete = constants.%tuple.21c]
// CHECK:STDOUT:       %HasPair: type = class_type @HasPair, @HasPair(constants.%tuple.21c) [concrete = constants.%HasPair.369]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: %HasPair.369 = bind_name h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HasPair(%Pair.loc4_15.1: %tuple.type.d07) {
// CHECK:STDOUT:   %Pair.loc4_15.2: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic = %Pair.loc4_15.2 (constants.%Pair)]
// CHECK:STDOUT:   %Pair.patt.loc4_15.2: %tuple.type.d07 = symbolic_binding_pattern Pair, 0 [symbolic = %Pair.patt.loc4_15.2 (constants.%Pair.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HasPair.920
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc6_6.1: %i32, %B.loc6_15.1: %i32) {
// CHECK:STDOUT:   %A.loc6_6.2: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc6_6.2 (constants.%A)]
// CHECK:STDOUT:   %A.patt.loc6_6.2: %i32 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc6_6.2 (constants.%A.patt)]
// CHECK:STDOUT:   %B.loc6_15.2: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_15.2: %i32 = symbolic_binding_pattern B, 1 [symbolic = %B.patt.loc6_15.2 (constants.%B.patt)]
// CHECK:STDOUT:   %tuple.loc6_40.2: %tuple.type.d07 = tuple_value (%A.loc6_6.2, %B.loc6_15.2) [symbolic = %tuple.loc6_40.2 (constants.%tuple.159)]
// CHECK:STDOUT:   %HasPair.loc6_41.2: type = class_type @HasPair, @HasPair(%tuple.loc6_40.2) [symbolic = %HasPair.loc6_41.2 (constants.%HasPair.568)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%HasPair.loc6_41.2 (%HasPair.568) [symbolic = %require_complete (constants.%require_complete.6bc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%A.patt.loc6_6.1: %i32, %B.patt.loc6_15.1: %i32](%h.param_patt: @F.%HasPair.loc6_41.2 (%HasPair.568)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %B.ref.loc6_60: %i32 = name_ref B, %B.loc6_15.1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:     return %B.ref.loc6_60
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param_patt: %HasPair.369) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %h.ref: %HasPair.369 = name_ref h, %h
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_1.5d2, constants.%int_2.ef8) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%h.ref)
// CHECK:STDOUT:   %.loc9_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_14.2: %i32 = converted %F.call, %.loc9_14.1
// CHECK:STDOUT:   return %.loc9_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%Pair) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%Pair
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%Pair
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.159) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%tuple.159
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%tuple.159
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc6_6.2 => constants.%A
// CHECK:STDOUT:   %A.patt.loc6_6.2 => constants.%A
// CHECK:STDOUT:   %B.loc6_15.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_15.2 => constants.%B
// CHECK:STDOUT:   %tuple.loc6_40.2 => constants.%tuple.159
// CHECK:STDOUT:   %HasPair.loc6_41.2 => constants.%HasPair.568
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(@F.%tuple.loc6_40.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.21c) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%tuple.21c
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%tuple.21c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_1.5d2, constants.%int_2.ef8) {
// CHECK:STDOUT:   %A.loc6_6.2 => constants.%int_1.5d2
// CHECK:STDOUT:   %A.patt.loc6_6.2 => constants.%int_1.5d2
// CHECK:STDOUT:   %B.loc6_15.2 => constants.%int_2.ef8
// CHECK:STDOUT:   %B.patt.loc6_15.2 => constants.%int_2.ef8
// CHECK:STDOUT:   %tuple.loc6_40.2 => constants.%tuple.21c
// CHECK:STDOUT:   %HasPair.loc6_41.2 => constants.%HasPair.369
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d00: type = tuple_type (%T, %T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.e8a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %pair.patt: @F.%tuple.type (%tuple.type.d00) = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: @F.%tuple.type (%tuple.type.d00) = value_param_pattern %pair.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc7_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc7_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_33: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.d00) = value_param call_param0
// CHECK:STDOUT:     %.loc7_27.1: type = splice_block %.loc7_27.3 [symbolic = %tuple.type (constants.%tuple.type.d00)] {
// CHECK:STDOUT:       %T.ref.loc7_23: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc7_26: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:       %.loc7_27.2: %tuple.type.24b = tuple_literal (%T.ref.loc7_23, %T.ref.loc7_26)
// CHECK:STDOUT:       %.loc7_27.3: type = converted %.loc7_27.2, constants.%tuple.type.d00 [symbolic = %tuple.type (constants.%tuple.type.d00)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.d00) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.patt: %tuple.type.e8a = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: %tuple.type.e8a = value_param_pattern %pair.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %pair.param: %tuple.type.e8a = value_param call_param0
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type.e8a] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc9_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc9_16)
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type.e8a [concrete = constants.%tuple.type.e8a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.e8a = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%T.loc7_6.2 (%T), @F.%T.loc7_6.2 (%T)) [symbolic = %tuple.type (constants.%tuple.type.d00)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc7_6.1: type](%pair.param_patt: @F.%tuple.type (%tuple.type.d00)) -> @F.%T.loc7_6.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param_patt: %tuple.type.e8a) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.e8a = name_ref pair, %pair
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d00
// CHECK:STDOUT: }
// CHECK:STDOUT:

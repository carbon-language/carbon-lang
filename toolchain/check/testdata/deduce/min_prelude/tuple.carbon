// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/min_prelude/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/min_prelude/tuple.carbon

// --- tuple_type.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

//@dump-sem-ir-begin
fn F[T:! type, U:! type](pair: (T, U)) -> U {
  //@dump-sem-ir-end
  return F(pair);
}

fn G(pair: (C, D)) -> D {
  //@dump-sem-ir-begin
  return F(pair);
  //@dump-sem-ir-end
}

// --- tuple_value.carbon

library "[[@TEST_NAME]]";

class HasPair(Pair:! (i32, i32)) {}

//@dump-sem-ir-begin
fn F[A:! i32, B:! i32](h: HasPair((A, B))) -> i32 {
  //@dump-sem-ir-end
  return B;
}

fn G(h: HasPair((1, 2))) -> i32 {
  //@dump-sem-ir-begin
  return F(h);
  //@dump-sem-ir-end
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %pattern_type.a32: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %F.specific_fn.dd9: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.e8a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.9ec: type = pattern_type %tuple.type.e8a [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F.specific_fn.4a7: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.53b: <witness> = complete_type_witness %tuple.type.e8a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc8_26 (%pattern_type.65c) = binding_pattern pair [concrete]
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc8_26 (%pattern_type.65c) = value_param_pattern %pair.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc8_40 (%pattern_type.a32) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc8_40 (%pattern_type.a32) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc8_43: type = name_ref U, %U.loc8_16.1 [symbolic = %U.loc8_16.2 (constants.%U)]
// CHECK:STDOUT:     %T.loc8_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc8_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc8_16.2 (constants.%U)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.30b) = value_param call_param0
// CHECK:STDOUT:     %.loc8_37.1: type = splice_block %.loc8_37.3 [symbolic = %tuple.type (constants.%tuple.type.30b)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc8_6.1 [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc8_36: type = name_ref U, %U.loc8_16.1 [symbolic = %U.loc8_16.2 (constants.%U)]
// CHECK:STDOUT:       %.loc8_37.2: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc8_36)
// CHECK:STDOUT:       %.loc8_37.3: type = converted %.loc8_37.2, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.30b) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc8_16.2 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc8_16.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.1: type, %U.loc8_16.1: type) {
// CHECK:STDOUT:   %T.loc8_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:   %U.loc8_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc8_16.2 (constants.%U)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc8_6.2, %U.loc8_16.2) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type.loc8_26: type = pattern_type %tuple.type [symbolic = %pattern_type.loc8_26 (constants.%pattern_type.65c)]
// CHECK:STDOUT:   %pattern_type.loc8_40: type = pattern_type %U.loc8_16.2 [symbolic = %pattern_type.loc8_40 (constants.%pattern_type.a32)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_40: <witness> = require_complete_type %U.loc8_16.2 [symbolic = %require_complete.loc8_40 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %require_complete.loc8_30: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc8_30 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.30b)) -> @F.%U.loc8_16.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.e8a) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.e8a = name_ref pair, %pair
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.4a7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.call: init %D = call %F.specific_fn(%pair.ref) to %.loc13_20
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc8_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc8_16.2 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type.loc8_26 => constants.%pattern_type.65c
// CHECK:STDOUT:   %pattern_type.loc8_40 => constants.%pattern_type.a32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_40 => constants.%require_complete.b54
// CHECK:STDOUT:   %require_complete.loc8_30 => constants.%require_complete.fe1
// CHECK:STDOUT:   %F.specific_fn.loc10_10.2 => constants.%F.specific_fn.dd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc8_6.2 => constants.%C
// CHECK:STDOUT:   %U.loc8_16.2 => constants.%D
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e8a
// CHECK:STDOUT:   %pattern_type.loc8_26 => constants.%pattern_type.9ec
// CHECK:STDOUT:   %pattern_type.loc8_40 => constants.%pattern_type.510
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc8_30 => constants.%complete_type.53b
// CHECK:STDOUT:   %F.specific_fn.loc10_10.2 => constants.%F.specific_fn.4a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [concrete]
// CHECK:STDOUT:   %HasPair.generic: %HasPair.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A: %i32 = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %B: %i32 = bind_symbolic_name B, 1 [symbolic]
// CHECK:STDOUT:   %tuple.159: %tuple.type.d07 = tuple_value (%A, %B) [symbolic]
// CHECK:STDOUT:   %HasPair.568: type = class_type @HasPair, @HasPair(%tuple.159) [symbolic]
// CHECK:STDOUT:   %pattern_type.88c: type = pattern_type %HasPair.568 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6bc: <witness> = require_complete_type %HasPair.568 [symbolic]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %HasPair.369: type = class_type @HasPair, @HasPair(%tuple.21c) [concrete]
// CHECK:STDOUT:   %pattern_type.93e: type = pattern_type %HasPair.369 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.7ce = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.7ce = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:     %h.patt: @F.%pattern_type (%pattern_type.88c) = binding_pattern h [concrete]
// CHECK:STDOUT:     %h.param_patt: @F.%pattern_type (%pattern_type.88c) = value_param_pattern %h.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_47: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_47: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_10: type = splice_block %i32.loc7_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc7_6.1: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc7_6.2 (constants.%A)]
// CHECK:STDOUT:     %.loc7_19: type = splice_block %i32.loc7_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc7_15.1: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc7_15.2 (constants.%B)]
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc7_41.2 (%HasPair.568) = value_param call_param0
// CHECK:STDOUT:     %.loc7_41.1: type = splice_block %HasPair.loc7_41.1 [symbolic = %HasPair.loc7_41.2 (constants.%HasPair.568)] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %A.ref: %i32 = name_ref A, %A.loc7_6.1 [symbolic = %A.loc7_6.2 (constants.%A)]
// CHECK:STDOUT:       %B.ref.loc7: %i32 = name_ref B, %B.loc7_15.1 [symbolic = %B.loc7_15.2 (constants.%B)]
// CHECK:STDOUT:       %.loc7_40: %tuple.type.d07 = tuple_literal (%A.ref, %B.ref.loc7)
// CHECK:STDOUT:       %tuple.loc7_40.1: %tuple.type.d07 = tuple_value (%A.ref, %B.ref.loc7) [symbolic = %tuple.loc7_40.2 (constants.%tuple.159)]
// CHECK:STDOUT:       %.loc7_41.2: %tuple.type.d07 = converted %.loc7_40, %tuple.loc7_40.1 [symbolic = %tuple.loc7_40.2 (constants.%tuple.159)]
// CHECK:STDOUT:       %HasPair.loc7_41.1: type = class_type @HasPair, @HasPair(constants.%tuple.159) [symbolic = %HasPair.loc7_41.2 (constants.%HasPair.568)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: @F.%HasPair.loc7_41.2 (%HasPair.568) = bind_name h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc7_6.1: %i32, %B.loc7_15.1: %i32) {
// CHECK:STDOUT:   %A.loc7_6.2: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc7_6.2 (constants.%A)]
// CHECK:STDOUT:   %B.loc7_15.2: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc7_15.2 (constants.%B)]
// CHECK:STDOUT:   %tuple.loc7_40.2: %tuple.type.d07 = tuple_value (%A.loc7_6.2, %B.loc7_15.2) [symbolic = %tuple.loc7_40.2 (constants.%tuple.159)]
// CHECK:STDOUT:   %HasPair.loc7_41.2: type = class_type @HasPair, @HasPair(%tuple.loc7_40.2) [symbolic = %HasPair.loc7_41.2 (constants.%HasPair.568)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HasPair.loc7_41.2 [symbolic = %pattern_type (constants.%pattern_type.88c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasPair.loc7_41.2 [symbolic = %require_complete (constants.%require_complete.6bc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%h.param: @F.%HasPair.loc7_41.2 (%HasPair.568)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param: %HasPair.369) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %h.ref: %HasPair.369 = name_ref h, %h
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_1.5d2, constants.%int_2.ef8) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%h.ref)
// CHECK:STDOUT:   %.loc14_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc14_14.2: %i32 = converted %F.call, %.loc14_14.1
// CHECK:STDOUT:   return %.loc14_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc7_6.2 => constants.%A
// CHECK:STDOUT:   %B.loc7_15.2 => constants.%B
// CHECK:STDOUT:   %tuple.loc7_40.2 => constants.%tuple.159
// CHECK:STDOUT:   %HasPair.loc7_41.2 => constants.%HasPair.568
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_1.5d2, constants.%int_2.ef8) {
// CHECK:STDOUT:   %A.loc7_6.2 => constants.%int_1.5d2
// CHECK:STDOUT:   %B.loc7_15.2 => constants.%int_2.ef8
// CHECK:STDOUT:   %tuple.loc7_40.2 => constants.%tuple.21c
// CHECK:STDOUT:   %HasPair.loc7_41.2 => constants.%HasPair.369
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.93e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:

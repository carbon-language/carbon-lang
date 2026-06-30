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

fn F[A:! i32, B:! i32](unused h: HasPair((A, B))) -> i32 { return B; }

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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %pair.param_patt.185: %pattern_type.eee = value_param_pattern [symbolic]
// CHECK:STDOUT:   %pair.patt.940: %pattern_type.eee = at_binding_pattern pair, %pair.param_patt.185 [symbolic]
// CHECK:STDOUT:   %.822: Core.Form = init_form %U [symbolic]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %return.param_patt.6dd: %pattern_type.946 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4dd: %pattern_type.946 = return_slot_pattern %return.param_patt.6dd, %U [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.19a: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.b04: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.af1: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %pair.param_patt.f67: %pattern_type.c66 = value_param_pattern [concrete]
// CHECK:STDOUT:   %pair.patt.a61: %pattern_type.c66 = at_binding_pattern pair, %pair.param_patt.f67 [concrete]
// CHECK:STDOUT:   %.592: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.eae: %pattern_type.d8d = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e38: %pattern_type.d8d = return_slot_pattern %return.param_patt.eae, %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.fca: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.6db: <witness> = complete_type_witness %tuple.type.af1 [concrete]
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
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc7_17.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc7_17.2 (constants.%U.patt)]
// CHECK:STDOUT:     %pair.param_patt.loc7_30.1: @F.%pattern_type.loc7_30 (%pattern_type.eee) = value_param_pattern [symbolic = %pair.param_patt.loc7_30.2 (constants.%pair.param_patt.185)]
// CHECK:STDOUT:     %pair.patt.loc7_30.1: @F.%pattern_type.loc7_30 (%pattern_type.eee) = at_binding_pattern pair, %pair.param_patt.loc7_30.1 [symbolic = %pair.patt.loc7_30.2 (constants.%pair.patt.940)]
// CHECK:STDOUT:     %return.param_patt.loc7_43.1: @F.%pattern_type.loc7_43 (%pattern_type.946) = out_param_pattern [symbolic = %return.param_patt.loc7_43.2 (constants.%return.param_patt.6dd)]
// CHECK:STDOUT:     %return.patt.loc7_40.1: @F.%pattern_type.loc7_43 (%pattern_type.946) = return_slot_pattern %return.param_patt.loc7_43.1, %U.ref.loc7_43 [symbolic = %return.patt.loc7_40.2 (constants.%return.patt.4dd)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_43: type = name_ref U, %U.loc7_17.2 [symbolic = %U.loc7_17.1 (constants.%U)]
// CHECK:STDOUT:     %.loc7_43.3: Core.Form = init_form %U.ref.loc7_43 [symbolic = %.loc7_43.2 (constants.%.822)]
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc7_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_20.1: type = splice_block %.loc7_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc7_17: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_17.2: type = symbolic_binding U, 1 [symbolic = %U.loc7_17.1 (constants.%U)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.a5e) = value_param call_param0
// CHECK:STDOUT:     %.loc7_37.1: type = splice_block %.loc7_37.3 [symbolic = %tuple.type (constants.%tuple.type.a5e)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc7_36: type = name_ref U, %U.loc7_17.2 [symbolic = %U.loc7_17.1 (constants.%U)]
// CHECK:STDOUT:       %.loc7_37.2: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc7_36) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:       %.loc7_37.3: type = converted %.loc7_37.2, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.a5e) = wrapper_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc7_17.1 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc7_17.1 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.c66 = value_param_pattern [concrete = constants.%pair.param_patt.f67]
// CHECK:STDOUT:     %pair.patt: %pattern_type.c66 = at_binding_pattern pair, %pair.param_patt [concrete = constants.%pair.patt.a61]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete = constants.%return.param_patt.eae]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %D.ref.loc9_23 [concrete = constants.%return.patt.e38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_23.2: Core.Form = init_form %D.ref.loc9_23 [concrete = constants.%.592]
// CHECK:STDOUT:     %pair.param: %tuple.type.af1 = value_param call_param0
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type.af1] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc9_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc9_16) [concrete = constants.%tuple.b04]
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type.af1 [concrete = constants.%tuple.type.af1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.af1 = wrapper_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_7.2: type, %U.loc7_17.2: type) {
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc7_17.2: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc7_17.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc7_17.1: type = symbolic_binding U, 1 [symbolic = %U.loc7_17.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc7_7.1, %U.loc7_17.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc7_7.1, %U.loc7_17.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %pattern_type.loc7_30: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_30 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %pair.param_patt.loc7_30.2: @F.%pattern_type.loc7_30 (%pattern_type.eee) = value_param_pattern [symbolic = %pair.param_patt.loc7_30.2 (constants.%pair.param_patt.185)]
// CHECK:STDOUT:   %pair.patt.loc7_30.2: @F.%pattern_type.loc7_30 (%pattern_type.eee) = at_binding_pattern pair, %pair.param_patt.loc7_30.2 [symbolic = %pair.patt.loc7_30.2 (constants.%pair.patt.940)]
// CHECK:STDOUT:   %.loc7_43.2: Core.Form = init_form %U.loc7_17.1 [symbolic = %.loc7_43.2 (constants.%.822)]
// CHECK:STDOUT:   %pattern_type.loc7_43: type = pattern_type %U.loc7_17.1 [symbolic = %pattern_type.loc7_43 (constants.%pattern_type.946)]
// CHECK:STDOUT:   %return.param_patt.loc7_43.2: @F.%pattern_type.loc7_43 (%pattern_type.946) = out_param_pattern [symbolic = %return.param_patt.loc7_43.2 (constants.%return.param_patt.6dd)]
// CHECK:STDOUT:   %return.patt.loc7_40.2: @F.%pattern_type.loc7_43 (%pattern_type.946) = return_slot_pattern %return.param_patt.loc7_43.2, %U.loc7_17.1 [symbolic = %return.patt.loc7_40.2 (constants.%return.patt.4dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1, %U.loc7_17.1) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.19a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.a5e)) -> out %return.param: @F.%U.loc7_17.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %pair.ref: @F.%tuple.type (%tuple.type.a5e) = name_ref pair, %pair
// CHECK:STDOUT:     %F.specific_fn.loc7_54.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.19a)]
// CHECK:STDOUT:     %.loc7_43.1: ref @F.%U.loc7_17.1 (%U) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%U.loc7_17.1 (%U) to %.loc7_43.1 = call %F.specific_fn.loc7_54.1(%pair.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.af1) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.af1 = name_ref pair, %pair
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.fca]
// CHECK:STDOUT:   %.loc9_23.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %D to %.loc9_23.1 = call %F.specific_fn(%pair.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc7_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc7_17.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.eee
// CHECK:STDOUT:   %pair.param_patt.loc7_30.2 => constants.%pair.param_patt.185
// CHECK:STDOUT:   %pair.patt.loc7_30.2 => constants.%pair.patt.940
// CHECK:STDOUT:   %.loc7_43.2 => constants.%.822
// CHECK:STDOUT:   %pattern_type.loc7_43 => constants.%pattern_type.946
// CHECK:STDOUT:   %return.param_patt.loc7_43.2 => constants.%return.param_patt.6dd
// CHECK:STDOUT:   %return.patt.loc7_40.2 => constants.%return.patt.4dd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.19a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%C
// CHECK:STDOUT:   %U.patt.loc7_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc7_17.1 => constants.%D
// CHECK:STDOUT:   %tuple => constants.%tuple.b04
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.af1
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.c66
// CHECK:STDOUT:   %pair.param_patt.loc7_30.2 => constants.%pair.param_patt.f67
// CHECK:STDOUT:   %pair.patt.loc7_30.2 => constants.%pair.patt.a61
// CHECK:STDOUT:   %.loc7_43.2 => constants.%.592
// CHECK:STDOUT:   %pattern_type.loc7_43 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %return.param_patt.loc7_43.2 => constants.%return.param_patt.eae
// CHECK:STDOUT:   %return.patt.loc7_40.2 => constants.%return.patt.e38
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.6db
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.fca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.b59: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %Pair.patt: %pattern_type.394 = symbolic_binding_pattern Pair, 0 [symbolic]
// CHECK:STDOUT:   %Pair: %tuple.type.e55 = symbolic_binding Pair, 0 [symbolic]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [concrete]
// CHECK:STDOUT:   %HasPair.generic: %HasPair.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasPair.c64: type = class_type @HasPair, @HasPair(%Pair) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A.patt: %pattern_type.6b6 = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %A: %i32 = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: %pattern_type.6b6 = symbolic_binding_pattern B, 1 [symbolic]
// CHECK:STDOUT:   %B: %i32 = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %tuple.509: %tuple.type.e55 = tuple_value (%A, %B) [symbolic]
// CHECK:STDOUT:   %HasPair.72f: type = class_type @HasPair, @HasPair(%tuple.509) [symbolic]
// CHECK:STDOUT:   %pattern_type.725: type = pattern_type %HasPair.72f [symbolic]
// CHECK:STDOUT:   %h.param_patt.cdf: %pattern_type.725 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %h.patt.596: %pattern_type.725 = at_binding_pattern h, %h.param_patt.cdf [symbolic]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.39d: <witness> = require_complete_type %HasPair.72f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.273: <bound method> = bound_method %B, %Int.as.Copy.impl.Op.4f6 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.523: <bound method> = bound_method %B, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.102: %tuple.type.e55 = tuple_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %HasPair.69b: type = class_type @HasPair, @HasPair(%tuple.102) [concrete]
// CHECK:STDOUT:   %pattern_type.adf: type = pattern_type %HasPair.69b [concrete]
// CHECK:STDOUT:   %h.param_patt.7d9: %pattern_type.adf = value_param_pattern [concrete]
// CHECK:STDOUT:   %h.patt.25e585.1: %pattern_type.adf = at_binding_pattern h, %h.param_patt.7d9 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %h.patt.25e585.2: %pattern_type.adf = at_binding_pattern h, %h.param_patt.7d9 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.245: <bound method> = bound_method %int_2.295, %Int.as.Copy.impl.Op.4f6 [concrete]
// CHECK:STDOUT:   %bound_method.03a: <bound method> = bound_method %int_2.295, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %Pair.patt.loc4_19.1: %pattern_type.394 = symbolic_binding_pattern Pair, 0 [symbolic = %Pair.patt.loc4_19.2 (constants.%Pair.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_31.1: type = splice_block %.loc4_31.3 [concrete = constants.%tuple.type.e55] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc4_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc4_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_31.2: %tuple.type.24b = tuple_literal (%i32.loc4_23, %i32.loc4_28) [concrete = constants.%tuple.b59]
// CHECK:STDOUT:       %.loc4_31.3: type = converted %.loc4_31.2, constants.%tuple.type.e55 [concrete = constants.%tuple.type.e55]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Pair.loc4_19.2: %tuple.type.e55 = symbolic_binding Pair, 0 [symbolic = %Pair.loc4_19.1 (constants.%Pair)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt.loc6_7.1: %pattern_type.6b6 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc6_7.2 (constants.%A.patt)]
// CHECK:STDOUT:     %B.patt.loc6_16.1: %pattern_type.6b6 = symbolic_binding_pattern B, 1 [symbolic = %B.patt.loc6_16.2 (constants.%B.patt)]
// CHECK:STDOUT:     %h.param_patt.loc6_32.1: @F.%pattern_type (%pattern_type.725) = value_param_pattern [symbolic = %h.param_patt.loc6_32.2 (constants.%h.param_patt.cdf)]
// CHECK:STDOUT:     %h.patt.loc6_32.1: @F.%pattern_type (%pattern_type.725) = at_binding_pattern h, %h.param_patt.loc6_32.1 [symbolic = %h.patt.loc6_32.2 (constants.%h.patt.596)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc6_54 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_54: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_54: Core.Form = init_form %i32.loc6_54 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.loc6_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc6_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc6_7.2: %i32 = symbolic_binding A, 0 [symbolic = %A.loc6_7.1 (constants.%A)]
// CHECK:STDOUT:     %.loc6_19: type = splice_block %i32.loc6_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.loc6_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc6_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_16.2: %i32 = symbolic_binding B, 1 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc6_48.1 (%HasPair.72f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_48.1: type = splice_block %HasPair.loc6_48.2 [symbolic = %HasPair.loc6_48.1 (constants.%HasPair.72f)] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %A.ref: %i32 = name_ref A, %A.loc6_7.2 [symbolic = %A.loc6_7.1 (constants.%A)]
// CHECK:STDOUT:       %B.ref.loc6_46: %i32 = name_ref B, %B.loc6_16.2 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:       %.loc6_47: %tuple.type.e55 = tuple_literal (%A.ref, %B.ref.loc6_46) [symbolic = %tuple.loc6_47.1 (constants.%tuple.509)]
// CHECK:STDOUT:       %tuple.loc6_47.2: %tuple.type.e55 = tuple_value (%A.ref, %B.ref.loc6_46) [symbolic = %tuple.loc6_47.1 (constants.%tuple.509)]
// CHECK:STDOUT:       %.loc6_48.2: %tuple.type.e55 = converted %.loc6_47, %tuple.loc6_47.2 [symbolic = %tuple.loc6_47.1 (constants.%tuple.509)]
// CHECK:STDOUT:       %HasPair.loc6_48.2: type = class_type @HasPair, @HasPair(constants.%tuple.509) [symbolic = %HasPair.loc6_48.1 (constants.%HasPair.72f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: @F.%HasPair.loc6_48.1 (%HasPair.72f) = wrapper_binding h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %h.param_patt: %pattern_type.adf = value_param_pattern [concrete = constants.%h.param_patt.7d9]
// CHECK:STDOUT:     %h.patt: %pattern_type.adf = at_binding_pattern h, %h.param_patt [concrete = constants.%h.patt.25e585.1]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_29: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %h.param: %HasPair.69b = value_param call_param0
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %HasPair [concrete = constants.%HasPair.69b] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:       %.loc8_22.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %impl.elem0.loc8_22.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:       %bound_method.loc8_22.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_22.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:       %specific_fn.loc8_22.1: <specific function> = specific_function %impl.elem0.loc8_22.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.2: <bound method> = bound_method %int_1, %specific_fn.loc8_22.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.1: init %i32 = call %bound_method.loc8_22.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:       %.loc8_22.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:       %.loc8_22.3: %i32 = converted %int_1, %.loc8_22.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:       %impl.elem0.loc8_22.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:       %bound_method.loc8_22.3: <bound method> = bound_method %int_2, %impl.elem0.loc8_22.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:       %specific_fn.loc8_22.2: <specific function> = specific_function %impl.elem0.loc8_22.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.4: <bound method> = bound_method %int_2, %specific_fn.loc8_22.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.2: init %i32 = call %bound_method.loc8_22.4(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:       %.loc8_22.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:       %.loc8_22.5: %i32 = converted %int_2, %.loc8_22.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:       %tuple: %tuple.type.e55 = tuple_value (%.loc8_22.3, %.loc8_22.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:       %.loc8_23.2: %tuple.type.e55 = converted %.loc8_22.1, %tuple [concrete = constants.%tuple.102]
// CHECK:STDOUT:       %HasPair: type = class_type @HasPair, @HasPair(constants.%tuple.102) [concrete = constants.%HasPair.69b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: %HasPair.69b = wrapper_binding h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HasPair(%Pair.loc4_19.2: %tuple.type.e55) {
// CHECK:STDOUT:   %Pair.patt.loc4_19.2: %pattern_type.394 = symbolic_binding_pattern Pair, 0 [symbolic = %Pair.patt.loc4_19.2 (constants.%Pair.patt)]
// CHECK:STDOUT:   %Pair.loc4_19.1: %tuple.type.e55 = symbolic_binding Pair, 0 [symbolic = %Pair.loc4_19.1 (constants.%Pair)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HasPair.c64
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc6_7.2: %i32, %B.loc6_16.2: %i32) {
// CHECK:STDOUT:   %A.patt.loc6_7.2: %pattern_type.6b6 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc6_7.2 (constants.%A.patt)]
// CHECK:STDOUT:   %A.loc6_7.1: %i32 = symbolic_binding A, 0 [symbolic = %A.loc6_7.1 (constants.%A)]
// CHECK:STDOUT:   %B.patt.loc6_16.2: %pattern_type.6b6 = symbolic_binding_pattern B, 1 [symbolic = %B.patt.loc6_16.2 (constants.%B.patt)]
// CHECK:STDOUT:   %B.loc6_16.1: %i32 = symbolic_binding B, 1 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:   %tuple.loc6_47.1: %tuple.type.e55 = tuple_value (%A.loc6_7.1, %B.loc6_16.1) [symbolic = %tuple.loc6_47.1 (constants.%tuple.509)]
// CHECK:STDOUT:   %HasPair.loc6_48.1: type = class_type @HasPair, @HasPair(%tuple.loc6_47.1) [symbolic = %HasPair.loc6_48.1 (constants.%HasPair.72f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HasPair.loc6_48.1 [symbolic = %pattern_type (constants.%pattern_type.725)]
// CHECK:STDOUT:   %h.param_patt.loc6_32.2: @F.%pattern_type (%pattern_type.725) = value_param_pattern [symbolic = %h.param_patt.loc6_32.2 (constants.%h.param_patt.cdf)]
// CHECK:STDOUT:   %h.patt.loc6_32.2: @F.%pattern_type (%pattern_type.725) = at_binding_pattern h, %h.param_patt.loc6_32.2 [symbolic = %h.patt.loc6_32.2 (constants.%h.patt.596)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasPair.loc6_48.1 [symbolic = %require_complete (constants.%require_complete.39d)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %B.loc6_16.1, constants.%Int.as.Copy.impl.Op.4f6 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.273)]
// CHECK:STDOUT:   %bound_method.loc6_67.3: <bound method> = bound_method %B.loc6_16.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_67.3 (constants.%bound_method.523)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%h.param: @F.%HasPair.loc6_48.1 (%HasPair.72f)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %B.ref.loc6_67: %i32 = name_ref B, %B.loc6_16.2 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:     %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:     %bound_method.loc6_67.1: <bound method> = bound_method %B.ref.loc6_67, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.273)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_67.2: <bound method> = bound_method %B.ref.loc6_67, %specific_fn [symbolic = %bound_method.loc6_67.3 (constants.%bound_method.523)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_67.2(%B.ref.loc6_67) [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param: %HasPair.69b) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %h.ref: %HasPair.69b = name_ref h, %h
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_1.0c6, constants.%int_2.295) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%h.ref)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%Pair) {
// CHECK:STDOUT:   %Pair.patt.loc4_19.2 => constants.%Pair.patt
// CHECK:STDOUT:   %Pair.loc4_19.1 => constants.%Pair
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.509) {
// CHECK:STDOUT:   %Pair.patt.loc4_19.2 => constants.%Pair.patt
// CHECK:STDOUT:   %Pair.loc4_19.1 => constants.%tuple.509
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.patt.loc6_7.2 => constants.%A.patt
// CHECK:STDOUT:   %A.loc6_7.1 => constants.%A
// CHECK:STDOUT:   %B.patt.loc6_16.2 => constants.%B.patt
// CHECK:STDOUT:   %B.loc6_16.1 => constants.%B
// CHECK:STDOUT:   %tuple.loc6_47.1 => constants.%tuple.509
// CHECK:STDOUT:   %HasPair.loc6_48.1 => constants.%HasPair.72f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.725
// CHECK:STDOUT:   %h.param_patt.loc6_32.2 => constants.%h.param_patt.cdf
// CHECK:STDOUT:   %h.patt.loc6_32.2 => constants.%h.patt.596
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.102) {
// CHECK:STDOUT:   %Pair.patt.loc4_19.2 => constants.%Pair.patt
// CHECK:STDOUT:   %Pair.loc4_19.1 => constants.%tuple.102
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_1.0c6, constants.%int_2.295) {
// CHECK:STDOUT:   %A.patt.loc6_7.2 => constants.%A.patt
// CHECK:STDOUT:   %A.loc6_7.1 => constants.%int_1.0c6
// CHECK:STDOUT:   %B.patt.loc6_16.2 => constants.%B.patt
// CHECK:STDOUT:   %B.loc6_16.1 => constants.%int_2.295
// CHECK:STDOUT:   %tuple.loc6_47.1 => constants.%tuple.102
// CHECK:STDOUT:   %HasPair.loc6_48.1 => constants.%HasPair.69b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.adf
// CHECK:STDOUT:   %h.param_patt.loc6_32.2 => constants.%h.param_patt.7d9
// CHECK:STDOUT:   %h.patt.loc6_32.2 => constants.%h.patt.25e585.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.245
// CHECK:STDOUT:   %bound_method.loc6_67.3 => constants.%bound_method.03a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.11e: %tuple.type.24b = tuple_value (%T, %T) [symbolic]
// CHECK:STDOUT:   %tuple.type.07a: type = tuple_type (%T, %T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %tuple.type.07a [symbolic]
// CHECK:STDOUT:   %pair.param_patt.5a3: %pattern_type.c3f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %pair.patt.f0f: %pattern_type.c3f = at_binding_pattern pair, %pair.param_patt.5a3 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.b04: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.af1: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %pair.param_patt.f67: %pattern_type.c66 = value_param_pattern [concrete]
// CHECK:STDOUT:   %pair.patt.a61: %pattern_type.c66 = at_binding_pattern pair, %pair.param_patt.f67 [concrete]
// CHECK:STDOUT:   %.592: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.eae: %pattern_type.d8d = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e38: %pattern_type.d8d = return_slot_pattern %return.param_patt.eae, %D [concrete]
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
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %pair.param_patt.loc7_20.1: @F.%pattern_type.loc7_20 (%pattern_type.c3f) = value_param_pattern [symbolic = %pair.param_patt.loc7_20.2 (constants.%pair.param_patt.5a3)]
// CHECK:STDOUT:     %pair.patt.loc7_20.1: @F.%pattern_type.loc7_20 (%pattern_type.c3f) = at_binding_pattern pair, %pair.param_patt.loc7_20.1 [symbolic = %pair.patt.loc7_20.2 (constants.%pair.patt.f0f)]
// CHECK:STDOUT:     %return.param_patt.loc7_33.1: @F.%pattern_type.loc7_33 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc7_33.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:     %return.patt.loc7_30.1: @F.%pattern_type.loc7_33 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_33.1, %T.ref.loc7_33 [symbolic = %return.patt.loc7_30.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_33: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_33.2: Core.Form = init_form %T.ref.loc7_33 [symbolic = %.loc7_33.1 (constants.%.184)]
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.07a) = value_param call_param0
// CHECK:STDOUT:     %.loc7_27.1: type = splice_block %.loc7_27.3 [symbolic = %tuple.type (constants.%tuple.type.07a)] {
// CHECK:STDOUT:       %T.ref.loc7_23: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc7_26: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_27.2: %tuple.type.24b = tuple_literal (%T.ref.loc7_23, %T.ref.loc7_26) [symbolic = %tuple (constants.%tuple.11e)]
// CHECK:STDOUT:       %.loc7_27.3: type = converted %.loc7_27.2, constants.%tuple.type.07a [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.07a) = wrapper_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.c66 = value_param_pattern [concrete = constants.%pair.param_patt.f67]
// CHECK:STDOUT:     %pair.patt: %pattern_type.c66 = at_binding_pattern pair, %pair.param_patt [concrete = constants.%pair.patt.a61]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete = constants.%return.param_patt.eae]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %D.ref.loc9_23 [concrete = constants.%return.patt.e38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_23: Core.Form = init_form %D.ref.loc9_23 [concrete = constants.%.592]
// CHECK:STDOUT:     %pair.param: %tuple.type.af1 = value_param call_param0
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type.af1] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc9_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc9_16) [concrete = constants.%tuple.b04]
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type.af1 [concrete = constants.%tuple.type.af1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.af1 = wrapper_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc7_7.1, %T.loc7_7.1) [symbolic = %tuple (constants.%tuple.11e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc7_7.1, %T.loc7_7.1) [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:   %pattern_type.loc7_20: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_20 (constants.%pattern_type.c3f)]
// CHECK:STDOUT:   %pair.param_patt.loc7_20.2: @F.%pattern_type.loc7_20 (%pattern_type.c3f) = value_param_pattern [symbolic = %pair.param_patt.loc7_20.2 (constants.%pair.param_patt.5a3)]
// CHECK:STDOUT:   %pair.patt.loc7_20.2: @F.%pattern_type.loc7_20 (%pattern_type.c3f) = at_binding_pattern pair, %pair.param_patt.loc7_20.2 [symbolic = %pair.patt.loc7_20.2 (constants.%pair.patt.f0f)]
// CHECK:STDOUT:   %.loc7_33.1: Core.Form = init_form %T.loc7_7.1 [symbolic = %.loc7_33.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_33: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7_33 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc7_33.2: @F.%pattern_type.loc7_33 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc7_33.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc7_30.2: @F.%pattern_type.loc7_33 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_33.2, %T.loc7_7.1 [symbolic = %return.patt.loc7_30.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.07a)) -> out %return.param: @F.%T.loc7_7.1 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.af1) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.af1 = name_ref pair, %pair
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %tuple => constants.%tuple.11e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.07a
// CHECK:STDOUT:   %pattern_type.loc7_20 => constants.%pattern_type.c3f
// CHECK:STDOUT:   %pair.param_patt.loc7_20.2 => constants.%pair.param_patt.5a3
// CHECK:STDOUT:   %pair.patt.loc7_20.2 => constants.%pair.patt.f0f
// CHECK:STDOUT:   %.loc7_33.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_33 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc7_33.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc7_30.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:

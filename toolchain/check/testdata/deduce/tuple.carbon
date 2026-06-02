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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %.822: Core.Form = init_form %U [symbolic]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.367: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.ac7: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.a6a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.ce9: type = pattern_type %tuple.type.a6a [concrete]
// CHECK:STDOUT:   %.f19: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.f1a: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.335: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.3cb: <witness> = complete_type_witness %tuple.type.a6a [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc7_30 (%pattern_type.eee) = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc7_30 (%pattern_type.eee) = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_43 (%pattern_type.946) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_43 (%pattern_type.946) = return_slot_pattern %return.param_patt, %U.ref.loc7_43 [concrete]
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
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.a5e) = value_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc7_17.1 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc7_17.1 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.ce9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: %pattern_type.ce9 = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f1a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f1a = return_slot_pattern %return.param_patt, %D.ref.loc9_23 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_23.2: Core.Form = init_form %D.ref.loc9_23 [concrete = constants.%.f19]
// CHECK:STDOUT:     %pair.param: %tuple.type.a6a = value_param call_param0
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type.a6a] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc9_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc9_16) [concrete = constants.%tuple.ac7]
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type.a6a [concrete = constants.%tuple.type.a6a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.a6a = value_binding pair, %pair.param
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
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_17.1: type = symbolic_binding U, 1 [symbolic = %U.loc7_17.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc7_7.1, %U.loc7_17.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc7_7.1, %U.loc7_17.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %pattern_type.loc7_30: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_30 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %.loc7_43.2: Core.Form = init_form %U.loc7_17.1 [symbolic = %.loc7_43.2 (constants.%.822)]
// CHECK:STDOUT:   %pattern_type.loc7_43: type = pattern_type %U.loc7_17.1 [symbolic = %pattern_type.loc7_43 (constants.%pattern_type.946)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1, %U.loc7_17.1) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.367)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.a5e)) -> out %return.param: @F.%U.loc7_17.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %pair.ref: @F.%tuple.type (%tuple.type.a5e) = name_ref pair, %pair
// CHECK:STDOUT:     %F.specific_fn.loc7_54.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.367)]
// CHECK:STDOUT:     %.loc7_43.1: ref @F.%U.loc7_17.1 (%U) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%U.loc7_17.1 (%U) to %.loc7_43.1 = call %F.specific_fn.loc7_54.1(%pair.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.a6a) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.a6a = name_ref pair, %pair
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.335]
// CHECK:STDOUT:   %.loc9_23.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %D to %.loc9_23.1 = call %F.specific_fn(%pair.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_17.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.eee
// CHECK:STDOUT:   %.loc7_43.2 => constants.%.822
// CHECK:STDOUT:   %pattern_type.loc7_43 => constants.%pattern_type.946
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.367
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%C
// CHECK:STDOUT:   %U.loc7_17.1 => constants.%D
// CHECK:STDOUT:   %tuple => constants.%tuple.ac7
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a6a
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.ce9
// CHECK:STDOUT:   %.loc7_43.2 => constants.%.f19
// CHECK:STDOUT:   %pattern_type.loc7_43 => constants.%pattern_type.f1a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3cb
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.335
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
// CHECK:STDOUT:   %tuple.79d: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e5a: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.7d6: type = pattern_type %tuple.type.e5a [concrete]
// CHECK:STDOUT:   %Pair: %tuple.type.e5a = symbolic_binding Pair, 0 [symbolic]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [concrete]
// CHECK:STDOUT:   %HasPair.generic: %HasPair.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasPair.38e: type = class_type @HasPair, @HasPair(%Pair) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A: %i32 = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %B: %i32 = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %tuple.20c: %tuple.type.e5a = tuple_value (%A, %B) [symbolic]
// CHECK:STDOUT:   %HasPair.fc9: type = class_type @HasPair, @HasPair(%tuple.20c) [symbolic]
// CHECK:STDOUT:   %pattern_type.684: type = pattern_type %HasPair.fc9 [symbolic]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.be4: <witness> = require_complete_type %HasPair.fc9 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.f26: <bound method> = bound_method %B, %Int.as.Copy.impl.Op.ce7 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b25: <bound method> = bound_method %B, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.cd5: %tuple.type.e5a = tuple_value (%int_1.d9c, %int_2.228) [concrete]
// CHECK:STDOUT:   %HasPair.387: type = class_type @HasPair, @HasPair(%tuple.cd5) [concrete]
// CHECK:STDOUT:   %pattern_type.36d: type = pattern_type %HasPair.387 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_1.d9c, %int_2.228) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.a01: <bound method> = bound_method %int_2.228, %Int.as.Copy.impl.Op.ce7 [concrete]
// CHECK:STDOUT:   %bound_method.190: <bound method> = bound_method %int_2.228, %Int.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %Pair.patt: %pattern_type.7d6 = symbolic_binding_pattern Pair, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_31.1: type = splice_block %.loc4_31.3 [concrete = constants.%tuple.type.e5a] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc4_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc4_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_31.2: %tuple.type.24b = tuple_literal (%i32.loc4_23, %i32.loc4_28) [concrete = constants.%tuple.79d]
// CHECK:STDOUT:       %.loc4_31.3: type = converted %.loc4_31.2, constants.%tuple.type.e5a [concrete = constants.%tuple.type.e5a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Pair.loc4_19.2: %tuple.type.e5a = symbolic_binding Pair, 0 [symbolic = %Pair.loc4_19.1 (constants.%Pair)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.104 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.104 = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:     %h.param_patt: @F.%pattern_type (%pattern_type.684) = value_param_pattern [concrete]
// CHECK:STDOUT:     %h.patt: @F.%pattern_type (%pattern_type.684) = at_binding_pattern h, %h.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32.loc6_54 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_54: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_54: Core.Form = init_form %i32.loc6_54 [concrete = constants.%.7ea]
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
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc6_48.1 (%HasPair.fc9) = value_param call_param0
// CHECK:STDOUT:     %.loc6_48.1: type = splice_block %HasPair.loc6_48.2 [symbolic = %HasPair.loc6_48.1 (constants.%HasPair.fc9)] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %A.ref: %i32 = name_ref A, %A.loc6_7.2 [symbolic = %A.loc6_7.1 (constants.%A)]
// CHECK:STDOUT:       %B.ref.loc6_46: %i32 = name_ref B, %B.loc6_16.2 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:       %.loc6_47: %tuple.type.e5a = tuple_literal (%A.ref, %B.ref.loc6_46) [symbolic = %tuple.loc6_47.1 (constants.%tuple.20c)]
// CHECK:STDOUT:       %tuple.loc6_47.2: %tuple.type.e5a = tuple_value (%A.ref, %B.ref.loc6_46) [symbolic = %tuple.loc6_47.1 (constants.%tuple.20c)]
// CHECK:STDOUT:       %.loc6_48.2: %tuple.type.e5a = converted %.loc6_47, %tuple.loc6_47.2 [symbolic = %tuple.loc6_47.1 (constants.%tuple.20c)]
// CHECK:STDOUT:       %HasPair.loc6_48.2: type = class_type @HasPair, @HasPair(constants.%tuple.20c) [symbolic = %HasPair.loc6_48.1 (constants.%HasPair.fc9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: @F.%HasPair.loc6_48.1 (%HasPair.fc9) = value_binding h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %h.param_patt: %pattern_type.36d = value_param_pattern [concrete]
// CHECK:STDOUT:     %h.patt: %pattern_type.36d = at_binding_pattern h, %h.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_29: Core.Form = init_form %i32 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %h.param: %HasPair.387 = value_param call_param0
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %HasPair [concrete = constants.%HasPair.387] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:       %.loc8_22.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %impl.elem0.loc8_22.1: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:       %bound_method.loc8_22.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_22.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:       %specific_fn.loc8_22.1: <specific function> = specific_function %impl.elem0.loc8_22.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.2: <bound method> = bound_method %int_1, %specific_fn.loc8_22.1 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.1: init %i32 = call %bound_method.loc8_22.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:       %.loc8_22.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:       %.loc8_22.3: %i32 = converted %int_1, %.loc8_22.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:       %impl.elem0.loc8_22.2: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:       %bound_method.loc8_22.3: <bound method> = bound_method %int_2, %impl.elem0.loc8_22.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:       %specific_fn.loc8_22.2: <specific function> = specific_function %impl.elem0.loc8_22.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.4: <bound method> = bound_method %int_2, %specific_fn.loc8_22.2 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.2: init %i32 = call %bound_method.loc8_22.4(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:       %.loc8_22.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:       %.loc8_22.5: %i32 = converted %int_2, %.loc8_22.4 [concrete = constants.%int_2.228]
// CHECK:STDOUT:       %tuple: %tuple.type.e5a = tuple_value (%.loc8_22.3, %.loc8_22.5) [concrete = constants.%tuple.cd5]
// CHECK:STDOUT:       %.loc8_23.2: %tuple.type.e5a = converted %.loc8_22.1, %tuple [concrete = constants.%tuple.cd5]
// CHECK:STDOUT:       %HasPair: type = class_type @HasPair, @HasPair(constants.%tuple.cd5) [concrete = constants.%HasPair.387]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: %HasPair.387 = value_binding h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HasPair(%Pair.loc4_19.2: %tuple.type.e5a) {
// CHECK:STDOUT:   %Pair.loc4_19.1: %tuple.type.e5a = symbolic_binding Pair, 0 [symbolic = %Pair.loc4_19.1 (constants.%Pair)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HasPair.38e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc6_7.2: %i32, %B.loc6_16.2: %i32) {
// CHECK:STDOUT:   %A.loc6_7.1: %i32 = symbolic_binding A, 0 [symbolic = %A.loc6_7.1 (constants.%A)]
// CHECK:STDOUT:   %B.loc6_16.1: %i32 = symbolic_binding B, 1 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:   %tuple.loc6_47.1: %tuple.type.e5a = tuple_value (%A.loc6_7.1, %B.loc6_16.1) [symbolic = %tuple.loc6_47.1 (constants.%tuple.20c)]
// CHECK:STDOUT:   %HasPair.loc6_48.1: type = class_type @HasPair, @HasPair(%tuple.loc6_47.1) [symbolic = %HasPair.loc6_48.1 (constants.%HasPair.fc9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HasPair.loc6_48.1 [symbolic = %pattern_type (constants.%pattern_type.684)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasPair.loc6_48.1 [symbolic = %require_complete (constants.%require_complete.be4)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %B.loc6_16.1, constants.%Int.as.Copy.impl.Op.ce7 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.f26)]
// CHECK:STDOUT:   %bound_method.loc6_67.3: <bound method> = bound_method %B.loc6_16.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_67.3 (constants.%bound_method.b25)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%h.param: @F.%HasPair.loc6_48.1 (%HasPair.fc9)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %B.ref.loc6_67: %i32 = name_ref B, %B.loc6_16.2 [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:     %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:     %bound_method.loc6_67.1: <bound method> = bound_method %B.ref.loc6_67, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.f26)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_67.2: <bound method> = bound_method %B.ref.loc6_67, %specific_fn [symbolic = %bound_method.loc6_67.3 (constants.%bound_method.b25)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_67.2(%B.ref.loc6_67) [symbolic = %B.loc6_16.1 (constants.%B)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param: %HasPair.387) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %h.ref: %HasPair.387 = name_ref h, %h
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_1.d9c, constants.%int_2.228) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%h.ref)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%Pair) {
// CHECK:STDOUT:   %Pair.loc4_19.1 => constants.%Pair
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.20c) {
// CHECK:STDOUT:   %Pair.loc4_19.1 => constants.%tuple.20c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc6_7.1 => constants.%A
// CHECK:STDOUT:   %B.loc6_16.1 => constants.%B
// CHECK:STDOUT:   %tuple.loc6_47.1 => constants.%tuple.20c
// CHECK:STDOUT:   %HasPair.loc6_48.1 => constants.%HasPair.fc9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.684
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.cd5) {
// CHECK:STDOUT:   %Pair.loc4_19.1 => constants.%tuple.cd5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_1.d9c, constants.%int_2.228) {
// CHECK:STDOUT:   %A.loc6_7.1 => constants.%int_1.d9c
// CHECK:STDOUT:   %B.loc6_16.1 => constants.%int_2.228
// CHECK:STDOUT:   %tuple.loc6_47.1 => constants.%tuple.cd5
// CHECK:STDOUT:   %HasPair.loc6_48.1 => constants.%HasPair.387
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.a01
// CHECK:STDOUT:   %bound_method.loc6_67.3 => constants.%bound_method.190
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.11e: %tuple.type.24b = tuple_value (%T, %T) [symbolic]
// CHECK:STDOUT:   %tuple.type.07a: type = tuple_type (%T, %T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %tuple.type.07a [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.ac7: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.a6a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.ce9: type = pattern_type %tuple.type.a6a [concrete]
// CHECK:STDOUT:   %.f19: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.f1a: type = pattern_type %D [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc7_20 (%pattern_type.c3f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc7_20 (%pattern_type.c3f) = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_33 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_33 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc7_33 [concrete]
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
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.07a) = value_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.ce9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: %pattern_type.ce9 = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f1a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f1a = return_slot_pattern %return.param_patt, %D.ref.loc9_23 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_23: Core.Form = init_form %D.ref.loc9_23 [concrete = constants.%.f19]
// CHECK:STDOUT:     %pair.param: %tuple.type.a6a = value_param call_param0
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type.a6a] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc9_16: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc9_16) [concrete = constants.%tuple.ac7]
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type.a6a [concrete = constants.%tuple.type.a6a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.a6a = value_binding pair, %pair.param
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
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc7_7.1, %T.loc7_7.1) [symbolic = %tuple (constants.%tuple.11e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc7_7.1, %T.loc7_7.1) [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:   %pattern_type.loc7_20: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_20 (constants.%pattern_type.c3f)]
// CHECK:STDOUT:   %.loc7_33.1: Core.Form = init_form %T.loc7_7.1 [symbolic = %.loc7_33.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_33: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7_33 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.07a)) -> out %return.param: @F.%T.loc7_7.1 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.a6a) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.a6a = name_ref pair, %pair
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %tuple => constants.%tuple.11e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.07a
// CHECK:STDOUT:   %pattern_type.loc7_20 => constants.%pattern_type.c3f
// CHECK:STDOUT:   %.loc7_33.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_33 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:

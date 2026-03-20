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
// CHECK:STDOUT:   %F.specific_fn.a54: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.a0a: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.281: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.881: type = pattern_type %tuple.type.281 [concrete]
// CHECK:STDOUT:   %.9a5: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.107: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.734: <witness> = complete_type_witness %tuple.type.281 [concrete]
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
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc25 (%pattern_type.eee) = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc25 (%pattern_type.eee) = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc33 (%pattern_type.946) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc33 (%pattern_type.946) = return_slot_pattern %return.param_patt, %U.ref.loc33 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc33: type = name_ref U, %U.loc19.2 [symbolic = %U.loc19.1 (constants.%U)]
// CHECK:STDOUT:     %.loc33.3: Core.Form = init_form %U.ref.loc33 [symbolic = %.loc33.2 (constants.%.822)]
// CHECK:STDOUT:     %.loc17.1: type = splice_block %.loc17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15.2: type = symbolic_binding T, 0 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21.1: type = splice_block %.loc21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc19.2: type = symbolic_binding U, 1 [symbolic = %U.loc19.1 (constants.%U)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.a5e) = value_param call_param0
// CHECK:STDOUT:     %.loc30.1: type = splice_block %.loc30.3 [symbolic = %tuple.type (constants.%tuple.type.a5e)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc15.2 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc29: type = name_ref U, %U.loc19.2 [symbolic = %U.loc19.1 (constants.%U)]
// CHECK:STDOUT:       %.loc30.2: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc29) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:       %.loc30.3: type = converted %.loc30.2, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.a5e) = value_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc19.1 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc19.1 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.881 = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: %pattern_type.881 = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9c8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9c8 = return_slot_pattern %return.param_patt, %D.ref.loc54 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc54: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc54.2: Core.Form = init_form %D.ref.loc54 [concrete = constants.%.9a5]
// CHECK:STDOUT:     %pair.param: %tuple.type.281 = value_param call_param0
// CHECK:STDOUT:     %.loc51.1: type = splice_block %.loc51.3 [concrete = constants.%tuple.type.281] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc50: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc51.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc50) [concrete = constants.%tuple.a0a]
// CHECK:STDOUT:       %.loc51.3: type = converted %.loc51.2, constants.%tuple.type.281 [concrete = constants.%tuple.type.281]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.281 = value_binding pair, %pair.param
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
// CHECK:STDOUT: generic fn @F(%T.loc15.2: type, %U.loc19.2: type) {
// CHECK:STDOUT:   %T.loc15.1: type = symbolic_binding T, 0 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc19.1: type = symbolic_binding U, 1 [symbolic = %U.loc19.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc15.1, %U.loc19.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc15.1, %U.loc19.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %pattern_type.loc25: type = pattern_type %tuple.type [symbolic = %pattern_type.loc25 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %.loc33.2: Core.Form = init_form %U.loc19.1 [symbolic = %.loc33.2 (constants.%.822)]
// CHECK:STDOUT:   %pattern_type.loc33: type = pattern_type %U.loc19.1 [symbolic = %pattern_type.loc33 (constants.%pattern_type.946)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc36.2: <specific function> = specific_function constants.%F, @F(%T.loc15.1, %U.loc19.1) [symbolic = %F.specific_fn.loc36.2 (constants.%F.specific_fn.a54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.a5e)) -> out %return.param: @F.%U.loc19.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %pair.ref: @F.%tuple.type (%tuple.type.a5e) = name_ref pair, %pair
// CHECK:STDOUT:     %F.specific_fn.loc36.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc36.2 (constants.%F.specific_fn.a54)]
// CHECK:STDOUT:     %.loc33.1: ref @F.%U.loc19.1 (%U) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%U.loc19.1 (%U) to %.loc33.1 = call %F.specific_fn.loc36.1(%pair.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.281) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.281 = name_ref pair, %pair
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.107]
// CHECK:STDOUT:   %.loc54.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %D to %.loc54.1 = call %F.specific_fn(%pair.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15.1 => constants.%T
// CHECK:STDOUT:   %U.loc19.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %pattern_type.loc25 => constants.%pattern_type.eee
// CHECK:STDOUT:   %.loc33.2 => constants.%.822
// CHECK:STDOUT:   %pattern_type.loc33 => constants.%pattern_type.946
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc36.2 => constants.%F.specific_fn.a54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc15.1 => constants.%C
// CHECK:STDOUT:   %U.loc19.1 => constants.%D
// CHECK:STDOUT:   %tuple => constants.%tuple.a0a
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.281
// CHECK:STDOUT:   %pattern_type.loc25 => constants.%pattern_type.881
// CHECK:STDOUT:   %.loc33.2 => constants.%.9a5
// CHECK:STDOUT:   %pattern_type.loc33 => constants.%pattern_type.9c8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.734
// CHECK:STDOUT:   %F.specific_fn.loc36.2 => constants.%F.specific_fn.107
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
// CHECK:STDOUT:   %tuple.95a: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %Pair: %tuple.type.d07 = symbolic_binding Pair, 0 [symbolic]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [concrete]
// CHECK:STDOUT:   %HasPair.generic: %HasPair.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasPair.7cc: type = class_type @HasPair, @HasPair(%Pair) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A: %i32 = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %B: %i32 = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %tuple.9c9: %tuple.type.d07 = tuple_value (%A, %B) [symbolic]
// CHECK:STDOUT:   %HasPair.2e7: type = class_type @HasPair, @HasPair(%tuple.9c9) [symbolic]
// CHECK:STDOUT:   %pattern_type.dc2: type = pattern_type %HasPair.2e7 [symbolic]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.76f: <witness> = require_complete_type %HasPair.2e7 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.5f8: <bound method> = bound_method %B, %Int.as.Copy.impl.Op.664 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dfb: <bound method> = bound_method %B, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %HasPair.867: type = class_type @HasPair, @HasPair(%tuple.21c) [concrete]
// CHECK:STDOUT:   %pattern_type.530: type = pattern_type %HasPair.867 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.5e8: <bound method> = bound_method %int_2.ef8, %Int.as.Copy.impl.Op.664 [concrete]
// CHECK:STDOUT:   %bound_method.f15: <bound method> = bound_method %int_2.ef8, %Int.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %Pair.patt: %pattern_type.511 = symbolic_binding_pattern Pair, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc13.2: %tuple.type.24b = tuple_literal (%i32.loc10, %i32.loc12) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:       %.loc13.3: type = converted %.loc13.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Pair.loc7.2: %tuple.type.d07 = symbolic_binding Pair, 0 [symbolic = %Pair.loc7.1 (constants.%Pair)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.7ce = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.7ce = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:     %h.param_patt: @F.%pattern_type (%pattern_type.dc2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %h.patt: @F.%pattern_type (%pattern_type.dc2) = at_binding_pattern h, %h.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc42 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc42: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc42: Core.Form = init_form %i32.loc42 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %.loc22: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc20.2: %i32 = symbolic_binding A, 0 [symbolic = %A.loc20.1 (constants.%A)]
// CHECK:STDOUT:     %.loc26: type = splice_block %i32.loc26 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc24.2: %i32 = symbolic_binding B, 1 [symbolic = %B.loc24.1 (constants.%B)]
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc39.1 (%HasPair.2e7) = value_param call_param0
// CHECK:STDOUT:     %.loc39.1: type = splice_block %HasPair.loc39.2 [symbolic = %HasPair.loc39.1 (constants.%HasPair.2e7)] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %A.ref: %i32 = name_ref A, %A.loc20.2 [symbolic = %A.loc20.1 (constants.%A)]
// CHECK:STDOUT:       %B.ref.loc37: %i32 = name_ref B, %B.loc24.2 [symbolic = %B.loc24.1 (constants.%B)]
// CHECK:STDOUT:       %.loc38: %tuple.type.d07 = tuple_literal (%A.ref, %B.ref.loc37) [symbolic = %tuple.loc38.1 (constants.%tuple.9c9)]
// CHECK:STDOUT:       %tuple.loc38.2: %tuple.type.d07 = tuple_value (%A.ref, %B.ref.loc37) [symbolic = %tuple.loc38.1 (constants.%tuple.9c9)]
// CHECK:STDOUT:       %.loc39.2: %tuple.type.d07 = converted %.loc38, %tuple.loc38.2 [symbolic = %tuple.loc38.1 (constants.%tuple.9c9)]
// CHECK:STDOUT:       %HasPair.loc39.2: type = class_type @HasPair, @HasPair(constants.%tuple.9c9) [symbolic = %HasPair.loc39.1 (constants.%HasPair.2e7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: @F.%HasPair.loc39.1 (%HasPair.2e7) = value_binding h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %h.param_patt: %pattern_type.530 = value_param_pattern [concrete]
// CHECK:STDOUT:     %h.patt: %pattern_type.530 = at_binding_pattern h, %h.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc63: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %h.param: %HasPair.867 = value_param call_param0
// CHECK:STDOUT:     %.loc60.1: type = splice_block %HasPair [concrete = constants.%HasPair.867] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:       %.loc59.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %impl.elem0.loc59.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:       %bound_method.loc59.1: <bound method> = bound_method %int_1, %impl.elem0.loc59.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:       %specific_fn.loc59.1: <specific function> = specific_function %impl.elem0.loc59.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc59.2: <bound method> = bound_method %int_1, %specific_fn.loc59.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc59.1: init %i32 = call %bound_method.loc59.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %.loc59.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc59.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %.loc59.3: %i32 = converted %int_1, %.loc59.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %impl.elem0.loc59.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:       %bound_method.loc59.3: <bound method> = bound_method %int_2, %impl.elem0.loc59.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:       %specific_fn.loc59.2: <specific function> = specific_function %impl.elem0.loc59.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc59.4: <bound method> = bound_method %int_2, %specific_fn.loc59.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc59.2: init %i32 = call %bound_method.loc59.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %.loc59.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc59.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %.loc59.5: %i32 = converted %int_2, %.loc59.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %tuple: %tuple.type.d07 = tuple_value (%.loc59.3, %.loc59.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:       %.loc60.2: %tuple.type.d07 = converted %.loc59.1, %tuple [concrete = constants.%tuple.21c]
// CHECK:STDOUT:       %HasPair: type = class_type @HasPair, @HasPair(constants.%tuple.21c) [concrete = constants.%HasPair.867]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: %HasPair.867 = value_binding h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HasPair(%Pair.loc7.2: %tuple.type.d07) {
// CHECK:STDOUT:   %Pair.loc7.1: %tuple.type.d07 = symbolic_binding Pair, 0 [symbolic = %Pair.loc7.1 (constants.%Pair)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HasPair.7cc
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc20.2: %i32, %B.loc24.2: %i32) {
// CHECK:STDOUT:   %A.loc20.1: %i32 = symbolic_binding A, 0 [symbolic = %A.loc20.1 (constants.%A)]
// CHECK:STDOUT:   %B.loc24.1: %i32 = symbolic_binding B, 1 [symbolic = %B.loc24.1 (constants.%B)]
// CHECK:STDOUT:   %tuple.loc38.1: %tuple.type.d07 = tuple_value (%A.loc20.1, %B.loc24.1) [symbolic = %tuple.loc38.1 (constants.%tuple.9c9)]
// CHECK:STDOUT:   %HasPair.loc39.1: type = class_type @HasPair, @HasPair(%tuple.loc38.1) [symbolic = %HasPair.loc39.1 (constants.%HasPair.2e7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HasPair.loc39.1 [symbolic = %pattern_type (constants.%pattern_type.dc2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasPair.loc39.1 [symbolic = %require_complete (constants.%require_complete.76f)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %B.loc24.1, constants.%Int.as.Copy.impl.Op.664 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.5f8)]
// CHECK:STDOUT:   %bound_method.loc45.3: <bound method> = bound_method %B.loc24.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc45.3 (constants.%bound_method.dfb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%h.param: @F.%HasPair.loc39.1 (%HasPair.2e7)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %B.ref.loc45: %i32 = name_ref B, %B.loc24.2 [symbolic = %B.loc24.1 (constants.%B)]
// CHECK:STDOUT:     %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:     %bound_method.loc45.1: <bound method> = bound_method %B.ref.loc45, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.5f8)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc45.2: <bound method> = bound_method %B.ref.loc45, %specific_fn [symbolic = %bound_method.loc45.3 (constants.%bound_method.dfb)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc45.2(%B.ref.loc45) [symbolic = %B.loc24.1 (constants.%B)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param: %HasPair.867) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %h.ref: %HasPair.867 = name_ref h, %h
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_1.5d2, constants.%int_2.ef8) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%h.ref)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%Pair) {
// CHECK:STDOUT:   %Pair.loc7.1 => constants.%Pair
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.9c9) {
// CHECK:STDOUT:   %Pair.loc7.1 => constants.%tuple.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc20.1 => constants.%A
// CHECK:STDOUT:   %B.loc24.1 => constants.%B
// CHECK:STDOUT:   %tuple.loc38.1 => constants.%tuple.9c9
// CHECK:STDOUT:   %HasPair.loc39.1 => constants.%HasPair.2e7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.21c) {
// CHECK:STDOUT:   %Pair.loc7.1 => constants.%tuple.21c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_1.5d2, constants.%int_2.ef8) {
// CHECK:STDOUT:   %A.loc20.1 => constants.%int_1.5d2
// CHECK:STDOUT:   %B.loc24.1 => constants.%int_2.ef8
// CHECK:STDOUT:   %tuple.loc38.1 => constants.%tuple.21c
// CHECK:STDOUT:   %HasPair.loc39.1 => constants.%HasPair.867
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.530
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.5e8
// CHECK:STDOUT:   %bound_method.loc45.3 => constants.%bound_method.f15
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
// CHECK:STDOUT:   %tuple.a0a: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.281: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.881: type = pattern_type %tuple.type.281 [concrete]
// CHECK:STDOUT:   %.9a5: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %D [concrete]
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
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc21 (%pattern_type.c3f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc21 (%pattern_type.c3f) = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc29 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc29 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc29 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc29: type = name_ref T, %T.loc15.2 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc29.2: Core.Form = init_form %T.ref.loc29 [symbolic = %.loc29.1 (constants.%.184)]
// CHECK:STDOUT:     %.loc17.1: type = splice_block %.loc17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15.2: type = symbolic_binding T, 0 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.07a) = value_param call_param0
// CHECK:STDOUT:     %.loc26.1: type = splice_block %.loc26.3 [symbolic = %tuple.type (constants.%tuple.type.07a)] {
// CHECK:STDOUT:       %T.ref.loc23: type = name_ref T, %T.loc15.2 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc25: type = name_ref T, %T.loc15.2 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:       %.loc26.2: %tuple.type.24b = tuple_literal (%T.ref.loc23, %T.ref.loc25) [symbolic = %tuple (constants.%tuple.11e)]
// CHECK:STDOUT:       %.loc26.3: type = converted %.loc26.2, constants.%tuple.type.07a [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.07a) = value_binding pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc15.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc15.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.881 = value_param_pattern [concrete]
// CHECK:STDOUT:     %pair.patt: %pattern_type.881 = at_binding_pattern pair, %pair.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9c8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9c8 = return_slot_pattern %return.param_patt, %D.ref.loc43 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc43: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc43: Core.Form = init_form %D.ref.loc43 [concrete = constants.%.9a5]
// CHECK:STDOUT:     %pair.param: %tuple.type.281 = value_param call_param0
// CHECK:STDOUT:     %.loc40.1: type = splice_block %.loc40.3 [concrete = constants.%tuple.type.281] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %D.ref.loc39: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %.loc40.2: %tuple.type.24b = tuple_literal (%C.ref, %D.ref.loc39) [concrete = constants.%tuple.a0a]
// CHECK:STDOUT:       %.loc40.3: type = converted %.loc40.2, constants.%tuple.type.281 [concrete = constants.%tuple.type.281]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: %tuple.type.281 = value_binding pair, %pair.param
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
// CHECK:STDOUT: generic fn @F(%T.loc15.2: type) {
// CHECK:STDOUT:   %T.loc15.1: type = symbolic_binding T, 0 [symbolic = %T.loc15.1 (constants.%T)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc15.1, %T.loc15.1) [symbolic = %tuple (constants.%tuple.11e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc15.1, %T.loc15.1) [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:   %pattern_type.loc21: type = pattern_type %tuple.type [symbolic = %pattern_type.loc21 (constants.%pattern_type.c3f)]
// CHECK:STDOUT:   %.loc29.1: Core.Form = init_form %T.loc15.1 [symbolic = %.loc29.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc29: type = pattern_type %T.loc15.1 [symbolic = %pattern_type.loc29 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.07a)) -> out %return.param: @F.%T.loc15.1 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.281) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.281 = name_ref pair, %pair
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc15.1 => constants.%T
// CHECK:STDOUT:   %tuple => constants.%tuple.11e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.07a
// CHECK:STDOUT:   %pattern_type.loc21 => constants.%pattern_type.c3f
// CHECK:STDOUT:   %.loc29.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc29 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:

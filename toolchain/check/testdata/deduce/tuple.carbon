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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.73d: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.4a7: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.53b: <witness> = complete_type_witness %tuple.type.e8a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc7_26 (%pattern_type.65c) = binding_pattern pair [concrete]
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc7_26 (%pattern_type.65c) = value_param_pattern %pair.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_40 (%pattern_type.a32) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_40 (%pattern_type.a32) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_43: type = name_ref U, %U.loc7_16.2 [symbolic = %U.loc7_16.1 (constants.%U)]
// CHECK:STDOUT:     %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %U.loc7_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc7_16.1 (constants.%U)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.30b) = value_param call_param0
// CHECK:STDOUT:     %.loc7_37.1: type = splice_block %.loc7_37.3 [symbolic = %tuple.type (constants.%tuple.type.30b)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc7_36: type = name_ref U, %U.loc7_16.2 [symbolic = %U.loc7_16.1 (constants.%U)]
// CHECK:STDOUT:       %.loc7_37.2: %tuple.type.24b = tuple_literal (%T.ref, %U.ref.loc7_36)
// CHECK:STDOUT:       %.loc7_37.3: type = converted %.loc7_37.2, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.30b) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc7_16.1 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc7_16.1 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.patt: %pattern_type.9ec = binding_pattern pair [concrete]
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.9ec = value_param_pattern %pair.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: @D.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc5: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.73d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: type, %U.loc7_16.2: type) {
// CHECK:STDOUT:   %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc7_16.1 (constants.%U)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc7_6.1, %U.loc7_16.1) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type.loc7_26: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_26 (constants.%pattern_type.65c)]
// CHECK:STDOUT:   %pattern_type.loc7_40: type = pattern_type %U.loc7_16.1 [symbolic = %pattern_type.loc7_40 (constants.%pattern_type.a32)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_40: <witness> = require_complete_type %U.loc7_16.1 [symbolic = %require_complete.loc7_40 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %require_complete.loc7_30: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc7_30 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.1, %U.loc7_16.1) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.dd9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.30b)) -> @F.%U.loc7_16.1 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %pair.ref: @F.%tuple.type (%tuple.type.30b) = name_ref pair, %pair
// CHECK:STDOUT:     %F.specific_fn.loc7_54.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc7_54.2 (constants.%F.specific_fn.dd9)]
// CHECK:STDOUT:     %F.call: init @F.%U.loc7_16.1 (%U) = call %F.specific_fn.loc7_54.1(%pair.ref)
// CHECK:STDOUT:     %.loc7_61.1: @F.%U.loc7_16.1 (%U) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_61.2: @F.%U.loc7_16.1 (%U) = converted %F.call, %.loc7_61.1
// CHECK:STDOUT:     return %.loc7_61.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.e8a) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.e8a = name_ref pair, %pair
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.4a7]
// CHECK:STDOUT:   %.loc9_20: ref %D = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %D = call %F.specific_fn(%pair.ref) to %.loc9_20
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_20, constants.%D.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc9_20
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.call: init %empty_tuple.type = call %D.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_16.1 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type.loc7_26 => constants.%pattern_type.65c
// CHECK:STDOUT:   %pattern_type.loc7_40 => constants.%pattern_type.a32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_40 => constants.%require_complete.b54
// CHECK:STDOUT:   %require_complete.loc7_30 => constants.%require_complete.fe1
// CHECK:STDOUT:   %F.specific_fn.loc7_54.2 => constants.%F.specific_fn.dd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%C
// CHECK:STDOUT:   %U.loc7_16.1 => constants.%D
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e8a
// CHECK:STDOUT:   %pattern_type.loc7_26 => constants.%pattern_type.9ec
// CHECK:STDOUT:   %pattern_type.loc7_40 => constants.%pattern_type.510
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Pair: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [concrete]
// CHECK:STDOUT:   %HasPair.generic: %HasPair.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasPair.920: type = class_type @HasPair, @HasPair(%Pair) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.347: <witness> = impl_witness @HasPair.%Destroy.impl_witness_table, @HasPair.as.Destroy.impl(%Pair) [symbolic]
// CHECK:STDOUT:   %ptr.c3d: type = ptr_type %HasPair.920 [symbolic]
// CHECK:STDOUT:   %pattern_type.ba5: type = pattern_type %ptr.c3d [symbolic]
// CHECK:STDOUT:   %HasPair.as.Destroy.impl.Op.type: type = fn_type @HasPair.as.Destroy.impl.Op, @HasPair.as.Destroy.impl(%Pair) [symbolic]
// CHECK:STDOUT:   %HasPair.as.Destroy.impl.Op: %HasPair.as.Destroy.impl.Op.type = struct_value () [symbolic]
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
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %HasPair.369: type = class_type @HasPair, @HasPair(%tuple.21c) [concrete]
// CHECK:STDOUT:   %pattern_type.93e: type = pattern_type %HasPair.369 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_1.5d2, %int_2.ef8) [concrete]
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
// CHECK:STDOUT:     .HasPair = %HasPair.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasPair.decl: %HasPair.type = class_decl @HasPair [concrete = constants.%HasPair.generic] {
// CHECK:STDOUT:     %Pair.patt: %pattern_type.511 = symbolic_binding_pattern Pair, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_31.1: type = splice_block %.loc4_31.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:       %int_32.loc4_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc4_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_31.2: %tuple.type.24b = tuple_literal (%i32.loc4_23, %i32.loc4_28)
// CHECK:STDOUT:       %.loc4_31.3: type = converted %.loc4_31.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Pair.loc4_15.2: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic = %Pair.loc4_15.1 (constants.%Pair)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.7ce = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.7ce = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:     %h.patt: @F.%pattern_type (%pattern_type.88c) = binding_pattern h [concrete]
// CHECK:STDOUT:     %h.param_patt: @F.%pattern_type (%pattern_type.88c) = value_param_pattern %h.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_47: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_47: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc6_6.2: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc6_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc6_19: type = splice_block %i32.loc6_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_15.2: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc6_15.1 (constants.%B)]
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc6_41.1 (%HasPair.568) = value_param call_param0
// CHECK:STDOUT:     %.loc6_41.1: type = splice_block %HasPair.loc6_41.2 [symbolic = %HasPair.loc6_41.1 (constants.%HasPair.568)] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %A.ref: %i32 = name_ref A, %A.loc6_6.2 [symbolic = %A.loc6_6.1 (constants.%A)]
// CHECK:STDOUT:       %B.ref.loc6_39: %i32 = name_ref B, %B.loc6_15.2 [symbolic = %B.loc6_15.1 (constants.%B)]
// CHECK:STDOUT:       %.loc6_40: %tuple.type.d07 = tuple_literal (%A.ref, %B.ref.loc6_39)
// CHECK:STDOUT:       %tuple.loc6_40.2: %tuple.type.d07 = tuple_value (%A.ref, %B.ref.loc6_39) [symbolic = %tuple.loc6_40.1 (constants.%tuple.159)]
// CHECK:STDOUT:       %.loc6_41.2: %tuple.type.d07 = converted %.loc6_40, %tuple.loc6_40.2 [symbolic = %tuple.loc6_40.1 (constants.%tuple.159)]
// CHECK:STDOUT:       %HasPair.loc6_41.2: type = class_type @HasPair, @HasPair(constants.%tuple.159) [symbolic = %HasPair.loc6_41.1 (constants.%HasPair.568)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %h: @F.%HasPair.loc6_41.1 (%HasPair.568) = bind_name h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %h.patt: %pattern_type.93e = binding_pattern h [concrete]
// CHECK:STDOUT:     %h.param_patt: %pattern_type.93e = value_param_pattern %h.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %h.param: %HasPair.369 = value_param call_param0
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %HasPair [concrete = constants.%HasPair.369] {
// CHECK:STDOUT:       %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [concrete = constants.%HasPair.generic]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:       %.loc8_22.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:       %impl.elem0.loc8_22.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc8_22.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_22.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:       %specific_fn.loc8_22.1: <specific function> = specific_function %impl.elem0.loc8_22.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.2: <bound method> = bound_method %int_1, %specific_fn.loc8_22.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.1: init %i32 = call %bound_method.loc8_22.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %.loc8_22.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %.loc8_22.3: %i32 = converted %int_1, %.loc8_22.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:       %impl.elem0.loc8_22.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:       %bound_method.loc8_22.3: <bound method> = bound_method %int_2, %impl.elem0.loc8_22.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9]
// CHECK:STDOUT:       %specific_fn.loc8_22.2: <specific function> = specific_function %impl.elem0.loc8_22.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc8_22.4: <bound method> = bound_method %int_2, %specific_fn.loc8_22.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.2: init %i32 = call %bound_method.loc8_22.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:       %.loc8_22.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22.2 [concrete = constants.%int_2.ef8]
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
// CHECK:STDOUT: generic impl @HasPair.as.Destroy.impl(@HasPair.%Pair.loc4_15.2: %tuple.type.d07) {
// CHECK:STDOUT:   %Pair: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic = %Pair (constants.%Pair)]
// CHECK:STDOUT:   %HasPair: type = class_type @HasPair, @HasPair(%Pair) [symbolic = %HasPair (constants.%HasPair.920)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @HasPair.%Destroy.impl_witness_table, @HasPair.as.Destroy.impl(%Pair) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.347)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasPair.as.Destroy.impl.Op.type: type = fn_type @HasPair.as.Destroy.impl.Op, @HasPair.as.Destroy.impl(%Pair) [symbolic = %HasPair.as.Destroy.impl.Op.type (constants.%HasPair.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %HasPair.as.Destroy.impl.Op: @HasPair.as.Destroy.impl.%HasPair.as.Destroy.impl.Op.type (%HasPair.as.Destroy.impl.Op.type) = struct_value () [symbolic = %HasPair.as.Destroy.impl.Op (constants.%HasPair.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @HasPair.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %HasPair.as.Destroy.impl.Op.decl: @HasPair.as.Destroy.impl.%HasPair.as.Destroy.impl.Op.type (%HasPair.as.Destroy.impl.Op.type) = fn_decl @HasPair.as.Destroy.impl.Op [symbolic = @HasPair.as.Destroy.impl.%HasPair.as.Destroy.impl.Op (constants.%HasPair.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @HasPair.as.Destroy.impl.Op.%pattern_type (%pattern_type.ba5) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @HasPair.as.Destroy.impl.Op.%pattern_type (%pattern_type.ba5) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_34.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @HasPair.as.Destroy.impl.Op.%ptr (%ptr.c3d) = value_param call_param0
// CHECK:STDOUT:       %.loc4_34.2: type = splice_block %Self.ref [symbolic = %HasPair (constants.%HasPair.920)] {
// CHECK:STDOUT:         %.loc4_34.3: type = specific_constant constants.%HasPair.920, @HasPair(constants.%Pair) [symbolic = %HasPair (constants.%HasPair.920)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_34.3 [symbolic = %HasPair (constants.%HasPair.920)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @HasPair.as.Destroy.impl.Op.%ptr (%ptr.c3d) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %HasPair.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @HasPair.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HasPair(%Pair.loc4_15.2: %tuple.type.d07) {
// CHECK:STDOUT:   %Pair.loc4_15.1: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic = %Pair.loc4_15.1 (constants.%Pair)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%HasPair.920 [symbolic = @HasPair.as.Destroy.impl.%HasPair (constants.%HasPair.920)]
// CHECK:STDOUT:     impl_decl @HasPair.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@HasPair.as.Destroy.impl.%HasPair.as.Destroy.impl.Op.decl), @HasPair.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @HasPair.as.Destroy.impl(constants.%Pair) [symbolic = @HasPair.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.347)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HasPair.920
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasPair.as.Destroy.impl.Op(@HasPair.%Pair.loc4_15.2: %tuple.type.d07) {
// CHECK:STDOUT:   %Pair: %tuple.type.d07 = bind_symbolic_name Pair, 0 [symbolic = %Pair (constants.%Pair)]
// CHECK:STDOUT:   %HasPair: type = class_type @HasPair, @HasPair(%Pair) [symbolic = %HasPair (constants.%HasPair.920)]
// CHECK:STDOUT:   %ptr: type = ptr_type %HasPair [symbolic = %ptr (constants.%ptr.c3d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.ba5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasPair.as.Destroy.impl.Op.%ptr (%ptr.c3d)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc6_6.2: %i32, %B.loc6_15.2: %i32) {
// CHECK:STDOUT:   %A.loc6_6.1: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc6_6.1 (constants.%A)]
// CHECK:STDOUT:   %B.loc6_15.1: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc6_15.1 (constants.%B)]
// CHECK:STDOUT:   %tuple.loc6_40.1: %tuple.type.d07 = tuple_value (%A.loc6_6.1, %B.loc6_15.1) [symbolic = %tuple.loc6_40.1 (constants.%tuple.159)]
// CHECK:STDOUT:   %HasPair.loc6_41.1: type = class_type @HasPair, @HasPair(%tuple.loc6_40.1) [symbolic = %HasPair.loc6_41.1 (constants.%HasPair.568)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HasPair.loc6_41.1 [symbolic = %pattern_type (constants.%pattern_type.88c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasPair.loc6_41.1 [symbolic = %require_complete (constants.%require_complete.6bc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%h.param: @F.%HasPair.loc6_41.1 (%HasPair.568)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %B.ref.loc6_60: %i32 = name_ref B, %B.loc6_15.2 [symbolic = %B.loc6_15.1 (constants.%B)]
// CHECK:STDOUT:     return %B.ref.loc6_60
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param: %HasPair.369) -> %i32 {
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
// CHECK:STDOUT:   %Pair.loc4_15.1 => constants.%Pair
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair.as.Destroy.impl(constants.%Pair) {
// CHECK:STDOUT:   %Pair => constants.%Pair
// CHECK:STDOUT:   %HasPair => constants.%HasPair.920
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.347
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair.as.Destroy.impl.Op(constants.%Pair) {
// CHECK:STDOUT:   %Pair => constants.%Pair
// CHECK:STDOUT:   %HasPair => constants.%HasPair.920
// CHECK:STDOUT:   %ptr => constants.%ptr.c3d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ba5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.159) {
// CHECK:STDOUT:   %Pair.loc4_15.1 => constants.%tuple.159
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc6_6.1 => constants.%A
// CHECK:STDOUT:   %B.loc6_15.1 => constants.%B
// CHECK:STDOUT:   %tuple.loc6_40.1 => constants.%tuple.159
// CHECK:STDOUT:   %HasPair.loc6_41.1 => constants.%HasPair.568
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.21c) {
// CHECK:STDOUT:   %Pair.loc4_15.1 => constants.%tuple.21c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_1.5d2, constants.%int_2.ef8) {
// CHECK:STDOUT:   %A.loc6_6.1 => constants.%int_1.5d2
// CHECK:STDOUT:   %B.loc6_15.1 => constants.%int_2.ef8
// CHECK:STDOUT:   %tuple.loc6_40.1 => constants.%tuple.21c
// CHECK:STDOUT:   %HasPair.loc6_41.1 => constants.%HasPair.369
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.93e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.73d: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d00: type = tuple_type (%T, %T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type %tuple.type.d00 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.e8a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.9ec: type = pattern_type %tuple.type.e8a [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %pair.patt: @F.%pattern_type.loc7_16 (%pattern_type.9a5) = binding_pattern pair [concrete]
// CHECK:STDOUT:     %pair.param_patt: @F.%pattern_type.loc7_16 (%pattern_type.9a5) = value_param_pattern %pair.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_30 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_30 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_33: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.d00) = value_param call_param0
// CHECK:STDOUT:     %.loc7_27.1: type = splice_block %.loc7_27.3 [symbolic = %tuple.type (constants.%tuple.type.d00)] {
// CHECK:STDOUT:       %T.ref.loc7_23: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc7_26: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_27.2: %tuple.type.24b = tuple_literal (%T.ref.loc7_23, %T.ref.loc7_26)
// CHECK:STDOUT:       %.loc7_27.3: type = converted %.loc7_27.2, constants.%tuple.type.d00 [symbolic = %tuple.type (constants.%tuple.type.d00)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.d00) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %pair.patt: %pattern_type.9ec = binding_pattern pair [concrete]
// CHECK:STDOUT:     %pair.param_patt: %pattern_type.9ec = value_param_pattern %pair.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: @D.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc5: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.73d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: type) {
// CHECK:STDOUT:   %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc7_6.1, %T.loc7_6.1) [symbolic = %tuple.type (constants.%tuple.type.d00)]
// CHECK:STDOUT:   %pattern_type.loc7_16: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_16 (constants.%pattern_type.9a5)]
// CHECK:STDOUT:   %pattern_type.loc7_30: type = pattern_type %T.loc7_6.1 [symbolic = %pattern_type.loc7_30 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%pair.param: @F.%tuple.type (%tuple.type.d00)) -> @F.%T.loc7_6.1 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param: %tuple.type.e8a) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.e8a = name_ref pair, %pair
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d00
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.9a5
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:

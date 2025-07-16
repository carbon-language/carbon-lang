// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_self_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_self_type_mismatch.carbon

class C[T:! type](X:! T) {}

interface I {
  // Uses of `Self` inside the definition of `I` have type `I`.
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+6]]:8: error: cannot implicitly convert expression of type `C(i32 as I)` to `C(i32)` [ConversionFailure]
  // CHECK:STDERR:   fn F(c: C(Self));
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+3]]:8: note: type `C(i32 as I)` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F(c: C(Self));
  // CHECK:STDERR:        ^~~~~~~~~~
  fn F(c: C(Self));
}

impl bool as I {
  // In this scope, `Self` is `bool`, but the signature of `F` uses the `Self`
  // from the interface, not the `Self` from the impl.

  fn F(c: C(bool as I));
}

impl i32 as I {
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+7]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F(c: C(i32));
  // CHECK:STDERR:        ^~~~~~~~~
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE-14]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F(c: C(Self));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(c: C(i32));
}

// CHECK:STDOUT: --- fail_self_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.b36: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %C.dbb: type = class_type @C, @C(%I.type, %Self.826) [symbolic]
// CHECK:STDOUT:   %pattern_type.4fb: type = pattern_type %C.dbb [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.loc25 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness.c21: <witness> = impl_witness file.%I.impl_witness_table.loc28 [concrete]
// CHECK:STDOUT:   %I.facet.e8a: %I.type = facet_value bool, (%I.impl_witness.c21) [concrete]
// CHECK:STDOUT:   %C.7db: type = class_type @C, @C(%I.type, %I.facet.e8a) [concrete]
// CHECK:STDOUT:   %pattern_type.225: type = pattern_type %C.7db [concrete]
// CHECK:STDOUT:   %F.type.8ba: type = fn_type @F.loc32 [concrete]
// CHECK:STDOUT:   %F.2cf: %F.type.8ba = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness.863: <witness> = impl_witness file.%I.impl_witness_table.loc35 [concrete]
// CHECK:STDOUT:   %C.6fb: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.52b: type = pattern_type %C.6fb [concrete]
// CHECK:STDOUT:   %F.type.066a53.1: type = fn_type @F.loc43_18.1 [concrete]
// CHECK:STDOUT:   %F.9ec58f.1: %F.type.066a53.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.118: %I.type = facet_value %i32, (%I.impl_witness.863) [concrete]
// CHECK:STDOUT:   %C.d88: type = class_type @C, @C(%I.type, %I.facet.118) [concrete]
// CHECK:STDOUT:   %pattern_type.c65: type = pattern_type %C.d88 [concrete]
// CHECK:STDOUT:   %F.type.066a53.2: type = fn_type @F.loc43_18.2 [concrete]
// CHECK:STDOUT:   %F.9ec58f.2: %F.type.066a53.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %X.patt: @C.%pattern_type (%pattern_type.7dcd0a.1) = symbolic_binding_pattern X, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_9.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_9.1 [symbolic = %T.loc15_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc15_19.1: @C.%T.loc15_9.2 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc15_19.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @I.impl.049 [concrete] {} {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc28_6.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc28_6.2: type = converted %bool.make_type, %.loc28_6.1 [concrete = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc28 = impl_witness_table (@I.impl.049.%F.decl), @I.impl.049 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc28: <witness> = impl_witness %I.impl_witness_table.loc28 [concrete = constants.%I.impl_witness.c21]
// CHECK:STDOUT:   impl_decl @I.impl.a9a [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc35 = impl_witness_table (@I.impl.a9a.%F.decl.loc43_18.2), @I.impl.a9a [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc35: <witness> = impl_witness %I.impl_witness_table.loc35 [concrete = constants.%I.impl_witness.863]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.loc25 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %c.patt: @F.loc25.%pattern_type (%pattern_type.4fb) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @F.loc25.%pattern_type (%pattern_type.4fb) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @F.loc25.%C.loc25_17.1 (%C.dbb) = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %C.loc25_17.2 [symbolic = %C.loc25_17.1 (constants.%C.dbb)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %C.loc25_17.2: type = class_type @C, @C(constants.%I.type, constants.%Self.826) [symbolic = %C.loc25_17.1 (constants.%C.dbb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @F.loc25.%C.loc25_17.1 (%C.dbb) = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @I.impl.049: %.loc28_6.2 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.8ba = fn_decl @F.loc32 [concrete = constants.%F.2cf] {
// CHECK:STDOUT:     %c.patt: %pattern_type.225 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.225 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.7db = value_param call_param0
// CHECK:STDOUT:     %.loc32_22: type = splice_block %C [concrete = constants.%C.7db] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %I.facet: %I.type = facet_value bool, (constants.%I.impl_witness.c21) [concrete = constants.%I.facet.e8a]
// CHECK:STDOUT:       %.loc32_18: %I.type = converted %bool.make_type, %I.facet [concrete = constants.%I.facet.e8a]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, constants.%I.facet.e8a) [concrete = constants.%C.7db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.7db = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @I.impl.a9a: %i32 as %I.ref {
// CHECK:STDOUT:   %F.decl.loc43_18.1: %F.type.066a53.1 = fn_decl @F.loc43_18.1 [concrete = constants.%F.9ec58f.1] {
// CHECK:STDOUT:     %c.patt: %pattern_type.52b = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.52b = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.6fb = value_param call_param0
// CHECK:STDOUT:     %.loc43: type = splice_block %C [concrete = constants.%C.6fb] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type, constants.%i32) [concrete = constants.%C.6fb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.6fb = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc43_18.2: %F.type.066a53.2 = fn_decl @F.loc43_18.2 [concrete = constants.%F.9ec58f.2] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c65 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c65 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.d88 = value_param call_param0
// CHECK:STDOUT:     %c: %C.d88 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc43_18.1
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc15_9.1: type, %X.loc15_19.1: @C.%T.loc15_9.2 (%T)) {
// CHECK:STDOUT:   %T.loc15_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_9.2 (constants.%T)]
// CHECK:STDOUT:   %X.loc15_19.2: @C.%T.loc15_9.2 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc15_19.2 (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc15_9.2 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.b36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc25(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %C.loc25_17.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc25_17.1 (constants.%C.dbb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc25_17.1 [symbolic = %pattern_type (constants.%pattern_type.4fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @F.loc25.%C.loc25_17.1 (%C.dbb));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc32(%c.param: %C.7db);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc43_18.1(%c.param: %C.6fb);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc43_18.2(%c.param: %C.d88) [thunk @I.impl.a9a.%F.decl.loc43_18.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.066a53.1 = name_ref F, @I.impl.a9a.%F.decl.loc43_18.1 [concrete = constants.%F.9ec58f.1]
// CHECK:STDOUT:   %c.ref: %C.d88 = name_ref c, %c.param
// CHECK:STDOUT:   %.loc25: %C.6fb = converted %c.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc15_9.2 => constants.%T
// CHECK:STDOUT:   %X.loc15_19.2 => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self.826) {
// CHECK:STDOUT:   %T.loc15_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.2 => constants.%Self.826
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc25(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.dbb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.e8a) {
// CHECK:STDOUT:   %T.loc15_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.2 => constants.%I.facet.e8a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc25(constants.%I.facet.e8a) {
// CHECK:STDOUT:   %Self => constants.%I.facet.e8a
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.7db
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc15_9.2 => type
// CHECK:STDOUT:   %X.loc15_19.2 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc25(constants.%I.facet.118) {
// CHECK:STDOUT:   %Self => constants.%I.facet.118
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.d88
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.118) {
// CHECK:STDOUT:   %T.loc15_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.2 => constants.%I.facet.118
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

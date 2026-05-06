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
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+3]]:8: note: type `C(i32 as I)` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f05: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %C.eab: type = class_type @C, @C(%I.type, %Self.ab9) [symbolic]
// CHECK:STDOUT:   %pattern_type.a8e: type = pattern_type %C.eab [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.ab9) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.18e: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness.f69: <witness> = impl_witness @bool.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.e45: %I.type = facet_value bool, (%I.impl_witness.f69) [concrete]
// CHECK:STDOUT:   %C.60c: type = class_type @C, @C(%I.type, %I.facet.e45) [concrete]
// CHECK:STDOUT:   %pattern_type.212: type = pattern_type %C.60c [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F.type: type = fn_type @bool.as.I.impl.F [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F: %bool.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.347: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.e45) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.04c: %I.WithSelf.F.type.347 = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness.37a: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.899: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.619: type = pattern_type %C.899 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.6c53e7.1: type = fn_type @i32.as.I.impl.F.loc43_18.1 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.7f652f.1: %i32.as.I.impl.F.type.6c53e7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.20f: %I.type = facet_value %i32, (%I.impl_witness.37a) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.f37: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.20f) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.b05: %I.WithSelf.F.type.f37 = struct_value () [concrete]
// CHECK:STDOUT:   %C.719: type = class_type @C, @C(%I.type, %I.facet.20f) [concrete]
// CHECK:STDOUT:   %pattern_type.54c: type = pattern_type %C.719 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.6c53e7.2: type = fn_type @i32.as.I.impl.F.loc43_18.2 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.7f652f.2: %i32.as.I.impl.F.type.6c53e7.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %X.patt: @C.%pattern_type (%pattern_type.51d1c4.1) = symbolic_binding_pattern X, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_13.1: type = splice_block %.loc15_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc15_10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15_23: type = splice_block %T.ref [symbolic = %T.loc15_10.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.loc15_20: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc15_10.2 [symbolic = %T.loc15_10.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc15_20.2: @C.%T.loc15_10.1 (%T) = symbolic_binding X, 1 [symbolic = %X.loc15_20.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @bool.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {
// CHECK:STDOUT:     %c.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.a8e) = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: @I.WithSelf.F.%pattern_type (%pattern_type.a8e) = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @I.WithSelf.F.%C.loc25_17.1 (%C.eab) = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %C.loc25_17.2 [symbolic = %C.loc25_17.1 (constants.%C.eab)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %C.loc25_17.2: type = class_type @C, @C(constants.%I.type, constants.%Self.ab9) [symbolic = %C.loc25_17.1 (constants.%C.eab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @I.WithSelf.F.%C.loc25_17.1 (%C.eab) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.18e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @bool.as.I.impl: %.loc28 as %I.ref {
// CHECK:STDOUT:   %bool.as.I.impl.F.decl: %bool.as.I.impl.F.type = fn_decl @bool.as.I.impl.F [concrete = constants.%bool.as.I.impl.F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.212 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.212 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.60c = value_param call_param0
// CHECK:STDOUT:     %.loc32_22: type = splice_block %C [concrete = constants.%C.60c] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc32_13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %I.facet: %I.type = facet_value %.loc32_13, (constants.%I.impl_witness.f69) [concrete = constants.%I.facet.e45]
// CHECK:STDOUT:       %.loc32_18: %I.type = converted %.loc32_13, %I.facet [concrete = constants.%I.facet.e45]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, constants.%I.facet.e45) [concrete = constants.%C.60c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.60c = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%bool.as.I.impl.F.decl), @bool.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.f69]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = %bool.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc43_18.1: %i32.as.I.impl.F.type.6c53e7.1 = fn_decl @i32.as.I.impl.F.loc43_18.1 [concrete = constants.%i32.as.I.impl.F.7f652f.1] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.619 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.619 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.899 = value_param call_param0
// CHECK:STDOUT:     %.loc43: type = splice_block %C [concrete = constants.%C.899] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type, constants.%i32) [concrete = constants.%C.899]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.899 = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc43_18.2: %i32.as.I.impl.F.type.6c53e7.2 = fn_decl @i32.as.I.impl.F.loc43_18.2 [concrete = constants.%i32.as.I.impl.F.7f652f.2] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.54c = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.54c = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.719 = value_param call_param0
// CHECK:STDOUT:     %c: %C.719 = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%i32.as.I.impl.F.decl.loc43_18.2), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.37a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %i32.as.I.impl.F.decl.loc43_18.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc15_10.2: type, %X.loc15_20.2: @C.%T.loc15_10.1 (%T)) {
// CHECK:STDOUT:   %T.loc15_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_10.1 (constants.%T)]
// CHECK:STDOUT:   %X.loc15_20.1: @C.%T.loc15_10.1 (%T) = symbolic_binding X, 1 [symbolic = %X.loc15_20.1 (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc15_10.1 [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f05
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:   %C.loc25_17.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc25_17.1 (constants.%C.eab)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc25_17.1 [symbolic = %pattern_type (constants.%pattern_type.a8e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @I.WithSelf.F.%C.loc25_17.1 (%C.eab));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bool.as.I.impl.F(%c.param: %C.60c);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc43_18.1(%c.param: %C.899);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc43_18.2(%c.param: %C.719) [thunk @i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %i32.as.I.impl.F.type.6c53e7.1 = name_ref F, @i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.1 [concrete = constants.%i32.as.I.impl.F.7f652f.1]
// CHECK:STDOUT:   %.loc25: %C.899 = converted %c.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %i32.as.I.impl.F.call: init %empty_tuple.type = call %F.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc15_10.1 => constants.%T
// CHECK:STDOUT:   %X.loc15_20.1 => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self.ab9) {
// CHECK:STDOUT:   %T.loc15_10.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_20.1 => constants.%Self.ab9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.ab9) {
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.eab
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a8e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.e45) {
// CHECK:STDOUT:   %T.loc15_10.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_20.1 => constants.%I.facet.e45
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.e45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.e45
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.347
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.04c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.e45) {
// CHECK:STDOUT:   %Self => constants.%I.facet.e45
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.60c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc15_10.1 => type
// CHECK:STDOUT:   %X.loc15_20.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.20f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.20f
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.f37
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.b05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.20f) {
// CHECK:STDOUT:   %Self => constants.%I.facet.20f
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.719
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.54c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.20f) {
// CHECK:STDOUT:   %T.loc15_10.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_20.1 => constants.%I.facet.20f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

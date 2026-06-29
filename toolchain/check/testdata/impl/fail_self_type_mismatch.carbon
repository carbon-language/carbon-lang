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
// CHECK:STDERR: fail_self_type_mismatch.carbon: error: cannot implicitly convert expression of type `C(i32 as I)` to `C(i32)` [ConversionFailure]
// CHECK:STDERR: fail_self_type_mismatch.carbon: note: type `C(i32 as I)` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessInContext]

class C[T:! type](X:! T) {}

interface I {
  // Uses of `Self` inside the definition of `I` have type `I`.
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
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %X.patt.9f0: %pattern_type.51d1c4.1 = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.0b1: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %X.patt.3b7: %pattern_type.6cb = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.a21: type = class_type @C, @C(%I.type, %Self.37e) [symbolic]
// CHECK:STDOUT:   %pattern_type.59c: type = pattern_type %C.a21 [symbolic]
// CHECK:STDOUT:   %c.param_patt.397: %pattern_type.59c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt.dac: %pattern_type.59c = at_binding_pattern c, %c.param_patt.397 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.dcc: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.ec6: %I.WithSelf.F.type.dcc = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.6c9: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness.020: <witness> = impl_witness @bool.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.11c: %I.type = facet_value bool, (%I.impl_witness.020) [concrete]
// CHECK:STDOUT:   %C.071: type = class_type @C, @C(%I.type, %I.facet.11c) [concrete]
// CHECK:STDOUT:   %pattern_type.85b: type = pattern_type %C.071 [concrete]
// CHECK:STDOUT:   %c.param_patt.ff4: %pattern_type.85b = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.e1c: %pattern_type.85b = at_binding_pattern c, %c.param_patt.ff4 [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F.type: type = fn_type @bool.as.I.impl.F [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F: %bool.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.4fb: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.11c) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.492: %I.WithSelf.F.type.4fb = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness.903: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.patt.014: %pattern_type.98f = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.8dc: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.900: type = pattern_type %C.8dc [concrete]
// CHECK:STDOUT:   %c.param_patt.e0e: %pattern_type.900 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.c85: %pattern_type.900 = at_binding_pattern c, %c.param_patt.e0e [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.48cdab.1: type = fn_type @i32.as.I.impl.F.loc39_18.1 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.bd5be1.1: %i32.as.I.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.858: %I.type = facet_value %i32, (%I.impl_witness.903) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.b7c: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.858) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ae9: %I.WithSelf.F.type.b7c = struct_value () [concrete]
// CHECK:STDOUT:   %C.efa: type = class_type @C, @C(%I.type, %I.facet.858) [concrete]
// CHECK:STDOUT:   %pattern_type.b7c: type = pattern_type %C.efa [concrete]
// CHECK:STDOUT:   %c.param_patt.fb1: %pattern_type.b7c = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.d03: %pattern_type.b7c = at_binding_pattern c, %c.param_patt.fb1 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.48cdab.2: type = fn_type @i32.as.I.impl.F.loc39_18.2 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.bd5be1.2: %i32.as.I.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fcc: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.803: type = facet_type <@ImplicitAs, @ImplicitAs(%C.8dc)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.385: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%C.8dc) [concrete]
// CHECK:STDOUT:   %assoc0.82f: %ImplicitAs.assoc_type.385 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %assoc0.0ac: %ImplicitAs.assoc_type.fcc = assoc_entity element0, imports.%Core.import_ref.f9d [symbolic]
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
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.484: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fcc) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.0ac)]
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97e) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.945)]
// CHECK:STDOUT:   %Core.import_ref.f9d = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %T.patt.loc17_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc17_20.1: @C.%pattern_type (%pattern_type.51d1c4.1) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc17_20.2 (constants.%X.patt.9f0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_13.1: type = splice_block %.loc17_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc17_10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc17_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc17_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc17_23: type = splice_block %T.ref [symbolic = %T.loc17_10.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.loc17_20: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc17_10.2 [symbolic = %T.loc17_10.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc17_20.2: @C.%T.loc17_10.1 (%T) = symbolic_binding X, 1 [symbolic = %X.loc17_20.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @bool.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc24: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.dcc) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ec6)] {
// CHECK:STDOUT:     %c.param_patt.loc21_9.1: @I.WithSelf.F.%pattern_type (%pattern_type.59c) = value_param_pattern [symbolic = %c.param_patt.loc21_9.2 (constants.%c.param_patt.397)]
// CHECK:STDOUT:     %c.patt.loc21_9.1: @I.WithSelf.F.%pattern_type (%pattern_type.59c) = at_binding_pattern c, %c.param_patt.loc21_9.1 [symbolic = %c.patt.loc21_9.2 (constants.%c.patt.dac)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @I.WithSelf.F.%C.loc21_17.1 (%C.a21) = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %C.loc21_17.2 [symbolic = %C.loc21_17.1 (constants.%C.a21)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %C.loc21_17.2: type = class_type @C, @C(constants.%I.type, constants.%Self.37e) [symbolic = %C.loc21_17.1 (constants.%C.a21)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @I.WithSelf.F.%C.loc21_17.1 (%C.a21) = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.6c9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @bool.as.I.impl: %.loc24 as %I.ref {
// CHECK:STDOUT:   %bool.as.I.impl.F.decl: %bool.as.I.impl.F.type = fn_decl @bool.as.I.impl.F [concrete = constants.%bool.as.I.impl.F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.85b = value_param_pattern [concrete = constants.%c.param_patt.ff4]
// CHECK:STDOUT:     %c.patt: %pattern_type.85b = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt.e1c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.071 = value_param call_param0
// CHECK:STDOUT:     %.loc28_22: type = splice_block %C [concrete = constants.%C.071] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc28_13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %I.facet: %I.type = facet_value %.loc28_13, (constants.%I.impl_witness.020) [concrete = constants.%I.facet.11c]
// CHECK:STDOUT:       %.loc28_18: %I.type = converted %.loc28_13, %I.facet [concrete = constants.%I.facet.11c]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, constants.%I.facet.11c) [concrete = constants.%C.071]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.071 = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%bool.as.I.impl.F.decl), @bool.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.020]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = %bool.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc39_18.1: %i32.as.I.impl.F.type.48cdab.1 = fn_decl @i32.as.I.impl.F.loc39_18.1 [concrete = constants.%i32.as.I.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.900 = value_param_pattern [concrete = constants.%c.param_patt.e0e]
// CHECK:STDOUT:     %c.patt: %pattern_type.900 = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt.c85]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.8dc = value_param call_param0
// CHECK:STDOUT:     %.loc39: type = splice_block %C [concrete = constants.%C.8dc] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type, constants.%i32) [concrete = constants.%C.8dc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.8dc = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc39_18.2: %i32.as.I.impl.F.type.48cdab.2 = fn_decl @i32.as.I.impl.F.loc39_18.2 [concrete = constants.%i32.as.I.impl.F.bd5be1.2] {
// CHECK:STDOUT:     %.1: %pattern_type.b7c = specific_constant constants.%c.param_patt.fb1, @I.WithSelf.F(constants.%I.facet.858) [concrete = constants.%c.param_patt.fb1]
// CHECK:STDOUT:     %c.patt: %pattern_type.b7c = at_binding_pattern c, %.1 [concrete = constants.%c.patt.d03]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.efa = value_param call_param0
// CHECK:STDOUT:     %c: %C.efa = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%i32.as.I.impl.F.decl.loc39_18.2), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.903]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %i32.as.I.impl.F.decl.loc39_18.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc17_10.2: type, %X.loc17_20.2: @C.%T.loc17_10.1 (%T)) {
// CHECK:STDOUT:   %T.patt.loc17_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc17_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc17_10.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc17_10.1 [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %X.patt.loc17_20.2: @C.%pattern_type (%pattern_type.51d1c4.1) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc17_20.2 (constants.%X.patt.9f0)]
// CHECK:STDOUT:   %X.loc17_20.1: @C.%T.loc17_10.1 (%T) = symbolic_binding X, 1 [symbolic = %X.loc17_20.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.0b1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:   %C.loc21_17.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc21_17.1 (constants.%C.a21)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc21_17.1 [symbolic = %pattern_type (constants.%pattern_type.59c)]
// CHECK:STDOUT:   %c.param_patt.loc21_9.2: @I.WithSelf.F.%pattern_type (%pattern_type.59c) = value_param_pattern [symbolic = %c.param_patt.loc21_9.2 (constants.%c.param_patt.397)]
// CHECK:STDOUT:   %c.patt.loc21_9.2: @I.WithSelf.F.%pattern_type (%pattern_type.59c) = at_binding_pattern c, %c.param_patt.loc21_9.2 [symbolic = %c.patt.loc21_9.2 (constants.%c.patt.dac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @I.WithSelf.F.%C.loc21_17.1 (%C.a21));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bool.as.I.impl.F(%c.param: %C.071);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc39_18.1(%c.param: %C.8dc);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc39_18.2(%c.param: %C.efa) [thunk @i32.as.I.impl.%i32.as.I.impl.F.decl.loc39_18.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.858)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %i32.as.I.impl.F.type.48cdab.1 = name_ref F, @i32.as.I.impl.%i32.as.I.impl.F.decl.loc39_18.1 [concrete = constants.%i32.as.I.impl.F.bd5be1.1]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%C.8dc)> [concrete = constants.%ImplicitAs.type.803]
// CHECK:STDOUT:   %.2: %ImplicitAs.assoc_type.385 = specific_constant imports.%Core.import_ref.484, @ImplicitAs.WithSelf(constants.%C.8dc, constants.%Self.294) [concrete = constants.%assoc0.82f]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.385 = name_ref Convert, %.2 [concrete = constants.%assoc0.82f]
// CHECK:STDOUT:   %.3: %C.8dc = converted %c.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %i32.as.I.impl.F.call: init %empty_tuple.type = call %F.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.patt.loc17_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_10.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %X.patt.loc17_20.2 => constants.%X.patt.9f0
// CHECK:STDOUT:   %X.loc17_20.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.dcc
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ec6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self.37e) {
// CHECK:STDOUT:   %T.patt.loc17_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_10.1 => constants.%I.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6cb
// CHECK:STDOUT:   %X.patt.loc17_20.2 => constants.%X.patt.3b7
// CHECK:STDOUT:   %X.loc17_20.1 => constants.%Self.37e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %C.loc21_17.1 => constants.%C.a21
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.59c
// CHECK:STDOUT:   %c.param_patt.loc21_9.2 => constants.%c.param_patt.397
// CHECK:STDOUT:   %c.patt.loc21_9.2 => constants.%c.patt.dac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.11c) {
// CHECK:STDOUT:   %T.patt.loc17_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_10.1 => constants.%I.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6cb
// CHECK:STDOUT:   %X.patt.loc17_20.2 => constants.%X.patt.3b7
// CHECK:STDOUT:   %X.loc17_20.1 => constants.%I.facet.11c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.11c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.11c
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.4fb
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.492
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.11c) {
// CHECK:STDOUT:   %Self => constants.%I.facet.11c
// CHECK:STDOUT:   %C.loc21_17.1 => constants.%C.071
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.85b
// CHECK:STDOUT:   %c.param_patt.loc21_9.2 => constants.%c.param_patt.ff4
// CHECK:STDOUT:   %c.patt.loc21_9.2 => constants.%c.patt.e1c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc17_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_10.1 => type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:   %X.patt.loc17_20.2 => constants.%X.patt.014
// CHECK:STDOUT:   %X.loc17_20.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.858) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.858
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.b7c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ae9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.858) {
// CHECK:STDOUT:   %Self => constants.%I.facet.858
// CHECK:STDOUT:   %C.loc21_17.1 => constants.%C.efa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b7c
// CHECK:STDOUT:   %c.param_patt.loc21_9.2 => constants.%c.param_patt.fb1
// CHECK:STDOUT:   %c.patt.loc21_9.2 => constants.%c.patt.d03
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.858) {
// CHECK:STDOUT:   %T.patt.loc17_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_10.1 => constants.%I.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6cb
// CHECK:STDOUT:   %X.patt.loc17_20.2 => constants.%X.patt.3b7
// CHECK:STDOUT:   %X.loc17_20.1 => constants.%I.facet.858
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

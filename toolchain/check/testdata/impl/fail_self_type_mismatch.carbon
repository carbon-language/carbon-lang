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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
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
// CHECK:STDOUT:   %Self.7ee: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.09a: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %C.b2a: type = class_type @C, @C(%I.type, %Self.7ee) [symbolic]
// CHECK:STDOUT:   %pattern_type.1b2: type = pattern_type %C.b2a [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness.ef0: <witness> = impl_witness file.%I.impl_witness_table.loc28 [concrete]
// CHECK:STDOUT:   %I.facet.acb: %I.type = facet_value bool, (%I.impl_witness.ef0) [concrete]
// CHECK:STDOUT:   %C.41d: type = class_type @C, @C(%I.type, %I.facet.acb) [concrete]
// CHECK:STDOUT:   %pattern_type.57d: type = pattern_type %C.41d [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F.type: type = fn_type @bool.as.I.impl.F [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F: %bool.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness.e51: <witness> = impl_witness file.%I.impl_witness_table.loc35 [concrete]
// CHECK:STDOUT:   %C.6fb: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.52b: type = pattern_type %C.6fb [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.4dd22d.1: type = fn_type @i32.as.I.impl.F.loc43_18.1 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.b178b8.1: %i32.as.I.impl.F.type.4dd22d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.0ee: %I.type = facet_value %i32, (%I.impl_witness.e51) [concrete]
// CHECK:STDOUT:   %C.808: type = class_type @C, @C(%I.type, %I.facet.0ee) [concrete]
// CHECK:STDOUT:   %pattern_type.2bf: type = pattern_type %C.808 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.4dd22d.2: type = fn_type @i32.as.I.impl.F.loc43_18.2 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.b178b8.2: %i32.as.I.impl.F.type.4dd22d.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15: type = splice_block %T.ref [symbolic = %T.loc15_9.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc15_9.2 [symbolic = %T.loc15_9.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc15_19.2: @C.%T.loc15_9.1 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc15_19.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @bool.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc28_6.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc28_6.2: type = converted %Bool.call, %.loc28_6.1 [concrete = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc28 = impl_witness_table (@bool.as.I.impl.%bool.as.I.impl.F.decl), @bool.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc28: <witness> = impl_witness %I.impl_witness_table.loc28 [concrete = constants.%I.impl_witness.ef0]
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc35 = impl_witness_table (@i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.2), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc35: <witness> = impl_witness %I.impl_witness_table.loc35 [concrete = constants.%I.impl_witness.e51]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %c.patt: @I.F.%pattern_type (%pattern_type.1b2) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @I.F.%pattern_type (%pattern_type.1b2) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @I.F.%C.loc25_17.1 (%C.b2a) = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %C.loc25_17.2 [symbolic = %C.loc25_17.1 (constants.%C.b2a)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.7ee)]
// CHECK:STDOUT:       %C.loc25_17.2: type = class_type @C, @C(constants.%I.type, constants.%Self.7ee) [symbolic = %C.loc25_17.1 (constants.%C.b2a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @I.F.%C.loc25_17.1 (%C.b2a) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @bool.as.I.impl: %.loc28_6.2 as %I.ref {
// CHECK:STDOUT:   %bool.as.I.impl.F.decl: %bool.as.I.impl.F.type = fn_decl @bool.as.I.impl.F [concrete = constants.%bool.as.I.impl.F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.57d = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.57d = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.41d = value_param call_param0
// CHECK:STDOUT:     %.loc32_22: type = splice_block %C [concrete = constants.%C.41d] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %I.facet: %I.type = facet_value %Bool.call, (constants.%I.impl_witness.ef0) [concrete = constants.%I.facet.acb]
// CHECK:STDOUT:       %.loc32_18: %I.type = converted %Bool.call, %I.facet [concrete = constants.%I.facet.acb]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, constants.%I.facet.acb) [concrete = constants.%C.41d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.41d = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = %bool.as.I.impl.F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc43_18.1: %i32.as.I.impl.F.type.4dd22d.1 = fn_decl @i32.as.I.impl.F.loc43_18.1 [concrete = constants.%i32.as.I.impl.F.b178b8.1] {
// CHECK:STDOUT:     %c.patt: %pattern_type.52b = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.52b = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.6fb = value_param call_param0
// CHECK:STDOUT:     %.loc43: type = splice_block %C [concrete = constants.%C.6fb] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type, constants.%i32) [concrete = constants.%C.6fb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.6fb = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc43_18.2: %i32.as.I.impl.F.type.4dd22d.2 = fn_decl @i32.as.I.impl.F.loc43_18.2 [concrete = constants.%i32.as.I.impl.F.b178b8.2] {
// CHECK:STDOUT:     %c.patt: %pattern_type.2bf = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.2bf = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.808 = value_param call_param0
// CHECK:STDOUT:     %c: %C.808 = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %i32.as.I.impl.F.decl.loc43_18.1
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc15_9.2: type, %X.loc15_19.2: @C.%T.loc15_9.1 (%T)) {
// CHECK:STDOUT:   %T.loc15_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_9.1 (constants.%T)]
// CHECK:STDOUT:   %X.loc15_19.1: @C.%T.loc15_9.1 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc15_19.1 (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc15_9.1 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.b36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.7ee)]
// CHECK:STDOUT:   %C.loc25_17.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc25_17.1 (constants.%C.b2a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc25_17.1 [symbolic = %pattern_type (constants.%pattern_type.1b2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @I.F.%C.loc25_17.1 (%C.b2a));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bool.as.I.impl.F(%c.param: %C.41d);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc43_18.1(%c.param: %C.6fb);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc43_18.2(%c.param: %C.808) [thunk @i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %i32.as.I.impl.F.type.4dd22d.1 = name_ref F, @i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.1 [concrete = constants.%i32.as.I.impl.F.b178b8.1]
// CHECK:STDOUT:   %c.ref: %C.808 = name_ref c, %c.param
// CHECK:STDOUT:   %.loc25: %C.6fb = converted %c.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %i32.as.I.impl.F.call: init %empty_tuple.type = call %F.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%T
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self.7ee) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%Self.7ee
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.09a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.7ee) {
// CHECK:STDOUT:   %Self => constants.%Self.7ee
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.b2a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.acb) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%I.facet.acb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.09a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.acb) {
// CHECK:STDOUT:   %Self => constants.%I.facet.acb
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.41d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.57d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc15_9.1 => type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.0ee) {
// CHECK:STDOUT:   %Self => constants.%I.facet.0ee
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.808
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.0ee) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%I.facet.0ee
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.09a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.b36: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.5d3: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T, %X) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %ptr.555: type = ptr_type %C.b36 [symbolic]
// CHECK:STDOUT:   %pattern_type.7e5: type = pattern_type %ptr.555 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T, %X) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ce4: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.71c: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %C.4cb: type = class_type @C, @C(%I.type, %Self.ce4) [symbolic]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %C.4cb [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness.cd6: <witness> = impl_witness file.%I.impl_witness_table.loc28 [concrete]
// CHECK:STDOUT:   %I.facet.898: %I.type = facet_value bool, (%I.impl_witness.cd6) [concrete]
// CHECK:STDOUT:   %C.6d9: type = class_type @C, @C(%I.type, %I.facet.898) [concrete]
// CHECK:STDOUT:   %pattern_type.928: type = pattern_type %C.6d9 [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F.type: type = fn_type @bool.as.I.impl.F [concrete]
// CHECK:STDOUT:   %bool.as.I.impl.F: %bool.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness.ce4: <witness> = impl_witness file.%I.impl_witness_table.loc35 [concrete]
// CHECK:STDOUT:   %C.6fb: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.52b: type = pattern_type %C.6fb [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.7b2a08.1: type = fn_type @i32.as.I.impl.F.loc43_18.1 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.4105f2.1: %i32.as.I.impl.F.type.7b2a08.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.274: %I.type = facet_value %i32, (%I.impl_witness.ce4) [concrete]
// CHECK:STDOUT:   %C.70c: type = class_type @C, @C(%I.type, %I.facet.274) [concrete]
// CHECK:STDOUT:   %pattern_type.f5b: type = pattern_type %C.70c [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type.7b2a08.2: type = fn_type @i32.as.I.impl.F.loc43_18.2 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.4105f2.2: %i32.as.I.impl.F.type.7b2a08.2 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.2f9: <witness> = complete_type_witness %I.type [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:     %T.loc15_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc15: type = splice_block %T.ref [symbolic = %T.loc15_9.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
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
// CHECK:STDOUT:   %I.impl_witness.loc28: <witness> = impl_witness %I.impl_witness_table.loc28 [concrete = constants.%I.impl_witness.cd6]
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc35 = impl_witness_table (@i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.2), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc35: <witness> = impl_witness %I.impl_witness_table.loc35 [concrete = constants.%I.impl_witness.ce4]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ce4]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %c.patt: @I.F.%pattern_type (%pattern_type.354) = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @I.F.%pattern_type (%pattern_type.354) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @I.F.%C.loc25_17.1 (%C.4cb) = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %C.loc25_17.2 [symbolic = %C.loc25_17.1 (constants.%C.4cb)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ce4)]
// CHECK:STDOUT:       %C.loc25_17.2: type = class_type @C, @C(constants.%I.type, constants.%Self.ce4) [symbolic = %C.loc25_17.1 (constants.%C.4cb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @I.F.%C.loc25_17.1 (%C.4cb) = bind_name c, %c.param
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
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%T.loc15_9.2: type, @C.%X.loc15_19.2: @C.%T.loc15_9.1 (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @C.as.Destroy.impl.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T, %X) [symbolic = %C (constants.%C.b36)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T, %X) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.5d3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T, %X) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.7e5) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.7e5) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc15_26.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.555) = value_param call_param0
// CHECK:STDOUT:       %.loc15_26.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.b36)] {
// CHECK:STDOUT:         %.loc15_26.3: type = specific_constant constants.%C.b36, @C(constants.%T, constants.%X) [symbolic = %C (constants.%C.b36)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc15_26.3 [symbolic = %C (constants.%C.b36)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.555) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @bool.as.I.impl: %.loc28_6.2 as %I.ref {
// CHECK:STDOUT:   %bool.as.I.impl.F.decl: %bool.as.I.impl.F.type = fn_decl @bool.as.I.impl.F [concrete = constants.%bool.as.I.impl.F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.928 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.928 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.6d9 = value_param call_param0
// CHECK:STDOUT:     %.loc32_22: type = splice_block %C [concrete = constants.%C.6d9] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %I.facet: %I.type = facet_value bool, (constants.%I.impl_witness.cd6) [concrete = constants.%I.facet.898]
// CHECK:STDOUT:       %.loc32_18: %I.type = converted %Bool.call, %I.facet [concrete = constants.%I.facet.898]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, constants.%I.facet.898) [concrete = constants.%C.6d9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.6d9 = bind_name c, %c.param
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
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc43_18.1: %i32.as.I.impl.F.type.7b2a08.1 = fn_decl @i32.as.I.impl.F.loc43_18.1 [concrete = constants.%i32.as.I.impl.F.4105f2.1] {
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
// CHECK:STDOUT:   %i32.as.I.impl.F.decl.loc43_18.2: %i32.as.I.impl.F.type.7b2a08.2 = fn_decl @i32.as.I.impl.F.loc43_18.2 [concrete = constants.%i32.as.I.impl.F.4105f2.2] {
// CHECK:STDOUT:     %c.patt: %pattern_type.f5b = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.f5b = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.70c = value_param call_param0
// CHECK:STDOUT:     %c: %C.70c = bind_name c, %c.param
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_9.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C.b36 [symbolic = @C.as.Destroy.impl.%C (constants.%C.b36)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T, constants.%X) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.5d3)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.b36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%T.loc15_9.2: type, @C.%X.loc15_19.2: @C.%T.loc15_9.1 (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @C.as.Destroy.impl.Op.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T, %X) [symbolic = %C (constants.%C.b36)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.555)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.7e5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.555)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ce4)]
// CHECK:STDOUT:   %C.loc25_17.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc25_17.1 (constants.%C.4cb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc25_17.1 [symbolic = %pattern_type (constants.%pattern_type.354)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @I.F.%C.loc25_17.1 (%C.4cb));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bool.as.I.impl.F(%c.param: %C.6d9);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc43_18.1(%c.param: %C.6fb);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F.loc43_18.2(%c.param: %C.70c) [thunk @i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %i32.as.I.impl.F.type.7b2a08.1 = name_ref F, @i32.as.I.impl.%i32.as.I.impl.F.decl.loc43_18.1 [concrete = constants.%i32.as.I.impl.F.4105f2.1]
// CHECK:STDOUT:   %c.ref: %C.70c = name_ref c, %c.param
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
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %C => constants.%C.b36
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.5d3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %C => constants.%C.b36
// CHECK:STDOUT:   %ptr => constants.%ptr.555
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self.ce4) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%Self.ce4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.71c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.ce4) {
// CHECK:STDOUT:   %Self => constants.%Self.ce4
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.4cb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.354
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.898) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%I.facet.898
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.71c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.898) {
// CHECK:STDOUT:   %Self => constants.%I.facet.898
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.6d9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.928
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc15_9.1 => type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.274) {
// CHECK:STDOUT:   %Self => constants.%I.facet.274
// CHECK:STDOUT:   %C.loc25_17.1 => constants.%C.70c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f5b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.274) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%I.type
// CHECK:STDOUT:   %X.loc15_19.1 => constants.%I.facet.274
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.71c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2f9
// CHECK:STDOUT: }
// CHECK:STDOUT:

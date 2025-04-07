// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_self_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_self_type_mismatch.carbon

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
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+7]]:8: error: type `C(i32)` of parameter 1 in redeclaration differs from previous parameter type `C(i32 as I)` [RedeclParamDiffersType]
  // CHECK:STDERR:   fn F(c: C(i32));
  // CHECK:STDERR:        ^~~~~~~~~
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE-14]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   fn F(c: C(Self));
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR:
  fn F(c: C(i32));
}

// CHECK:STDOUT: --- fail_self_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.51c: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.b36: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.3c8: %I.type = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.dbb: type = class_type @C, @C(%I.type, %Self) [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.8b6: <witness> = impl_witness (@impl.049.%F.decl) [concrete]
// CHECK:STDOUT:   %I.facet.ba7: %I.type = facet_value bool, (%impl_witness.8b6) [concrete]
// CHECK:STDOUT:   %C.338: type = class_type @C, @C(%I.type, %I.facet.ba7) [concrete]
// CHECK:STDOUT:   %F.type.8ba: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.2cf: %F.type.8ba = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %impl_witness.85b: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %X.patt.7a9: type = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.6fb: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %F.type.066: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.9ec: %F.type.066 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.6ef: %I.type = facet_value %i32, (%impl_witness.85b) [concrete]
// CHECK:STDOUT:   %C.90e: type = class_type @C, @C(%I.type, %I.facet.6ef) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc11_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc11_19.1: @C.%T.loc11_9.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc11_19.2 (constants.%X.patt.51c)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_9.1 [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc11_19.1: @C.%T.loc11_9.2 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc11_19.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.049 [concrete] {} {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc18_6.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc18_6.2: type = converted %bool.make_type, %.loc18_6.1 [concrete = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.049.%F.decl) [concrete = constants.%impl_witness.8b6]
// CHECK:STDOUT:   impl_decl @impl.a9a [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %c.patt: @F.1.%C.loc15_17.1 (%C.dbb) = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: @F.1.%C.loc15_17.1 (%C.dbb) = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @F.1.%C.loc15_17.1 (%C.dbb) = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %C.loc15_17.2 [symbolic = %C.loc15_17.1 (constants.%C.dbb)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %C.loc15_17.2: type = class_type @C, @C(constants.%I.type, constants.%Self) [symbolic = %C.loc15_17.1 (constants.%C.dbb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @F.1.%C.loc15_17.1 (%C.dbb) = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.049: %.loc18_6.2 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.8ba = fn_decl @F.2 [concrete = constants.%F.2cf] {
// CHECK:STDOUT:     %c.patt: %C.338 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.338 = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.338 = value_param call_param0
// CHECK:STDOUT:     %.loc22_22: type = splice_block %C [concrete = constants.%C.338] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %I.facet: %I.type = facet_value bool, (constants.%impl_witness.8b6) [concrete = constants.%I.facet.ba7]
// CHECK:STDOUT:       %.loc22_18: %I.type = converted %bool.make_type, %I.facet [concrete = constants.%I.facet.ba7]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, constants.%I.facet.ba7) [concrete = constants.%C.338]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.338 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.a9a: %i32 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.066 = fn_decl @F.3 [concrete = constants.%F.9ec] {
// CHECK:STDOUT:     %c.patt: %C.6fb = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.6fb = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.6fb = value_param call_param0
// CHECK:STDOUT:     %.loc33: type = splice_block %C [concrete = constants.%C.6fb] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type, constants.%i32) [concrete = constants.%C.6fb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.6fb = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc11_9.1: type, %X.loc11_19.1: @C.%T.loc11_9.2 (%T)) {
// CHECK:STDOUT:   %T.loc11_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %X.loc11_19.2: @C.%T.loc11_9.2 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc11_19.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc11_19.2: @C.%T.loc11_9.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc11_19.2 (constants.%X.patt.51c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.b36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C.loc15_17.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc15_17.1 (constants.%C.dbb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param_patt: @F.1.%C.loc15_17.1 (%C.dbb));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c.param_patt: %C.338);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%c.param_patt: %C.6fb);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T.patt
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X.patt.51c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T.patt
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%Self
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X.patt.3c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C.loc15_17.1 => constants.%C.dbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, @F.1.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.ba7) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T.patt
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%I.facet.ba7
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X.patt.3c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.ba7) {
// CHECK:STDOUT:   %Self => constants.%I.facet.ba7
// CHECK:STDOUT:   %C.loc15_17.1 => constants.%C.338
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc11_9.2 => type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T.patt
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%i32
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X.patt.7a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.6ef) {
// CHECK:STDOUT:   %Self => constants.%I.facet.6ef
// CHECK:STDOUT:   %C.loc15_17.1 => constants.%C.90e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%I.facet.6ef) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T.patt
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%I.facet.6ef
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X.patt.3c8
// CHECK:STDOUT: }
// CHECK:STDOUT:

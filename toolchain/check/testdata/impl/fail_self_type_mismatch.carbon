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
  // This tests that inside the definition of `I`, `Self` has type `I`.

  // Below, `C(Self)` deduces `T` to be the incomplete facet type `I`, which
  // leads to the error.

  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+10]]:11: error: forming value of incomplete type `I` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   fn F(c: C(Self));
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE-9]]:1: note: interface is currently being defined [InterfaceUndefinedWithinDefinition]
  // CHECK:STDERR: interface I {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE-14]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C[T:! type](X:! T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(c: C(Self));
}

impl i32 as I {
  fn F(c: C(i32));
}

// CHECK:STDOUT: --- fail_self_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.b36: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %C.6fb: type = class_type @C, @C(type, %i32) [concrete]
// CHECK:STDOUT:   %F.type.066: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.9ec: %F.type.066 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, %impl_witness [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_9.1, runtime_param<none> [symbolic = %T.patt.loc11_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc11_19.1: @C.%T.loc11_9.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc11_19.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: @C.%T.loc11_9.2 (%T) = value_param_pattern %X.patt.loc11_19.1, runtime_param<none> [symbolic = %X.patt.loc11_19.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc11_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.param: @C.%T.loc11_9.2 (%T) = value_param runtime_param<none>
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_9.1 [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc11_19.1: @C.%T.loc11_9.2 (%T) = bind_symbolic_name X, 1, %X.param [symbolic = %X.loc11_19.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: <error> = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %.loc29: type = splice_block %C [concrete = <error>] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%I.type, <error>) [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: <error> = bind_name c, %c.param
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
// CHECK:STDOUT: impl @impl: %i32 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.066 = fn_decl @F.2 [concrete = constants.%F.9ec] {
// CHECK:STDOUT:     %c.patt: %C.6fb = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.6fb = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.6fb = value_param runtime_param0
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
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc11_9.1: type, %X.loc11_19.1: @C.%T.loc11_9.2 (%T)) {
// CHECK:STDOUT:   %T.loc11_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %X.loc11_19.2: %T = bind_symbolic_name X, 1 [symbolic = %X.loc11_19.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc11_19.2: %T = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc11_19.2 (constants.%X.patt)]
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c.param_patt: %C.6fb);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, <error>) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc11_19.2 => <error>
// CHECK:STDOUT:   %X.patt.loc11_19.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc11_9.2 => type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => type
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%i32
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT: }
// CHECK:STDOUT:

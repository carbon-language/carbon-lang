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
  fn F(c: C(Self));
}

impl i32 as I {
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE+6]]:8: error: type `C(i32)` of parameter 1 in redeclaration differs from previous parameter type `C(i32 as I)` [RedeclParamDiffersType]
  // CHECK:STDERR:   fn F(c: C(i32));
  // CHECK:STDERR:        ^~~~~~~~~
  // CHECK:STDERR: fail_self_type_mismatch.carbon:[[@LINE-7]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   fn F(c: C(Self));
  // CHECK:STDERR:        ^~~~~~~~~~
  fn F(c: C(i32));
}

// CHECK:STDOUT: --- fail_self_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T, %X) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%I.type, %Self) [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.5 [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(type, %i32) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: %I.type = facet_value %i32, %i32 [template]
// CHECK:STDOUT:   %C.5: type = class_type @C, @C(%I.type, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc11_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_9.1, runtime_param<invalid> [symbolic = %T.patt.loc11_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc11_19.1: @C.%T.loc11_9.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc11_19.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: @C.%T.loc11_9.2 (%T) = value_param_pattern %X.patt.loc11_19.1, runtime_param<invalid> [symbolic = %X.patt.loc11_19.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_9.1 [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.param: @C.%T.loc11_9.2 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %X.loc11_19.1: @C.%T.loc11_9.2 (%T) = bind_symbolic_name X, 1, %X.param [symbolic = %X.loc11_19.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc17_6.1: Core.IntLiteral = int_value 32 [template = constants.%.5]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc17_6.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_6.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_6.3: type = converted %int.make_type_signed, %.loc17_6.2 [template = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %c.patt: @F.1.%C.loc14_12.1 (%C.3) = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: @F.1.%C.loc14_12.1 (%C.3) = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %C.loc14_12.2: type = class_type @C, @C(constants.%I.type, constants.%Self) [symbolic = %C.loc14_12.1 (constants.%C.3)]
// CHECK:STDOUT:     %c.param: @F.1.%C.loc14_12.1 (%C.3) = value_param runtime_param0
// CHECK:STDOUT:     %c: @F.1.%C.loc14_12.1 (%C.3) = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc14
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc17_6.3 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %c.patt: %C.4 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.4 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc24: Core.IntLiteral = int_value 32 [template = constants.%.5]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc24) [template = constants.%i32]
// CHECK:STDOUT:     %C: type = class_type @C, @C(type, constants.%i32) [template = constants.%C.4]
// CHECK:STDOUT:     %c.param: %C.4 = value_param runtime_param0
// CHECK:STDOUT:     %c: %C.4 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_15: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc17_15
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
// CHECK:STDOUT:     %.loc11: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C.loc14_12.1: type = class_type @C, @C(constants.%I.type, %Self) [symbolic = %C.loc14_12.1 (constants.%C.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param_patt: @F.1.%C.loc14_12.1 (%C.3));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c.param_patt: %C.4);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%T
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%Self) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%Self
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, @F.1.%Self) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%Self
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C.loc14_12.1 => constants.%C.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type, constants.%i32) {
// CHECK:STDOUT:   %T.loc11_9.2 => type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => type
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%i32
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.6) {
// CHECK:STDOUT:   %Self => constants.%.6
// CHECK:STDOUT:   %C.loc14_12.1 => constants.%C.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.type, constants.%.6) {
// CHECK:STDOUT:   %T.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %T.patt.loc11_9.2 => constants.%I.type
// CHECK:STDOUT:   %X.loc11_19.2 => constants.%.6
// CHECK:STDOUT:   %X.patt.loc11_19.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:

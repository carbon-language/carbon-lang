// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/symbolic.carbon

// Check that we propagate the `symbolic` tag through evaluations.
fn F(T:! type) {
  var u: (T*, const T);
  var v: {.a: T};
  var w: [T; 5];
}

fn G(N:! i32) {
  var k: [i32; N];
}

// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.e07: type = tuple_type (%ptr.251, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.21d: <witness> = require_complete_type %tuple.type.e07 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %require_complete.96f: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array_type.b78: type = array_type %int_5, %T [symbolic]
// CHECK:STDOUT:   %require_complete.43b: <witness> = require_complete_type %array_type.b78 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.3e4: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.52a: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.d18: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.079: %Convert.type.d18 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.3e4, %Convert.079 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.3e4) [symbolic]
// CHECK:STDOUT:   %array_type.352: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.70c: <witness> = require_complete_type %array_type.352 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc12_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_6.1, runtime_param<invalid> [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %N.patt.loc18_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.52a)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc18_6.1, runtime_param<invalid> [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.52a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc18: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc18_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc18_6.2 (constants.%N.3e4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_6.1: type) {
// CHECK:STDOUT:   %T.loc12_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr: type = ptr_type @F.%T.loc12_6.2 (%T) [symbolic = %ptr (constants.%ptr.251)]
// CHECK:STDOUT:   %const: type = const_type @F.%T.loc12_6.2 (%T) [symbolic = %const (constants.%const)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%ptr (%ptr.251), @F.%const (%const)) [symbolic = %tuple.type (constants.%tuple.type.e07)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type @F.%tuple.type (%tuple.type.e07) [symbolic = %require_complete.loc13 (constants.%require_complete.21d)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %struct_type.a (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type @F.%struct_type.a (%struct_type.a) [symbolic = %require_complete.loc14 (constants.%require_complete.96f)]
// CHECK:STDOUT:   %array_type: type = array_type constants.%int_5, @F.%T.loc12_6.2 (%T) [symbolic = %array_type (constants.%array_type.b78)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type @F.%array_type (%array_type.b78) [symbolic = %require_complete.loc15 (constants.%require_complete.43b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.e07) = var u
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.e07) = bind_name u, %u.var
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a (%struct_type.a) = var v
// CHECK:STDOUT:     %v: ref @F.%struct_type.a (%struct_type.a) = bind_name v, %v.var
// CHECK:STDOUT:     %w.var: ref @F.%array_type (%array_type.b78) = var w
// CHECK:STDOUT:     %w: ref @F.%array_type (%array_type.b78) = bind_name w, %w.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc18_6.1: %i32) {
// CHECK:STDOUT:   %N.loc18_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc18_6.2 (constants.%N.3e4)]
// CHECK:STDOUT:   %N.patt.loc18_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.52a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc18_6.2, constants.%Convert.079 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn)]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.loc18_6.2) [symbolic = %int.convert_checked (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type: type = array_type %int.convert_checked, %i32 [symbolic = %array_type (constants.%array_type.352)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%array_type (%array_type.352) [symbolic = %require_complete (constants.%require_complete.70c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %k.var: ref @G.%array_type (%array_type.352) = var k
// CHECK:STDOUT:     %k: ref @G.%array_type (%array_type.352) = bind_name k, %k.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.3e4) {
// CHECK:STDOUT:   %N.loc18_6.2 => constants.%N.3e4
// CHECK:STDOUT:   %N.patt.loc18_6.2 => constants.%N.3e4
// CHECK:STDOUT: }
// CHECK:STDOUT:

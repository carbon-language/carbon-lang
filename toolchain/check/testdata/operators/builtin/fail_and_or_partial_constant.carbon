// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon

// --- fail_non_constant_result.carbon

package FailNonConstantResult;

fn PartialConstant(x: bool) {
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var a: if true and x then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: if true and x then bool else ();
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var b: if false or x then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var b: if false or x then bool else ();
}

// --- fail_despite_known_result.carbon

package FailDespiteKnownResult;

fn KnownValueButNonConstantCondition(x: bool) {
  // We choose not to give these cases constant values, even though we could,
  // because they notionally involve evaluating a non-constant condition.
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var c: if x and false then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var c: if x and false then bool else ();
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+3]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var d: if x or true then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  var d: if x or true then bool else ();
}

// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [template]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [template = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc4_23.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc4_23.3: type = converted %bool.make_type, %.loc4_23.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref <error> = var b
// CHECK:STDOUT:   %b: ref <error> = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_despite_known_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.type: type = fn_type @KnownValueButNonConstantCondition [template]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition: %KnownValueButNonConstantCondition.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .KnownValueButNonConstantCondition = %KnownValueButNonConstantCondition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.decl: %KnownValueButNonConstantCondition.type = fn_decl @KnownValueButNonConstantCondition [template = constants.%KnownValueButNonConstantCondition] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_41.1: type = splice_block %.loc4_41.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc4_41.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc4_41.3: type = converted %bool.make_type, %.loc4_41.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @KnownValueButNonConstantCondition(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref <error> = var d
// CHECK:STDOUT:   %d: ref <error> = bind_name d, %d.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

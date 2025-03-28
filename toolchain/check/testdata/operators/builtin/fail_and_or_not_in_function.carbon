// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// TODO: Short-circuit operators should be permitted outside functions.
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+10]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+6]]:11: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:           ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var and_: F(true and true);

// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+12]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+8]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+4]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_val: bool = true and true;

// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+10]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+6]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var or_: F(true or true);

// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+12]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+8]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~~~~~~
// CHECK:STDERR:
var or_val: bool = true or true;

// CHECK:STDOUT: --- fail_and_or_not_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .and_ = %and_
// CHECK:STDOUT:     .and_val = %and_val
// CHECK:STDOUT:     .or_ = %or_
// CHECK:STDOUT:     .or_val = %or_val
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc12_9.1: type = splice_block %.loc12_9.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc12_9.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc12_9.3: type = converted %bool.make_type, %.loc12_9.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %and_.patt: <error> = binding_pattern and_
// CHECK:STDOUT:     %.loc27: <error> = var_pattern %and_.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %and_.var: ref <error> = var and_
// CHECK:STDOUT:   %and_: <error> = bind_name and_, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %and_val.patt: bool = binding_pattern and_val
// CHECK:STDOUT:     %.loc41_1: bool = var_pattern %and_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %and_val.var: ref bool = var and_val
// CHECK:STDOUT:   %.loc41_14.1: type = splice_block %.loc41_14.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc41: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc41_14.2: type = value_of_initializer %bool.make_type.loc41 [concrete = bool]
// CHECK:STDOUT:     %.loc41_14.3: type = converted %bool.make_type.loc41, %.loc41_14.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %and_val: ref bool = bind_name and_val, %and_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %or_.patt: <error> = binding_pattern or_
// CHECK:STDOUT:     %.loc53: <error> = var_pattern %or_.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %or_.var: ref <error> = var or_
// CHECK:STDOUT:   %or_: <error> = bind_name or_, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %or_val.patt: bool = binding_pattern or_val
// CHECK:STDOUT:     %.loc67_1: bool = var_pattern %or_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %or_val.var: ref bool = var or_val
// CHECK:STDOUT:   %.loc67_13.1: type = splice_block %.loc67_13.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc67: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc67_13.2: type = value_of_initializer %bool.make_type.loc67 [concrete = bool]
// CHECK:STDOUT:     %.loc67_13.3: type = converted %bool.make_type.loc67, %.loc67_13.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %or_val: ref bool = bind_name or_val, %or_val.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param_patt: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:   %.loc13_24.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:   %.loc13_24.2: type = converted %float.make_type, %.loc13_24.1 [concrete = f64]
// CHECK:STDOUT:   br !if.expr.result(%.loc13_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc13_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc13_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true br !and.rhs else br !and.result(%false)
// CHECK:STDOUT: }
// CHECK:STDOUT:

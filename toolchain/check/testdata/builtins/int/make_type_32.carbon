// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/make_type_32.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/make_type_32.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: types.carbon:[[@LINE+4]]:20: error: unknown builtin function name "int.make_type_32" [UnknownBuiltinFunctionName]
// CHECK:STDERR: fn Int() -> type = "int.make_type_32";
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Int() -> type = "int.make_type_32";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: use_types.carbon:[[@LINE+3]]:8: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var i: Int() = 0;
// CHECK:STDERR:        ^~~~~
var i: Int() = 0;

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int() -> type;
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %Int.type = import_ref Main//types, inst+7, loaded [template = constants.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = imports.%import_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .i = %i
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%import_ref [template = constants.%Int]
// CHECK:STDOUT:   %Int.call: init type = call %Int.ref()
// CHECK:STDOUT:   %.loc9_12.1: type = value_of_initializer %Int.call
// CHECK:STDOUT:   %.loc9_12.2: type = converted %Int.call, %.loc9_12.1
// CHECK:STDOUT:   %i.var: ref <error> = var i
// CHECK:STDOUT:   %i: ref <error> = bind_name i, %i.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int() -> type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%i.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

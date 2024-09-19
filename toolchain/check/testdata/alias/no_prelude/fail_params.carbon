// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+7]]:8: error: `alias` declaration cannot have parameters
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_params.carbon:[[@LINE+3]]:21: error: alias initializer must be a name reference
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:                     ^~
alias A(T:! type) = T*;

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.loc18_9.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:   %T.loc18_9.2: type = bind_symbolic_name T 0, %T.loc18_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T.loc18_9.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc18: type = ptr_type %T [symbolic = constants.%.1]
// CHECK:STDOUT:   %A: <error> = bind_alias A, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:

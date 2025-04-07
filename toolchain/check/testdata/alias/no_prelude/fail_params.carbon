// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+8]]:8: error: `alias` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:21: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:                     ^~
// CHECK:STDERR:
alias A(T:! type) = T*;

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.ref [symbolic = constants.%ptr]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = constants.%T]
// CHECK:STDOUT:   %A: <error> = bind_alias A, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_var_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_var_param.carbon

// A `var` parameter is not constant, so it is not allowed as a parameter of a
// generic type.
// CHECK:STDERR: fail_var_param.carbon:[[@LINE+3]]:13-18: error: parameters of generic types must be constant [GenericParamMustBeConstant]
// CHECK:STDERR: fail_var_param.carbon:[[@LINE+2]]:13-18: note: `var` parameters are runtime [VarParamIsRuntime]
// CHECK:STDERR:
class C(var x: i32);

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_no_extend.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_no_extend.carbon

base class B {}

class C {
  // CHECK:STDERR: fail_base_no_extend.carbon:[[@LINE+4]]:3: error: missing `extend` before `base` declaration [BaseMissingExtend]
  // CHECK:STDERR:   base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  base: B;
}

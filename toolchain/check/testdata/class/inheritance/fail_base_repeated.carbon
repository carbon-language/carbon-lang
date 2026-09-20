// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_base_repeated.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_base_repeated.carbon

base class B1 {}
base class B2 {}

class C {
  extend base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+3]]:3-18: error: multiple `base` declarations in class; multiple inheritance is not permitted [BaseDeclRepeated]
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-2]]:3-18: note: previous `base` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:
  extend base: B2;
}

class D {
  // TODO: Consider adding a custom diagnostic for this case.
  extend base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+3]]:3-18: error: multiple `base` declarations in class; multiple inheritance is not permitted [BaseDeclRepeated]
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-2]]:3-18: note: previous `base` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:
  extend base: B1;
}

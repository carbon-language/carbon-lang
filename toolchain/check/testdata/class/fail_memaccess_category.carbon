// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_memaccess_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_memaccess_category.carbon

class A {
  fn F[addr self: A*]();
}

class B {
  var a: A;
}

fn F(s: {.a: A}, b: B) {
  // `s` has only a value representation, so this must be invalid.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-23]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  b.a.F();
}

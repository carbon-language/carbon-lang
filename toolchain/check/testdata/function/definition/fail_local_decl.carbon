// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/fail_local_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/fail_local_decl.carbon

// --- fail_virtual.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F();
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:3: error: `override` not allowed; requires class scope [ModifierRequiresClass]
  // CHECK:STDERR:   override fn G();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  override fn G();
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:3: error: `virtual` not allowed; requires class scope [ModifierRequiresClass]
  // CHECK:STDERR:   virtual fn H();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual fn H();
}

// --- fail_access.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_access.carbon:[[@LINE+4]]:3: error: `private` not allowed; requires class or file scope [ModifierPrivateNotAllowed]
  // CHECK:STDERR:   private var unused v: {} = {};
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private var unused v: {} = {};
  // CHECK:STDERR: fail_access.carbon:[[@LINE+4]]:3: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
  // CHECK:STDERR:   protected var unused w: {} = {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  protected var unused w: {} = {};
}

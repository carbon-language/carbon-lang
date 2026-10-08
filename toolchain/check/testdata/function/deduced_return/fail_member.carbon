// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/fail_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/fail_member.carbon

// --- fail_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_interface.carbon:[[@LINE+4]]:10: error: interface member with deduced return type [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:   fn F() -> auto;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  fn F() -> auto;
}

// --- fail_interface_default.carbon
library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_interface_default.carbon:[[@LINE+4]]:18: error: interface member with deduced return type [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:   default fn F() -> auto { return 0 as i32; }
  // CHECK:STDERR:                  ^~~~~~~
  // CHECK:STDERR:
  default fn F() -> auto { return 0 as i32; }
}

// --- fail_virtual.carbon
library "[[@TEST_NAME]]";

base class B {
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:29: error: virtual function with deduced return type [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:   virtual fn F(unused self) -> auto { return 0 as i32; }
  // CHECK:STDERR:                             ^~~~~~~
  // CHECK:STDERR:
  virtual fn F(unused self) -> auto { return 0 as i32; }
}

// --- fail_abstract.carbon
library "[[@TEST_NAME]]";

abstract class A {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+4]]:23: error: virtual function with deduced return type [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:   abstract fn F(self) -> auto;
  // CHECK:STDERR:                       ^~~~~~~
  // CHECK:STDERR:
  abstract fn F(self) -> auto;
}

// --- fail_override.carbon
library "[[@TEST_NAME]]";

base class B {
  virtual fn F(unused self) -> i32 { return 0; }
}

class D {
  extend base: B;
  // CHECK:STDERR: fail_override.carbon:[[@LINE+4]]:30: error: virtual function with deduced return type [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:   override fn F(unused self) -> auto { return 0 as i32; }
  // CHECK:STDERR:                              ^~~~~~~
  // CHECK:STDERR:
  override fn F(unused self) -> auto { return 0 as i32; }
}

// --- fail_todo_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F() -> i32;
}

class C {}

impl C as I {
  // CHECK:STDERR: fail_todo_impl.carbon:[[@LINE+4]]:10: error: semantics TODO: `deduced return type for `impl` member` [SemanticsTodo]
  // CHECK:STDERR:   fn F() -> auto { return 0 as i32; }
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  fn F() -> auto { return 0 as i32; }
}

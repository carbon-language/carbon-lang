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
  // CHECK:STDERR:        ╭─┤ fail_interface.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   fn F() -> auto;
  // CHECK:STDERR:        ·          ━━━┳━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: interface member with deduced return type
  // CHECK:STDERR:        ·                ┃        [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  fn F() -> auto;
}

// --- fail_interface_default.carbon
library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR:        ╭─┤ fail_interface_default.carbon:[[@LINE+7]]:18
  // CHECK:STDERR: ->     │   default fn F() -> auto { return 0 as i32; }
  // CHECK:STDERR:        ·                  ━━━┳━━━
  // CHECK:STDERR:        ·                     ┗━━┫ error: interface member with deduced return type
  // CHECK:STDERR:        ·                        ┃        [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  default fn F() -> auto { return 0 as i32; }
}

// --- fail_virtual.carbon
library "[[@TEST_NAME]]";

base class B {
  // CHECK:STDERR:        ╭─┤ fail_virtual.carbon:[[@LINE+8]]:29
  // CHECK:STDERR: ->     │   virtual fn F(unused self) -> auto { return 0 as i32; }
  // CHECK:STDERR:        ·                             ┳━━━━━━
  // CHECK:STDERR:        ·                             ┗━━┫ error: virtual function with deduced return
  // CHECK:STDERR:        ·                                ┃        type
  // CHECK:STDERR:        ·                                ┃        [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  virtual fn F(unused self) -> auto { return 0 as i32; }
}

// --- fail_abstract.carbon
library "[[@TEST_NAME]]";

abstract class A {
  // CHECK:STDERR:        ╭─┤ fail_abstract.carbon:[[@LINE+7]]:23
  // CHECK:STDERR: ->     │   abstract fn F(self) -> auto;
  // CHECK:STDERR:        ·                       ━━━┳━━━
  // CHECK:STDERR:        ·                          ┗━━┫ error: virtual function with deduced return type
  // CHECK:STDERR:        ·                             ┃        [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_override.carbon:[[@LINE+8]]:30
  // CHECK:STDERR: ->     │   override fn F(unused self) -> auto { return 0 as i32; }
  // CHECK:STDERR:        ·                              ┳━━━━━━
  // CHECK:STDERR:        ·                              ┗━━┫ error: virtual function with deduced return
  // CHECK:STDERR:        ·                                 ┃        type
  // CHECK:STDERR:        ·                                 ┃        [DeducedReturnTypeNotAllowed]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_todo_impl.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   fn F() -> auto { return 0 as i32; }
  // CHECK:STDERR:        ·          ━━━┳━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: semantics TODO: `deduced return type for `impl` member`
  // CHECK:STDERR:        ·                ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  fn F() -> auto { return 0 as i32; }
}

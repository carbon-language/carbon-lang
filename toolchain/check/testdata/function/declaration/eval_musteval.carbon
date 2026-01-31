// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/eval_musteval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/eval_musteval.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon

// --- basic.carbon

fn Basic() -> i32 { return 0; }
eval fn Eval() -> i32 { return 0; }
musteval fn MustEval() -> i32 { return 0; }

// --- member.carbon

library "[[@TEST_NAME]]";

class C {
  eval fn Eval() {}
  musteval fn MustEval() {}
}

// --- private.carbon

library "[[@TEST_NAME]]";

private eval fn Eval() -> i32 { return 0; }
private musteval fn MustEval() -> i32 { return 0; }

// --- fail_conflict.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+7]]:6: error: `musteval` not allowed on declaration with `eval` [ModifierNotAllowedWith]
// CHECK:STDERR: eval musteval fn Mixed();
// CHECK:STDERR:      ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:1: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR: eval musteval fn Mixed();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval musteval fn Mixed();

// --- fail_order.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:6: error: `export` must appear before `eval` [ModifierMustAppearBefore]
// CHECK:STDERR: eval export fn ExportEval();
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR: fail_order.carbon:[[@LINE+4]]:1: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR: eval export fn ExportEval();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval export fn ExportEval();

// CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:6: error: `private` must appear before `eval` [ModifierMustAppearBefore]
// CHECK:STDERR: eval private fn PrivateEval();
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: fail_order.carbon:[[@LINE+4]]:1: note: `eval` previously appeared here [ModifierPrevious]
// CHECK:STDERR: eval private fn PrivateEval();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval private fn PrivateEval();

class C {
  // CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:8: error: `virtual` must appear before `eval` [ModifierMustAppearBefore]
  // CHECK:STDERR:   eval virtual fn VirtualEval();
  // CHECK:STDERR:        ^~~~~~~
  // CHECK:STDERR: fail_order.carbon:[[@LINE+4]]:3: note: `eval` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   eval virtual fn VirtualEval();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  eval virtual fn VirtualEval();
}

// --- fail_not_fn.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_not_fn.carbon:[[@LINE+4]]:1: error: `eval` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: eval var x: i32;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval var x: i32;

// CHECK:STDERR: fail_not_fn.carbon:[[@LINE+4]]:1: error: `eval` not allowed on `class` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: eval class C {};
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
eval class C {};

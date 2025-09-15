// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/base.carbon

// TODO: Tests marked as `fail_todo_5891_` to fixed as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.

// --- structs.h

struct A {
  int a;

  void f() {}
};

struct B : A {
  int b;
};

// --- convert.carbon

library "[[@TEST_NAME]]";

import Cpp library "structs.h";

fn ConvertPtr(p: Cpp.B*) -> Cpp.A* {
  return p;
}

fn AcceptVal(a: Cpp.A);

fn ConvertVal(b: Cpp.B) {
  AcceptVal(b);
}

// --- access_field.carbon

library "[[@TEST_NAME]]";

import Cpp library "structs.h";

fn AccessVal(b: Cpp.B) -> i32 {
  return b.a;
}

// --- fail_todo_5891_call_method.carbon

library "[[@TEST_NAME]]";

import Cpp library "structs.h";

fn Call(b: Cpp.B*) {
  // CHECK:STDERR: fail_todo_5891_call_method.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   b->f();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_todo_5891_call_method.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  b->f();
}

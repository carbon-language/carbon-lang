// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-std=c++26
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/method.carbon

// TODO: Tests marked as `fail_todo_5891_` to fixed as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.

// --- methods.h

struct A {
  int by_val() const { return n; }
  int by_ref() { return n; }
  int n;
};

// --- fail_todo_5891_call_by_val.carbon

library "[[@TEST_NAME]]";

import Cpp library "methods.h";

fn UseVal(a: Cpp.A) -> i32 {
  // CHECK:STDERR: fail_todo_5891_call_by_val.carbon:[[@LINE+5]]:10: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   return a.by_val();
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_call_by_val.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  return a.by_val();
}

// --- fail_todo_5891_call_by_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "methods.h";

fn UseVal(a: Cpp.A*) -> i32 {
  // CHECK:STDERR: fail_todo_5891_call_by_ref.carbon:[[@LINE+5]]:10: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   return a->by_ref();
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_call_by_ref.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  return a->by_ref();
}

// --- thunk.h

struct NeedThunk {
  void Implicit(signed char c) const;
  void Explicit(this NeedThunk, signed char c);
};

// --- fail_todo_5891_call_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp library "thunk.h";

fn Call(n: Cpp.NeedThunk) {
  // CHECK:STDERR: fail_todo_5891_call_thunk.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   n.Implicit(1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_call_thunk.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  n.Implicit(1);
  // CHECK:STDERR: fail_todo_5891_call_thunk.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   n.Explicit(1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_call_thunk.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  n.Explicit(1);
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/default_values.carbon

// --- generic_fn_concrete_default.carbon
library "[[@TEST_NAME]]";
import Cpp;

interface Z {};

class D { adapt {}; }
impl D as Z {}
fn MakeD() -> D { return {} as D; }

class E { adapt {}; }

class C(T: Z) {
  var e: E;
}

fn Make[T: Z](unused t: T, e: E = {} as E) -> C(T) {
  return {.e = e} as C(T);
}

inline Cpp '''
void CallMake() {
  Carbon::Make(Carbon::MakeD());
}
''';

// --- other.carbon
library "[[@TEST_NAME]]";

fn F(unused i: i32 = 0) {}

// --- import.carbon
library "[[@TEST_NAME]]";

import library "other";
import Cpp inline '''
void CallF() {
  Carbon::F();
}
''';

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/layout.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/layout.carbon

// --- simple.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {
  var a: i32;
  var b: i32;
  var c: i32;
}

inline Cpp '''c++
static_assert(sizeof(Carbon::C) == 12);
static_assert(alignof(Carbon::C) == 4);
''';

// --- packed.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {
  var a: i32*;
  var b: i32;
}

class D {
  var c: C;
  var d: i32;
}

inline Cpp '''c++
static_assert(sizeof(Carbon::C) == 12);
static_assert(alignof(Carbon::C) == 8);
static_assert(sizeof(Carbon::D) == 16);
static_assert(alignof(Carbon::D) == 8);
''';

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

class ThreeInts {
  var a: i32;
  var b: i32;
  var c: i32;
}

inline Cpp '''c++
static_assert(sizeof(Carbon::ThreeInts) == 12);
static_assert(alignof(Carbon::ThreeInts) == 4);
''';

// --- packed.carbon
library "[[@TEST_NAME]]";

import Cpp;

class PointerInt {
  var a: i32*;
  var b: i32;
}

class PointerIntInt {
  var c: PointerInt;
  var d: i32;
}

inline Cpp '''c++
static_assert(sizeof(Carbon::PointerInt) == 12);
static_assert(alignof(Carbon::PointerInt) == 8);
static_assert(sizeof(Carbon::PointerIntInt) == 16);
static_assert(alignof(Carbon::PointerIntInt) == 8);
''';

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --clang-arg=-fmodules --clang-arg=-fmodules-cache-path=%t.cache
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/modules/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/modules/import.carbon

// --- module.modulemap

module A {
  header "a.h"
  export *
}

module B {
  header "b.h"
  export *
}

// --- a.h

struct A {};

A *_Nonnull makeA();

// --- b.h

#include "a.h"

struct B {};

B *_Nonnull makeB(A *_Nonnull);

// --- use_a_b.carbon

library "[[@TEST_NAME]]";

import Cpp library "a.h";
import Cpp library "b.h";

fn Go() -> Cpp.B* {
  return Cpp.makeB(Cpp.makeA());
}

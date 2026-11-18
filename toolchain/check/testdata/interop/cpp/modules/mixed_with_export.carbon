// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --clang-arg=-fmodules --clang-arg=-fmodules-cache-path=%t.cache
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/modules/mixed_with_export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/modules/mixed_with_export.carbon

// --- module.modulemap

module A {
  header "a.h"
  export *
}

// --- a.h

struct A { void F(); };

struct C { void H(); };

namespace N { void I(); }

// --- b.carbon

library "[[@TEST_NAME]]";

import Cpp library "a.h";

class B {
  extend base: Cpp.A;
  fn G(self);
}

inline Cpp '''
Carbon::B b;
A *a = &b;

void call(C *c) {
  a->F();

  b.G();

  c->H();

  N::I();
}
''';

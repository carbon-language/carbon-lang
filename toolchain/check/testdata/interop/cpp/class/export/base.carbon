// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/base.carbon

// --- derived_to_base.carbon
library "[[@TEST_NAME]]";

import Cpp;

base class A {}

class B {
  extend base: A;
}

inline Cpp '''
Carbon::A* Convert(Carbon::B* p) {
  return p;
}
''';

// --- name_lookup.carbon
library "[[@TEST_NAME]]";

import Cpp;

base class A {
  fn F();
  fn G(n: i32);
}

class B {
  extend base: A;
  fn G();
}

inline Cpp '''
void Qualified() {
  Carbon::B::F();
  Carbon::B::G();
}

void MemberAccess(Carbon::A *a, Carbon::B *b) {
  // TODO: These cause a crash.
  // b->F();
  // b->G();
}
''';

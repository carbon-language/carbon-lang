// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/override.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/override.carbon

// --- const_ref_param.carbon
library "[[@TEST_NAME]]";
import Cpp;

inline Cpp '''
struct X {};

struct A {
  virtual void f(const X& x) = 0;
};
''';

class B {
  extend base: Cpp.A;
  override fn f(unused self, unused x: Cpp.X) {}
}

inline Cpp '''
Carbon::B b;
''';

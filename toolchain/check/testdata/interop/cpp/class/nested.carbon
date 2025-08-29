// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/nested.carbon

// --- import_nested_class.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct A {
  struct B {
    struct C {
      int abc;
    };
  };
  B::C c;
};
''';

fn F(p: Cpp.A*) -> i32 {
  return p->c.abc;
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/deduced_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/deduced_return.carbon

// --- non_generic.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn F() -> auto { return 1 as i32; }

inline Cpp '''
int CallF() { return Carbon::F(); }
''';

// --- generic.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn PointerId[T: type](p: T*) -> auto { return p; }

inline Cpp '''
int CallPointerId() {
  int x = 0;
  int* _Nonnull p = &x;
  return *Carbon::PointerId(p);
}
''';

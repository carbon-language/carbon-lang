// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/const/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/const/access.carbon

// --- const_member_access.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {
  int x;
};

const S& Get();
''';

fn F() -> i32 {
  // CHECK:STDERR: const_member_access.carbon:[[@LINE+4]]:10: error: type `const Cpp.S` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   return Cpp.Get().x;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  return Cpp.Get().x;
}

// --- fail_const_member_assign.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {
  int x;
};

const S& Get();
''';

fn F() {
  // CHECK:STDERR: fail_const_member_assign.carbon:[[@LINE+4]]:3: error: type `const Cpp.S` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   Cpp.Get().x = 123;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Get().x = 123;
}

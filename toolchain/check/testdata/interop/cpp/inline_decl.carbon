// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/inline_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/inline_decl.carbon

// --- inline_decl.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
void func1() {}
''';

fn Run1() {
  Cpp.func1();
}

inline Cpp '''
void func2() {
  Carbon::Run1();
}
''';

fn Run2() {
  Cpp.func2();
}

// --- fail_inline_no_import.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_inline_no_import.carbon:[[@LINE+4]]:8: error: name `Cpp` not found [NameNotFound]
// CHECK:STDERR: inline Cpp "";
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
inline Cpp "";

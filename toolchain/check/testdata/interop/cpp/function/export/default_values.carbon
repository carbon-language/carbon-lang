// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/default_values.carbon

// --- no_package.carbon
library "[[@TEST_NAME]]";

fn F(unused i: i32 = 0) {}

// --- fail_todo_cpp_library_import.carbon
library "[[@TEST_NAME]]";

// It seems that we cannot access an imported Carbon function in Cpp without it
// living in a package. Repair this and then re-enable this test.

import Cpp inline '''
void CallF() {
  // CHECK:STDERR: fail_todo_cpp_library_import.carbon:[[@LINE+2]]:11: error: no member named 'F' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::F();
}
''';

// --- other_function.carbon
package OtherFunction;
fn F(unused i: i32 = 0) {}

// --- basic.carbon
library "[[@TEST_NAME]]";

import OtherFunction;
import Cpp inline '''
void CallF() {
  Carbon::OtherFunction::F();
}
''';

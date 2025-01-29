// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/no_prelude/multiple_imports.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/no_prelude/multiple_imports.carbon

// --- file1.h

void foo1();

// --- file2.h

void foo1();

// --- multiple_imports.carbon

library "[[@TEST_NAME]]";

import Cpp library "file1.h";
import Cpp library "file2.h";

// CHECK:STDOUT: --- multiple_imports.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:

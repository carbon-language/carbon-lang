// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/reverse/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/reverse/simple.carbon

// --- fail_simple.carbon

library "[[@TEST_NAME]]";

// Demonstrate that the 'Carbon' namespace has been injected into the C++ compilation.

import Cpp inline '''
// CHECK:STDERR: fail_simple.carbon:[[@LINE+4]]:9: error: no type named 'NonExistent' in namespace 'Carbon' [CppInteropParseError]
// CHECK:STDERR:    11 | Carbon::NonExistent v;
// CHECK:STDERR:       | ~~~~~~~~^
// CHECK:STDERR:
Carbon::NonExistent v;
''';


// --- other.carbon
package Other;
// --- namespace.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
namespace X = Carbon::Other;
''';

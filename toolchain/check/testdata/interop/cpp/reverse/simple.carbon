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

// --- other.carbon
package Other;
namespace Nested;
namespace Nested.Again;
// --- namespace.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
namespace X = Carbon::Other;
namespace Y = Carbon::Other::Nested;
namespace Z = Carbon::Other::Nested::Again;
''';

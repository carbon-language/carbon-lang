// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/nested_in_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/nested_in_generic.carbon

// --- generic_with_nested.carbon
package P library "generic_with_nested";

class X {};

class C(T: type) {
  class Nested {
    var x: X;
  }
}

// --- import.carbon
library "[[@TEST_NAME]]";

import P library "generic_with_nested";

alias C = P.C;

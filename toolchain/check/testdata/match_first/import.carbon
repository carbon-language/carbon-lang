// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ^ARGS: -v compile %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/match_first/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/match_first/import.carbon

// --- for_import.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1:! type; }

final match_first {
  impl () as Z where .Z1 = () {}
}

// --- import.carbon
library "[[@TEST_NAME]]";

import library "for_import";

fn F() {
  let _: ().(Z.Z1) = ();
}

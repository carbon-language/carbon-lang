// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/early_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/early_impls.carbon

// Tests that `.X impls Y` in a facet type is available in later constraints of
// the same facet type.

// --- member_access_uses_impls_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}
interface Y {
  let Y1:! type;
}

fn F(_:! Z where .Z1 impls Y and .Z2 = .Z1.(Y.Y1)) {
}

class C(T:! type);
constraint X {
  require C(Self) impls Y;
}

fn G(_:! Z where .Z1 impls X and .Z2 = C(.Z1).(Y.Y1)) {
}

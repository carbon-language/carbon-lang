// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/period_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/period_self.carbon

// --- period_self_converted_in_impl_as_decl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

interface Y {}
impl forall [T: Z] T as Y {}

class C(T: Z & Y) {}

class D;
// This forms the specific `C(.Self)` where `.Self` is of type `Z`
// but we need type `Z & Y` so a conversion takes place. This forms
// a FacetValue with two witnesses. The witness for `Y` is symbolic
// (due to .Self) and points to the impl above. But the witness for
// `Z` is not symbolic and points to the impl being declared here.
impl D as Z where .Z1 = C(.Self) {}

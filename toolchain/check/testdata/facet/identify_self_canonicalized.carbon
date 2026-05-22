// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/identify_self_canonicalized.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/identify_self_canonicalized.carbon

// --- identify_self_canonicalized.carbon
library "[[@TEST_NAME]]";

interface Z {}

constraint W {
  extend require impls Z;
}

interface Y {}

// T is a facet value (a SymbolicBinding), which is converted to `type`. The
// identified facet type finds `(T as type) impls Z` from the top level facet
// type. It also finds `T as Z` through `W`. It's possible for these to disagree
// on the self and thus present as two different required interfaces in the
// identified facet type. Instead we should only get a single interface with the
// canonical `T` facet value as the self value.

impl forall [T:! Y] T as W & Z {}

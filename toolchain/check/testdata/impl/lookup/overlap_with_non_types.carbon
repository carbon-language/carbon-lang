// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/overlap_with_non_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/overlap_with_non_types.carbon

// --- class_with_different_values.carbon
library "[[@TEST_NAME]]";


interface Z {}

class C(B:! bool);

impl C(true) as Z {}
impl C(false) as Z {}

// --- array_with_different_bounds.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;

impl array(C, 1) as Z {}
impl array(C, 2) as Z {}

// --- tuple_vs_pointer.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;

impl (C, ) as Z {}
impl C* as Z {}

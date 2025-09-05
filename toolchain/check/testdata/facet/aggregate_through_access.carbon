// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/aggregate_through_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/aggregate_through_access.carbon

// --- fail_todo_tuple_access_through_witness.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// CHECK:STDERR: fail_todo_tuple_access_through_witness.carbon:[[@LINE+4]]:34: error: type `type` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
// CHECK:STDERR: fn F(T:! Z where .X = ({}, )) -> T.X.0 {
// CHECK:STDERR:                                  ^~~~~
// CHECK:STDERR:
fn F(T:! Z where .X = ({}, )) -> T.X.0 {
  return {};
}

// --- fail_todo_struct_access_through_witness.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// CHECK:STDERR: fail_todo_struct_access_through_witness.carbon:[[@LINE+4]]:36: error: type `type` does not support qualified expressions [QualifiedExprUnsupported]
// CHECK:STDERR: fn F(T:! Z where .X = {.t: ()}) -> T.X.t {
// CHECK:STDERR:                                    ^~~~~
// CHECK:STDERR:
fn F(T:! Z where .X = {.t: ()}) -> T.X.t {
  return ();
}

// --- fail_todo_array_access_through_witness.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// CHECK:STDERR: fail_todo_array_access_through_witness.carbon:[[@LINE+4]]:40: error: type `type` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: fn F(T:! Z where .X = array({}, 1)) -> T.X[0] {
// CHECK:STDERR:                                        ^~~~~~
// CHECK:STDERR:
fn F(T:! Z where .X = array({}, 1)) -> T.X[0] {
  return {};
}

// --- impl_access_through_witness.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X1:! type;
}

interface Y {
  let X2:! type;
}

class C;
impl C as Y where .X2 = {} {}

fn F(T:! Z where .X1 = C) -> T.X1.(Y.X2) {
  return {};
}

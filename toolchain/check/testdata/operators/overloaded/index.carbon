// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/index.carbon

// --- core_wrong_index_with.carbon

package Core library "[[@TEST_NAME]]";

class IndexWith {}

// --- fail_wrong_index_with.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_index_with";

// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+2]]:10-13: error: value of type `type` is not callable [CallToNonCallable]
// CHECK:STDERR:
fn F() { 0[1]; }
// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+2]]:1: error: missing `//@dump-sem-ir-begin` to match `//@dump-sem-ir-end` [DumpSemIRRangeMissingBegin]
// CHECK:STDERR:
//@dump-sem-ir-end

// --- fail_missing_index_with.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+2]]:10-13: error: name `Core.IndexWith` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR:
fn F() { 0[1]; }
// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+2]]:1: error: missing `//@dump-sem-ir-begin` to match `//@dump-sem-ir-end` [DumpSemIRRangeMissingBegin]
// CHECK:STDERR:
//@dump-sem-ir-end

// --- core_wrong_arg_count.carbon

package Core library "[[@TEST_NAME]]";

interface IndexWith(SubscriptType: type, ElementType: type) {
  fn At(self, subscript: SubscriptType) -> ElementType;
}


// --- fail_wrong_arg_count.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_arg_count";

class C {}

impl C as Core.IndexWith((), ()) {
  fn At(unused self, unused subscript: ()) -> () {
    return ();
  }
}

fn F() {
  let c: C = {};
  // CHECK:STDERR: fail_wrong_arg_count.carbon:[[@LINE+3]]:3-7: error: 1 argument passed to generic interface expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR: core_wrong_arg_count.carbon:4:1-61: note: calling generic interface declared here [InCallToEntity]
  // CHECK:STDERR:
  c[()];
  // CHECK:STDERR: fail_wrong_arg_count.carbon:[[@LINE+2]]:3: error: missing `//@dump-sem-ir-begin` to match `//@dump-sem-ir-end` [DumpSemIRRangeMissingBegin]
  // CHECK:STDERR:
  //@dump-sem-ir-end
}



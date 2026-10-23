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

class IndirectIndexWith {}

// --- fail_wrong_index_with.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_index_with";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+4]]:10-13: error: value of type `type` is not callable [CallToNonCallable]
// CHECK:STDERR:
// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+2]]:10-13: error: value of type `type` is not callable [CallToNonCallable]
// CHECK:STDERR:
fn F() { 0[1]; }
//@dump-sem-ir-end

// --- fail_missing_index_with.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+4]]:10-13: error: name `Core.IndirectIndexWith` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+2]]:10-13: error: name `Core.IndexWith` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR:
fn F() { 0[1]; }
//@dump-sem-ir-end

// --- core_wrong_arg_count.carbon

package Core library "[[@TEST_NAME]]";

interface IndexWith(SubscriptType: type, ElementType: type) {
  fn At(self, subscript: SubscriptType) -> ElementType;
}

interface IndirectIndexWith(SubscriptType: type) {
  let ElementType: type;
  fn Ref(self, subscript: SubscriptType) -> ref ElementType;
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


// CHECK:STDOUT: --- fail_wrong_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

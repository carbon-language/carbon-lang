// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/imported_type_completeness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/imported_type_completeness.carbon

// --- a.carbon
package A;

class C(T:! type) {
  // CHECK:STDERR: a.carbon:[[@LINE+4]]:13: error: member name `Int` not found in `Core` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   fn F() -> Core.Int(123) {
  // CHECK:STDERR:             ^~~~~~~~
  // CHECK:STDERR:
  fn F() -> Core.Int(123) {
    return 123;
  }
}

fn G(T:! type) {
  ({} as C(T)).F();
}

fn H() { G({}); }

// --- b.carbon
package B;
import A;

fn H() {
  // OK, even though nothing in the compilation of this file requires i123 to be
  // complete.
  A.G({});
}

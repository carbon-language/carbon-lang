// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/name_poisoning.carbon

// --- using_poisoned_name_in_impl.carbon

library "[[@TEST_NAME]]";

interface I {};

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F1(x:! I);

class N.C {
  extend impl as I {
  }
}

// --- fail_impl_function_poisoned.carbon

library "[[@TEST_NAME]]";

interface I {
  fn A(x: Self);
  fn B();
}

class B {
  impl as I {
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+11]]:13: error: accessing member from impl before it has a defined value [ImplAccessMemberBeforeSet]
    // CHECK:STDERR:     fn A(x: B);
    // CHECK:STDERR:             ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+7]]:13: error: cannot implicitly convert non-type value of type `<type of B in B as I>` to `type` [ConversionFailureNonTypeToFacet]
    // CHECK:STDERR:     fn A(x: B);
    // CHECK:STDERR:             ^
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+4]]:13: note: type `<type of B in B as I>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn A(x: B);
    // CHECK:STDERR:             ^
    // CHECK:STDERR:
    fn A(x: B);
    fn B();
  }
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon

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
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+3]]:13: error: name `B` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     fn A(x: B);
    // CHECK:STDERR:             ^
    fn A(x: B);
    // TODO: Avoid ImplMissingFunction for functions that were declared after they were poisoned.
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+11]]:8: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     fn B();
    // CHECK:STDERR:        ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE-10]]:3: error: missing implementation of B in impl of interface I [ImplMissingFunction]
    // CHECK:STDERR:   impl as I {
    // CHECK:STDERR:   ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE-17]]:3: note: associated function B declared here [AssociatedFunctionHere]
    // CHECK:STDERR:   fn B();
    // CHECK:STDERR:   ^~~~~~~
    // CHECK:STDERR:
    fn B();
  }
}

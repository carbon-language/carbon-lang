// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/access.carbon

// --- shared.carbon
library "[[@TEST_NAME]]";

interface I { fn Get(self: Self) -> i32; }
interface J { fn Get(self: Self) -> i32; }
interface K(T: type) { fn Get(t: T) -> i32; }

// --- impl_in_class_valid_access.carbon
library "[[@TEST_NAME]]";
import library "shared";

class A {
    private var n: i32;
    impl A as I {
        fn Get(self: Self) -> i32 {
            return self.n;
        }
    }

    impl as K(A) {
        fn Get(t: A) -> i32 {
            return t.n;
        }
    }
}

// --- fail_impl_in_other_class1.carbon
library "[[@TEST_NAME]]";
import library "shared";

class B {
    private var m: i32;
}

class A {
    private var n: i32;

    impl B as I {
        fn Get(self: B) -> i32 {
            // CHECK:STDERR: fail_impl_in_other_class1.carbon:[[@LINE+3]]:20-25: error: cannot access private member `m` of type `B` [ClassInvalidMemberAccess]
            // CHECK:STDERR: fail_impl_in_other_class1.carbon:[[@LINE-9]]:17-22: note: declared here [ClassMemberDeclaration]
            // CHECK:STDERR:
            return self.m;
        }
    }
}

// --- fail_impl_in_other_class2.carbon
library "[[@TEST_NAME]]";
import library "shared";

class B {
    private var m: i32;
}

class A {
    private var n: i32;

    impl as K(B) {
        fn Get(t: B) -> i32 {
            // CHECK:STDERR: fail_impl_in_other_class2.carbon:[[@LINE+3]]:20-22: error: cannot access private member `m` of type `B` [ClassInvalidMemberAccess]
            // CHECK:STDERR: fail_impl_in_other_class2.carbon:[[@LINE-9]]:17-22: note: declared here [ClassMemberDeclaration]
            // CHECK:STDERR:
            return t.m;
        }
    }
}

// --- fail_impl_outside_class1.carbon
library "[[@TEST_NAME]]";
import library "shared";

class A {
    private var n: i32;
}

impl A as J {
    fn Get(self: Self) -> i32 {
        // CHECK:STDERR: fail_impl_outside_class1.carbon:[[@LINE+3]]:16-21: error: cannot access private member `n` of type `A` [ClassInvalidMemberAccess]
        // CHECK:STDERR: fail_impl_outside_class1.carbon:[[@LINE-6]]:17-22: note: declared here [ClassMemberDeclaration]
        // CHECK:STDERR:
        return self.n;
    }
}

// --- fail_impl_outside_class2.carbon
library "[[@TEST_NAME]]";
import library "shared";

class B {
    private var m: i32;
}

impl B as J {
    fn Get(self: B) -> i32 {
        // CHECK:STDERR: fail_impl_outside_class2.carbon:[[@LINE+3]]:16-21: error: cannot access private member `m` of type `B` [ClassInvalidMemberAccess]
        // CHECK:STDERR: fail_impl_outside_class2.carbon:[[@LINE-6]]:17-22: note: declared here [ClassMemberDeclaration]
        // CHECK:STDERR:
        return self.m;
    }
}

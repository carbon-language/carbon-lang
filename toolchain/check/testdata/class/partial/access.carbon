// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/partial/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/partial/access.carbon

// --- common.carbon
package Common;

base class A {
    var n: i32;
    virtual fn F(self);
}

base class B {
    extend base: A;
}

// --- partial_member_access.carbon
library "[[@TEST_NAME]]";
import Common;

fn F(ref pb: partial Common.B) {
    // CHECK:STDERR: partial_member_access.carbon:[[@LINE+7]]:22: error: cannot implicitly convert expression of type `partial Common.B` to `Common.A` [ConversionFailure]
    // CHECK:STDERR:     let ref _: i32 = pb.n;
    // CHECK:STDERR:                      ^~~~
    // CHECK:STDERR: partial_member_access.carbon:[[@LINE+4]]:22: note: type `partial Common.B` does not implement interface `Core.ImplicitAs(Common.A)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     let ref _: i32 = pb.n;
    // CHECK:STDERR:                      ^~~~
    // CHECK:STDERR:
    let ref _: i32 = pb.n;
    // CHECK:STDERR: partial_member_access.carbon:[[@LINE+7]]:35: error: cannot implicitly convert expression of type `partial Common.B` to `Common.B` [ConversionFailure]
    // CHECK:STDERR:     let ref _: partial Common.A = pb.base;
    // CHECK:STDERR:                                   ^~~~~~~
    // CHECK:STDERR: partial_member_access.carbon:[[@LINE+4]]:35: note: type `partial Common.B` does not implement interface `Core.ImplicitAs(Common.B)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     let ref _: partial Common.A = pb.base;
    // CHECK:STDERR:                                   ^~~~~~~
    // CHECK:STDERR:
    let ref _: partial Common.A = pb.base;
}

// --- fail_invalid_partial_member_access.carbon
library "[[@TEST_NAME]]";
import Common;

fn F(ref pb: partial Common.B) {
    // CHECK:STDERR: fail_invalid_partial_member_access.carbon:[[@LINE+7]]:27: error: cannot implicitly convert expression of type `partial Common.B` to `Common.B` [ConversionFailure]
    // CHECK:STDERR:     let ref _: Common.A = pb.base;
    // CHECK:STDERR:                           ^~~~~~~
    // CHECK:STDERR: fail_invalid_partial_member_access.carbon:[[@LINE+4]]:27: note: type `partial Common.B` does not implement interface `Core.ImplicitAs(Common.B)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     let ref _: Common.A = pb.base;
    // CHECK:STDERR:                           ^~~~~~~
    // CHECK:STDERR:
    let ref _: Common.A = pb.base;
}

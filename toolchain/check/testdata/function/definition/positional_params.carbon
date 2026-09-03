// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/positional_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/positional_params.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F {
    $3;
}
//@dump-sem-ir-end

// --- fail_insufficient_args.carbon
library "[[@TEST_NAME]]";

import library "basic";

fn G() {
    // CHECK:STDERR: fail_insufficient_args.carbon:[[@LINE+8]]:5: error: 3 arguments passed to function expecting 4 arguments [CallArgCountMismatch]
    // CHECK:STDERR:     F(0, 1, 2);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_insufficient_args.carbon:[[@LINE-6]]:1: in import [InImport]
    // CHECK:STDERR: basic.carbon:4:1: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR: fn F {
    // CHECK:STDERR: ^~~~~~
    // CHECK:STDERR:
    F(0, 1, 2);
}

// --- nested_functions_inner.carbon
library "[[@TEST_NAME]]";

fn A() {
    fn B() {
        fn C {
            $0;
        }
    }
}

// --- nested_functions_middle.carbon
library "[[@TEST_NAME]]";

fn A() {
    fn B {
        fn C() {
            $0;
        }
    }
}

// --- nested_functions_outer.carbon
library "[[@TEST_NAME]]";

fn A {
    fn B() {
        fn C() {
            $0;
        }
    }
}

// --- todo_fail_nested_functions_middle_and_inner.carbon
library "[[@TEST_NAME]]";

fn A() {
    fn B {
        fn C {
            $0;
        }
    }
}

// --- todo_fail_nested_functions_outer_and_inner.carbon
library "[[@TEST_NAME]]";

fn A {
    fn B() {
        fn C {
            $0;
        }
    }
}

// --- todo_fail_nested_functions_all.carbon
library "[[@TEST_NAME]]";

fn A {
    fn B {
        fn C {
            $0;
        }
    }
}

// --- fail_wrong_scope.carbon
library "[[@TEST_NAME]]";

class A {
    // CHECK:STDERR: fail_wrong_scope.carbon:[[@LINE+11]]:19: error: positional parameters can only be used in a function [PositionalParamInWrongScope]
    // CHECK:STDERR:     var a: type = $0;
    // CHECK:STDERR:                   ^~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_wrong_scope.carbon:[[@LINE+7]]:19: error: cannot implicitly convert non-type value of type `auto` to `type` [ConversionFailureNonTypeToFacet]
    // CHECK:STDERR:     var a: type = $0;
    // CHECK:STDERR:                   ^~
    // CHECK:STDERR: fail_wrong_scope.carbon:[[@LINE+4]]:19: note: type `auto` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     var a: type = $0;
    // CHECK:STDERR:                   ^~
    // CHECK:STDERR:
    var a: type = $0;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %positional_param_3: ref auto = positional_param 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %positional_param_3: ref auto = positional_param 3 [concrete = constants.%positional_param_3]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

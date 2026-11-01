// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/element_mismatches.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/element_mismatches.carbon

// --- fail_arg_wrong_type.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR:        ╭─┤ fail_arg_wrong_type.carbon:[[@LINE+11]]:22
// CHECK:STDERR: ->     │ var a: array(C, 3) = ({}, (), true);
// CHECK:STDERR:        ·                      ┳━━━━━━━━━━━━━
// CHECK:STDERR:        ·                      ┗━━┫ error: cannot implicitly convert expression of type
// CHECK:STDERR:        ·                      │  ┃        `()` to `C`
// CHECK:STDERR:        ·                      │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                      ╰────┤ type `()` does not implement interface
// CHECK:STDERR:        ·                           │ `Core.ImplicitAs(C)`
// CHECK:STDERR:        ·                           │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var a: array(C, 3) = ({}, (), true);

// --- fail_var_wrong_type.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

var a: (D, C, D);
// CHECK:STDERR:        ╭─┤ fail_var_wrong_type.carbon:[[@LINE+11]]:22
// CHECK:STDERR: ->     │ var b: array(C, 3) = a;
// CHECK:STDERR:        ·                      ┳
// CHECK:STDERR:        ·                      ┗━━┫ error: cannot implicitly convert expression of type
// CHECK:STDERR:        ·                      │  ┃        `D` to `C`
// CHECK:STDERR:        ·                      │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                      ╰────┤ type `D` does not implement interface
// CHECK:STDERR:        ·                           │ `Core.ImplicitAs(C)`
// CHECK:STDERR:        ·                           │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var b: array(C, 3) = a;

var c: (C, D, D);
// CHECK:STDERR:        ╭─┤ fail_var_wrong_type.carbon:[[@LINE+9]]:22
// CHECK:STDERR: ->     │ var d: array(C, 3) = c;
// CHECK:STDERR:        ·                      ┳
// CHECK:STDERR:        ·                      ┗━━┫ error: cannot copy value of type `C`
// CHECK:STDERR:        ·                      │  ┃        [CopyOfUncopyableType]
// CHECK:STDERR:        ·                      ╰────┤ type `C` does not implement interface `Core.Copy`
// CHECK:STDERR:        ·                           │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var d: array(C, 3) = c;

// --- fail_arg_too_short.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR:        ╭─┤ fail_arg_too_short.carbon:[[@LINE+8]]:22
// CHECK:STDERR: ->     │ var a: array(C, 3) = ({}, {});
// CHECK:STDERR:        ·                      ┳━━━━━━━
// CHECK:STDERR:        ·                      ┗━━┫ error: cannot initialize array of 3 elements from 2
// CHECK:STDERR:        ·                         ┃        initializers
// CHECK:STDERR:        ·                         ┃        [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var a: array(C, 3) = ({}, {});

// --- fail_var_too_short.carbon

library "[[@TEST_NAME]]";

class C {}

var a: (C, C);
// CHECK:STDERR:        ╭─┤ fail_var_too_short.carbon:[[@LINE+8]]:22
// CHECK:STDERR: ->     │ var b: array(C, 3) = a;
// CHECK:STDERR:        ·                      ┳
// CHECK:STDERR:        ·                      ┗━━┫ error: cannot initialize array of 3 elements from
// CHECK:STDERR:        ·                         ┃        tuple with 2 elements
// CHECK:STDERR:        ·                         ┃        [ArrayInitFromExprArgCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var b: array(C, 3) = a;

// --- fail_arg_too_long.carbon

// CHECK:STDERR:        ╭─┤ fail_arg_too_long.carbon:[[@LINE+8]]:23
// CHECK:STDERR: ->     │ var a: array((), 1) = ((), (), ());
// CHECK:STDERR:        ·                       ┳━━━━━━━━━━━
// CHECK:STDERR:        ·                       ┗━━┫ error: cannot initialize array of 1 element from 3
// CHECK:STDERR:        ·                          ┃        initializers
// CHECK:STDERR:        ·                          ┃        [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var a: array((), 1) = ((), (), ());

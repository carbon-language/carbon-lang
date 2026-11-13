// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/class_tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/class_tuples.carbon

// Versus the no_prelude basics.carbon, this tests some interactions with
// `class`.

// --- fail_element_type_mismatch.carbon
library "[[@TEST_NAME]]";

class C {}
class D {}

var c: C;
var d: D;
// CHECK:STDERR:        ╭─┤ fail_element_type_mismatch.carbon:[[@LINE+11]]:19
// CHECK:STDERR: ->     │ var x: (C*, C*) = (c, d);
// CHECK:STDERR:        ·                   ┳━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: cannot implicitly convert expression of type `C`
// CHECK:STDERR:        ·                   │  ┃        to `C*`
// CHECK:STDERR:        ·                   │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                   ╰────┤ type `C` does not implement interface
// CHECK:STDERR:        ·                        │ `Core.ImplicitAs(C*)`
// CHECK:STDERR:        ·                        │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: (C*, C*) = (c, d);

// --- fail_nested_incomplete.carbon
library "[[@TEST_NAME]]";

class Complete {}
class Incomplete;

// CHECK:STDERR:        ╭─┤ fail_nested_incomplete.carbon:[[@LINE+12]]:8
// CHECK:STDERR:        │ class Incomplete;
// CHECK:STDERR:        · ────────┬────────
// CHECK:STDERR:        ·         ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ var t: (Complete, Incomplete);
// CHECK:STDERR:        ·        ┳━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: binding pattern has incomplete type `(Complete, Incomplete)`
// CHECK:STDERR:        ·           ┃        in name binding declaration
// CHECK:STDERR:        ·           ┃        [IncompleteTypeInBindingDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var t: (Complete, Incomplete);

var p: Incomplete* = &t[1];

// --- fail_type_assign.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR:        ╭─┤ fail_type_assign.carbon:[[@LINE+11]]:16
// CHECK:STDERR: ->     │ var x: (C, ) = (C, );
// CHECK:STDERR:        ·                ┳━━━━
// CHECK:STDERR:        ·                ┗━━┫ error: cannot implicitly convert expression of type `type`
// CHECK:STDERR:        ·                │  ┃        to `C`
// CHECK:STDERR:        ·                │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                ╰────┤ type `type` does not implement interface
// CHECK:STDERR:        ·                     │ `Core.ImplicitAs(C)`
// CHECK:STDERR:        ·                     │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: (C, ) = (C, );

// --- fail_value_as_type.carbon
library "[[@TEST_NAME]]";

class C {}

var c: C;
// CHECK:STDERR:        ╭─┤ fail_value_as_type.carbon:[[@LINE+10]]:8
// CHECK:STDERR: ->     │ var x: c;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: cannot implicitly convert non-type value of type `C` to
// CHECK:STDERR:        ·        │  ┃        `type`
// CHECK:STDERR:        ·        │  ┃        [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:        ·        ╰────┤ type `C` does not implement interface `Core.ImplicitAs(type)`
// CHECK:STDERR:        ·             │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: c;

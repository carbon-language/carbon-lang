// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_bad_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_bad_decl.carbon

// --- fail_not_type.carbon

library "[[@TEST_NAME]]";

class Bad {
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:3: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt 100;
}

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:18: error: member name `F` not found in `Bad` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn Use(b: Bad) { b.F(); }
// CHECK:STDERR:                  ^~~
// CHECK:STDERR:
fn Use(b: Bad) { b.F(); }

// --- fail_extend_not_type.carbon

library "[[@TEST_NAME]]";

class Bad {
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   extend adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+4]]:3: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   extend adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend adapt 100;
}

// No diagnostic here, we don't know what names Bad has.
fn Use(b: Bad) { b.F(); }

// --- fail_repeated.carbon

library "[[@TEST_NAME]]";

class MultipleAdapts {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+7]]:3: error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-4]]:3: note: previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

class MultipleAdaptsSameType {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+7]]:3: error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-4]]:3: note: previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt ();
}

// --- fail_bad_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:1: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
// CHECK:STDERR: adapt {};
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
adapt {};

interface I {
  // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

class C {
  interface I {
    // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:5: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
    // CHECK:STDERR:     adapt {};
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    adapt {};
  }
}

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
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+3]]:9-11: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+2]]:3-12: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplForType]
  // CHECK:STDERR:
  adapt 100;
}

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+3]]:18-20: error: member name `F` not found in `Bad` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fail_not_type.carbon:[[@LINE-8]]:1-11: note: `Bad` declared here [MemberNameNotFoundScope]
// CHECK:STDERR:
fn Use(b: Bad) { b.F(); }

// --- fail_extend_not_type.carbon

library "[[@TEST_NAME]]";

class Bad {
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+3]]:16-18: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+2]]:3-19: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplForType]
  // CHECK:STDERR:
  extend adapt 100;
}

// No diagnostic here, we don't know what names Bad has.
fn Use(b: Bad) { b.F(); }

// --- fail_repeated.carbon

library "[[@TEST_NAME]]";

class MultipleAdapts {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+3]]:3-11: error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-2]]:3-11: note: previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:
  adapt {};
}

class MultipleAdaptsSameType {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+3]]:3-11: error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-2]]:3-11: note: previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:
  adapt ();
}

// --- fail_bad_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+2]]:1-9: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
// CHECK:STDERR:
adapt {};

interface I {
  // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+3]]:3-11: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE-2]]:1-13: note: appears in this declaration [ClassSpecificDeclEnclosingScope]
  // CHECK:STDERR:
  adapt {};
}

class C {
  interface I {
    // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+3]]:5-13: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
    // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE-2]]:3-15: note: appears in this declaration [ClassSpecificDeclEnclosingScope]
    // CHECK:STDERR:
    adapt {};
  }
}

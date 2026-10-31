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
  // CHECK:STDERR:        ╭─┤ fail_not_type.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   adapt 100;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `Core.IntLiteral`
  // CHECK:STDERR:        ·   │  ┃        to `type`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `Core.IntLiteral` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(type)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  adapt 100;
}

// CHECK:STDERR:        ╭─┤ fail_not_type.carbon:[[@LINE+7]]:18
// CHECK:STDERR: ->     │ fn Use(b: Bad) { b.F(); }
// CHECK:STDERR:        ·                  ━┳━
// CHECK:STDERR:        ·                   ┗━━┫ error: member name `F` not found in `Bad`
// CHECK:STDERR:        ·                      ┃        [MemberNameNotFoundInInstScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Use(b: Bad) { b.F(); }

// --- fail_extend_not_type.carbon

library "[[@TEST_NAME]]";

class Bad {
  // CHECK:STDERR:        ╭─┤ fail_extend_not_type.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   extend adapt 100;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `Core.IntLiteral`
  // CHECK:STDERR:        ·   │  ┃        to `type`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `Core.IntLiteral` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(type)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend adapt 100;
}

// No diagnostic here, we don't know what names Bad has.
fn Use(b: Bad) { b.F(); }

// --- fail_repeated.carbon

library "[[@TEST_NAME]]";

class MultipleAdapts {
  adapt ();
  // CHECK:STDERR:        ╭─┤ fail_repeated.carbon:[[@LINE+10]]:3
  // CHECK:STDERR:        │   adapt ();
  // CHECK:STDERR:        ·   ────┬────
  // CHECK:STDERR:        ·       ╰──┤ previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   adapt {};
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  adapt {};
}

class MultipleAdaptsSameType {
  adapt ();
  // CHECK:STDERR:        ╭─┤ fail_repeated.carbon:[[@LINE+10]]:3
  // CHECK:STDERR:        │   adapt ();
  // CHECK:STDERR:        ·   ────┬────
  // CHECK:STDERR:        ·       ╰──┤ previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   adapt ();
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  adapt ();
}

// --- fail_bad_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_bad_scope.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ adapt {};
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `adapt` declaration outside class
// CHECK:STDERR:        ·        ┃        [ClassSpecificDeclOutsideClass]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
adapt {};

interface I {
  // CHECK:STDERR:        ╭─┤ fail_bad_scope.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   adapt {};
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: `adapt` declaration outside class
  // CHECK:STDERR:        ·          ┃        [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  adapt {};
}

class C {
  interface I {
    // CHECK:STDERR:        ╭─┤ fail_bad_scope.carbon:[[@LINE+7]]:5
    // CHECK:STDERR: ->     │     adapt {};
    // CHECK:STDERR:        ·     ━━━━┳━━━━
    // CHECK:STDERR:        ·         ┗━━┫ error: `adapt` declaration outside class
    // CHECK:STDERR:        ·            ┃        [ClassSpecificDeclOutsideClass]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    adapt {};
  }
}

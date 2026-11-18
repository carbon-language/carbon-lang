// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_invalid.carbon

// --- fail_require_outside_scope.carbon
library "[[@TEST_NAME]]";

interface Y {}

// CHECK:STDERR:        ╭─┤ fail_require_outside_scope.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ require impls Y;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `require` can only be used in an `interface` or `constraint`
// CHECK:STDERR:        ·       ┃        [RequireInWrongScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
require impls Y;

// --- fail_require_in_class.carbon
library "[[@TEST_NAME]]";

interface Y {}

class C {
  // CHECK:STDERR:        ╭─┤ fail_require_in_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   require impls Y;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: `require` can only be used in an `interface` or `constraint`
  // CHECK:STDERR:        ·         ┃        [RequireInWrongScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_require_in_fn.carbon
library "[[@TEST_NAME]]";

interface Y {}

fn F() {
  // require is not allowed outside of `interface` or `constraint`.
  //
  // CHECK:STDERR:        ╭─┤ fail_require_in_fn.carbon:[[@LINE+13]]:3
  // CHECK:STDERR: ->     │   require impls Y;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: expected expression [ExpectedExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_require_in_fn.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   require impls Y;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
  // CHECK:STDERR:        ·         ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_require_in_nested_class.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // TODO: Add `default` modifier.
  fn F() {
    class C {
      // CHECK:STDERR:        ╭─┤ fail_require_in_nested_class.carbon:[[@LINE+7]]:7
      // CHECK:STDERR: ->     │       require impls Y;
      // CHECK:STDERR:        ·       ━━┳━━━━
      // CHECK:STDERR:        ·         ┗━━┫ error: `require` can only be used in an `interface` or `constraint`
      // CHECK:STDERR:        ·            ┃        [RequireInWrongScope]
      // CHECK:STDERR:        ╰────
      // CHECK:STDERR:
      require impls Y;
    }
  }
}

// --- fail_require_in_nested_fn.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // TODO: Add `default` modifier.
  fn F() {
    // require is not allowed outside of `interface` or `constraint`.
    //
    // CHECK:STDERR:        ╭─┤ fail_require_in_nested_fn.carbon:[[@LINE+13]]:5
    // CHECK:STDERR: ->     │     require impls Y;
    // CHECK:STDERR:        ·     ━━━┳━━━
    // CHECK:STDERR:        ·        ┗━━┫ error: expected expression [ExpectedExpr]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    // CHECK:STDERR:        ╭─┤ fail_require_in_nested_fn.carbon:[[@LINE+7]]:5
    // CHECK:STDERR: ->     │     require impls Y;
    // CHECK:STDERR:        ·     ━━━┳━━━
    // CHECK:STDERR:        ·        ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
    // CHECK:STDERR:        ·           ┃        [SemanticsTodo]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    require impls Y;
  }
}

// --- fail_errors_in_require_still_found.carbon
library "[[@TEST_NAME]]";

class C {
  // The `require` is diagnosed as being in the wrong place. But we can still
  // diagnose issues inside the decl too.
  //
  // CHECK:STDERR:        ╭─┤ fail_errors_in_require_still_found.carbon:[[@LINE+13]]:3
  // CHECK:STDERR: ->     │   require impls Y;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: `require` can only be used in an `interface` or `constraint`
  // CHECK:STDERR:        ·         ┃        [RequireInWrongScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_errors_in_require_still_found.carbon:[[@LINE+6]]:17
  // CHECK:STDERR: ->     │   require impls Y;
  // CHECK:STDERR:        ·                 ┳
  // CHECK:STDERR:        ·                 ┗━━┫ error: name `Y` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_extend_require_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR:        ╭─┤ fail_extend_require_type.carbon:[[@LINE+7]]:18
  // CHECK:STDERR: ->     │   extend require () impls Y;
  // CHECK:STDERR:        ·                  ━┳
  // CHECK:STDERR:        ·                   ┗━━┫ error: `extend require impls` with explicit type
  // CHECK:STDERR:        ·                      ┃        [RequireImplsExtendWithExplicitSelf]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend require () impls Y;

  fn F() {
    // The erroneous self-type for an `extend` causes the error to be propagated
    // into the interface scope, which prevents errors if we fail to find a name
    // in that scope.
    Self.A;
  }
}

// --- fail_require_with_nonexistent_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR:        ╭─┤ fail_require_with_nonexistent_type.carbon:[[@LINE+6]]:11
  // CHECK:STDERR: ->     │   require nonexistent impls Y;
  // CHECK:STDERR:        ·           ━━━━━┳━━━━━
  // CHECK:STDERR:        ·                ┗━━┫ error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  require nonexistent impls Y;

  fn F() {
    // The name lookup error still happens, since the `require` is not `extend`.
    // CHECK:STDERR:        ╭─┤ fail_require_with_nonexistent_type.carbon:[[@LINE+7]]:5
    // CHECK:STDERR: ->     │     Self.A;
    // CHECK:STDERR:        ·     ━━━┳━━
    // CHECK:STDERR:        ·        ┗━━┫ error: member name `A` not found in `Z`
    // CHECK:STDERR:        ·           ┃        [MemberNameNotFoundInSpecificScope]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    Self.A;
  }
}

// --- fail_extend_require_with_nonexistent_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR:        ╭─┤ fail_extend_require_with_nonexistent_type.carbon:[[@LINE+6]]:18
  // CHECK:STDERR: ->     │   extend require nonexistent impls Y;
  // CHECK:STDERR:        ·                  ━━━━━┳━━━━━
  // CHECK:STDERR:        ·                       ┗━━┫ error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend require nonexistent impls Y;

  fn F() {
    // The erroneous self-type for an `extend` causes the error to be propagated
    // into the interface scope, which prevents errors if we fail to find a name
    // in that scope.
    Self.A;
  }
}

// --- fail_extend_require_with_nonexistent_type_pointer.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR:        ╭─┤ fail_extend_require_with_nonexistent_type_pointer.carbon:[[@LINE+6]]:18
  // CHECK:STDERR: ->     │   extend require nonexistent* impls Y;
  // CHECK:STDERR:        ·                  ━━━━━┳━━━━━
  // CHECK:STDERR:        ·                       ┗━━┫ error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend require nonexistent* impls Y;

  fn F() {
    // The erroneous self-type for an `extend` causes the error to be propagated
    // into the interface scope, which prevents errors if we fail to find a name
    // in that scope.
    Self.A;
  }
}

// --- fail_impl_mismatch_named_constraint_rewrites.carbon
library "[[@TEST_NAME]]";

interface Z { let Z0: type; }

constraint M {
  require impls Z where .Z0 = {};
}

fn G() {
  class C {}
  // .Z0 has the wrong type for M.
  impl C as Z where .Z0 = () {}
  // CHECK:STDERR:        ╭─┤ fail_impl_mismatch_named_constraint_rewrites.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   C as M;
  // CHECK:STDERR:        ·   ━━━┳━━
  // CHECK:STDERR:        ·      ┗━━┫ error: cannot convert type `C` into type implementing `M`
  // CHECK:STDERR:        ·         ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  C as M;
}

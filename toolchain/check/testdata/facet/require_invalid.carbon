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

// CHECK:STDERR: fail_require_outside_scope.carbon:[[@LINE+4]]:1: error: `require` can only be used in an `interface` or `constraint` [RequireInWrongScope]
// CHECK:STDERR: require impls Y;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
require impls Y;

// --- fail_require_in_class.carbon
library "[[@TEST_NAME]]";

interface Y {}

class C {
  // CHECK:STDERR: fail_require_in_class.carbon:[[@LINE+4]]:3: error: `require` can only be used in an `interface` or `constraint` [RequireInWrongScope]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_require_in_fn.carbon
library "[[@TEST_NAME]]";

interface Y {}

fn F() {
  // require is not allowed outside of `interface` or `constraint`.
  //
  // CHECK:STDERR: fail_require_in_fn.carbon:[[@LINE+8]]:3: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_require_in_fn.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
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
      // CHECK:STDERR: fail_require_in_nested_class.carbon:[[@LINE+4]]:7: error: `require` can only be used in an `interface` or `constraint` [RequireInWrongScope]
      // CHECK:STDERR:       require impls Y;
      // CHECK:STDERR:       ^~~~~~~
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
    // CHECK:STDERR: fail_require_in_nested_fn.carbon:[[@LINE+8]]:5: error: expected expression [ExpectedExpr]
    // CHECK:STDERR:     require impls Y;
    // CHECK:STDERR:     ^~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_require_in_nested_fn.carbon:[[@LINE+4]]:5: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
    // CHECK:STDERR:     require impls Y;
    // CHECK:STDERR:     ^~~~~~~
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
  // CHECK:STDERR: fail_errors_in_require_still_found.carbon:[[@LINE+8]]:3: error: `require` can only be used in an `interface` or `constraint` [RequireInWrongScope]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_errors_in_require_still_found.carbon:[[@LINE+4]]:17: error: name `Y` not found [NameNotFound]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_extend_require_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_extend_require_type.carbon:[[@LINE+4]]:18: error: `extend require impls` with explicit type [RequireImplsExtendWithExplicitSelf]
  // CHECK:STDERR:   extend require () impls Y;
  // CHECK:STDERR:                  ^~
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
  // CHECK:STDERR: fail_require_with_nonexistent_type.carbon:[[@LINE+4]]:11: error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:   require nonexistent impls Y;
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  require nonexistent impls Y;

  fn F() {
    // The name lookup error still happens, since the `require` is not `extend`.
    // CHECK:STDERR: fail_require_with_nonexistent_type.carbon:[[@LINE+4]]:5: error: member name `A` not found in `Z` [MemberNameNotFoundInSpecificScope]
    // CHECK:STDERR:     Self.A;
    // CHECK:STDERR:     ^~~~~~
    // CHECK:STDERR:
    Self.A;
  }
}

// --- fail_extend_require_with_nonexistent_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z {
  // CHECK:STDERR: fail_extend_require_with_nonexistent_type.carbon:[[@LINE+4]]:18: error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:   extend require nonexistent impls Y;
  // CHECK:STDERR:                  ^~~~~~~~~~~
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
  // CHECK:STDERR: fail_extend_require_with_nonexistent_type_pointer.carbon:[[@LINE+4]]:18: error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:   extend require nonexistent* impls Y;
  // CHECK:STDERR:                  ^~~~~~~~~~~
  // CHECK:STDERR:
  extend require nonexistent* impls Y;

  fn F() {
    // The erroneous self-type for an `extend` causes the error to be propagated
    // into the interface scope, which prevents errors if we fail to find a name
    // in that scope.
    Self.A;
  }
}

// --- todo_fail_impl_mismatch_named_constraint_rewrites.carbon
library "[[@TEST_NAME]]";

interface Z { let Z0:! type; }

constraint M {
  require impls Z where .Z0 = {};
}

fn G() {
  class C {}
  // .Z0 has the wrong type for M.
  impl C as Z where .Z0 = () {}
  // TODO: Converting C to M should fail since the rewrite constraints are incompatible.
  C as M;
}

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

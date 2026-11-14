// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/where_assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/where_assignment.carbon

// --- fail_todo_basic.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

// CHECK:STDERR:        ╭─┤ fail_todo_basic.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │   let Z1: type;
// CHECK:STDERR:        ·       ────┬───
// CHECK:STDERR:        ·           ╰──┤ associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as Z {
// CHECK:STDERR:        · ━━━━━┳━━━━━━━━
// CHECK:STDERR:        ·      ┗━━┫ error: associated constant Z1 not given a value in impl of interface Z
// CHECK:STDERR:        ·         ┃        [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_todo_basic.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   where Z1 = type;
  // CHECK:STDERR:        ·   ━━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: semantics TODO: `where assignment decl` [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = type;
}

// --- fail_todo_basic_in_class.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

class C {
  // CHECK:STDERR:        ╭─┤ fail_todo_basic_in_class.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   let Z1: type;
  // CHECK:STDERR:        ·       ────┬───
  // CHECK:STDERR:        ·           ╰──┤ associated constant declared here [AssociatedConstantHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   impl as Z {
  // CHECK:STDERR:        ·   ━━━┳━━━━━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: associated constant Z1 not given a value in impl of interface Z
  // CHECK:STDERR:        ·         ┃        [ImplAssociatedConstantNeedsValue]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl as Z {
    // CHECK:STDERR:        ╭─┤ fail_todo_basic_in_class.carbon:[[@LINE+6]]:5
    // CHECK:STDERR: ->     │     where Z1 = type;
    // CHECK:STDERR:        ·     ━━━━━━━━┳━━━━━━━
    // CHECK:STDERR:        ·             ┗━━┫ error: semantics TODO: `where assignment decl` [SemanticsTodo]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    where Z1 = type;
  }
}

// --- fail_invalid_scope.carbon
library "[[@TEST_NAME]]";

let Z1: type = type;
// CHECK:STDERR:        ╭─┤ fail_invalid_scope.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ where Z1 = type;
// CHECK:STDERR:        · ┳━━━━
// CHECK:STDERR:        · ┗━━┫ error: `where` assignment declaration found in invalid scope; must be in
// CHECK:STDERR:        ·    ┃        `impl` definition
// CHECK:STDERR:        ·    ┃        [WhereAssignmentInWrongScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
where Z1 = type;

// --- fail_class_invalid_scope.carbon
library "[[@TEST_NAME]]";

class C {
  let Z1: type = type;
  // CHECK:STDERR:        ╭─┤ fail_class_invalid_scope.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = type;
  // CHECK:STDERR:        ·   ┳━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: `where` assignment declaration found in invalid scope; must be in
  // CHECK:STDERR:        ·      ┃        `impl` definition
  // CHECK:STDERR:        ·      ┃        [WhereAssignmentInWrongScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = type;
}

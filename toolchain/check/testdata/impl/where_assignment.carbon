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

// --- basic.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z {
  where Z1 = type;
}

// --- basic_in_class.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

class C {
  impl as Z {
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

// --- fail_fn_invalid_scope.carbon
library "[[@TEST_NAME]]";

let Z1: type = type;
fn F() {
  // CHECK:STDERR:        ╭─┤ fail_fn_invalid_scope.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = type;
  // CHECK:STDERR:        ·   ┳━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: `where` assignment declaration found in invalid scope; must be in
  // CHECK:STDERR:        ·      ┃        `impl` definition
  // CHECK:STDERR:        ·      ┃        [WhereAssignmentInWrongScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = type;
}

// --- fail_region_invalid_scope.carbon
library "[[@TEST_NAME]]";

let Z1: type = type;
fn F() {
  // If blocks have a scope that has no declaration scope inst associated with
  // them.
  if (true) {
    // CHECK:STDERR:        ╭─┤ fail_region_invalid_scope.carbon:[[@LINE+8]]:5
    // CHECK:STDERR: ->     │     where Z1 = type;
    // CHECK:STDERR:        ·     ┳━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: `where` assignment declaration found in invalid scope; must be
    // CHECK:STDERR:        ·        ┃        in `impl` definition
    // CHECK:STDERR:        ·        ┃        [WhereAssignmentInWrongScope]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    where Z1 = type;
  }
}

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

// --- fail_name_not_assoc_const.carbon
library "[[@TEST_NAME]]";

let Y: type = type;

interface Z {
  let Z1: type;
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_name_not_assoc_const.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   where Y = type;
  // CHECK:STDERR:        ·   ━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: member name `Y` not found in `Z`
  // CHECK:STDERR:        ·             ┃        [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Y = type;
  where Z1 = type;
}

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

interface Z;

// CHECK:STDERR:        ╭─┤ fail_incomplete.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │ interface Z;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as Z {
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: definition of impl as incomplete facet type `Z`
// CHECK:STDERR:        ·           ┃        [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as Z {
  where Z1 = type;
}

// --- todo_fail_runtime.carbon
library "[[@TEST_NAME]]";

var Runtime: type = type;

interface Z {
  let Z1: type;
}

impl () as Z {
  // TODO: This assigns a constant value of VarStorage which seems weird.
  // https://discord.com/channels/655572317891461132/655578254970716160/1571972219964367039
  where Z1 = Runtime;
}

// --- fail_runtime_fn.carbon
library "[[@TEST_NAME]]";

fn Runtime() -> type;

interface Z {
  let Z1: type;
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_runtime_fn.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   where Z1 = Runtime();
  // CHECK:STDERR:        ·   ━━━━━━━━━━┳━━━━━━━━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: runtime value assigned to an associated constant
  // CHECK:STDERR:        ·                ┃        [WhereAssignmentRuntimeValue]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = Runtime();
}

// --- same_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z {
  where Z1 = type;
  where Z1 = type;
}

// --- same_value_as_decl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z where .Z1 = type {
  where Z1 = type;
}

// --- fail_different_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z {
  where Z1 = type;
  // CHECK:STDERR:        ╭─┤ fail_different_value.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = {};
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: associated constant Z1 given value `{}` that is different from
  // CHECK:STDERR:        ·      ┃        previously given value `type`
  // CHECK:STDERR:        ·      ┃        [AssociatedConstantGivenDifferentValues]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = {};
}

// --- fail_different_value_from_decl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z where .Z1 = type {
  // CHECK:STDERR:        ╭─┤ fail_different_value_from_decl.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = {};
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: associated constant Z1 given value `{}` that is different from
  // CHECK:STDERR:        ·      ┃        previously given value `type`
  // CHECK:STDERR:        ·      ┃        [AssociatedConstantGivenDifferentValues]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = {};
}

// --- fail_assign_fn.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn F();
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_assign_fn.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   where F = fn()=>();
  // CHECK:STDERR:        ·   ━━━━━━━━━┳━━━━━━━━━
  // CHECK:STDERR:        ·            ┗━━┫ error: assigning to associated function F
  // CHECK:STDERR:        ·               ┃        [RewriteForAssociatedFunction]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where F = fn()=>();

  fn F() {}
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_incomplete.carbon

// --- fail_forward_decl.carbon

library "[[@TEST_NAME]]";

class Class;

// CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+11]]:4
// CHECK:STDERR:        │ class Class;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn Class.Function() {}
// CHECK:STDERR:        ·    ━━┳━━
// CHECK:STDERR:        ·      ┗━━┫ error: cannot declare a member of incomplete class `Class`
// CHECK:STDERR:        ·         ┃        [QualifiedDeclInIncompleteClassScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Class.Function() {}

fn CallClassFunction() {
  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Class.Function();
  // CHECK:STDERR:        ·   ━━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: member access into incomplete class `Class`
  // CHECK:STDERR:        ·             ┃        [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Class.Function();
}

// CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+12]]:17
// CHECK:STDERR:        │ class Class;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ var global_var: Class;
// CHECK:STDERR:        ·                 ┳━━━━
// CHECK:STDERR:        ·                 ┗━━┫ error: binding pattern has incomplete type `Class` in name
// CHECK:STDERR:        ·                    ┃        binding declaration
// CHECK:STDERR:        ·                    ┃        [IncompleteTypeInBindingDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var global_var: Class;

// CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+11]]:24
// CHECK:STDERR:        │ class Class;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn ConvertFromStruct() -> Class { return {}; }
// CHECK:STDERR:        ·                        ━━━━┳━━━
// CHECK:STDERR:        ·                            ┗━━┫ error: function returns incomplete type `Class`
// CHECK:STDERR:        ·                               ┃        [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertFromStruct() -> Class { return {}; }

fn G(p: Class*) -> () {
  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+11]]:10
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return p->n;
  // CHECK:STDERR:        ·          ━━┳━
  // CHECK:STDERR:        ·            ┗━━┫ error: member access into object of incomplete type `Class`
  // CHECK:STDERR:        ·               ┃        [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return p->n;
}

fn MemberAccess(p: Class*) -> () {
  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+11]]:11
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return (*p).n;
  // CHECK:STDERR:        ·           ━┳
  // CHECK:STDERR:        ·            ┗━━┫ error: member access into object of incomplete type `Class`
  // CHECK:STDERR:        ·               ┃        [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return (*p).n;
}

// CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+11]]:20
// CHECK:STDERR:        │ class Class;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn Copy(p: Class*) -> Class {
// CHECK:STDERR:        ·                    ━━━━┳━━━
// CHECK:STDERR:        ·                        ┗━━┫ error: function returns incomplete type `Class`
// CHECK:STDERR:        ·                           ┃        [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+12]]:17
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let unused c: Class = *p;
  // CHECK:STDERR:        ·                 ┳━━━━
  // CHECK:STDERR:        ·                 ┗━━┫ error: binding pattern has incomplete type `Class` in name
  // CHECK:STDERR:        ·                    ┃        binding declaration
  // CHECK:STDERR:        ·                    ┃        [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let unused c: Class = *p;
}

fn TakeIncomplete(c: Class);

fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+16]]:18
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn TakeIncomplete(c: Class);
  // CHECK:STDERR:        ·                   ────┬───
  // CHECK:STDERR:        ·                       ╰──┤ initializing function parameter
  // CHECK:STDERR:        ·                          │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   TakeIncomplete(*p);
  // CHECK:STDERR:        ·                  ━┳
  // CHECK:STDERR:        ·                   ┗━━┫ error: forming value of incomplete type `Class`
  // CHECK:STDERR:        ·                      ┃        [IncompleteTypeInValueConversion]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+16]]:18
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn TakeIncomplete(c: Class);
  // CHECK:STDERR:        ·                   ────┬───
  // CHECK:STDERR:        ·                       ╰──┤ initializing function parameter
  // CHECK:STDERR:        ·                          │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   TakeIncomplete({});
  // CHECK:STDERR:        ·                  ━┳
  // CHECK:STDERR:        ·                   ┗━━┫ error: forming value of incomplete type `Class`
  // CHECK:STDERR:        ·                      ┃        [IncompleteTypeInValueConversion]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  // CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+16]]:3
  // CHECK:STDERR:        │ class Class;
  // CHECK:STDERR:        · ──────┬─────
  // CHECK:STDERR:        ·       ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn ReturnIncomplete() -> Class;
  // CHECK:STDERR:        ·                       ────┬───
  // CHECK:STDERR:        ·                           ╰──┤ return type declared here
  // CHECK:STDERR:        ·                              │ [IncompleteReturnTypeHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   ReturnIncomplete();
  // CHECK:STDERR:        ·   ━━━━━━━━━┳━━━━━━━━
  // CHECK:STDERR:        ·            ┗━━┫ error: function returns incomplete type `Class`
  // CHECK:STDERR:        ·               ┃        [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  ReturnIncomplete();
}

// --- fail_in_definition.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR:        ╭─┤ fail_in_definition.carbon:[[@LINE+13]]:10
  // CHECK:STDERR:        │ class C {
  // CHECK:STDERR:        · ────┬────
  // CHECK:STDERR:        ·     ╰──┤ class is incomplete within its definition
  // CHECK:STDERR:        ·        │ [ClassIncompleteWithinDefinition]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var c: C;
  // CHECK:STDERR:        ·          ┳
  // CHECK:STDERR:        ·          ┗━━┫ error: binding pattern has incomplete type `C` in name binding
  // CHECK:STDERR:        ·             ┃        declaration
  // CHECK:STDERR:        ·             ┃        [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var c: C;
}

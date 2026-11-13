// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_modifiers.carbon

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:9
// CHECK:STDERR: ->     │ private private class DuplicatePrivate;
// CHECK:STDERR:        · ───┬─── ━━━┳━━━
// CHECK:STDERR:        ·    │       ┗━━┫ error: `private` repeated on declaration [ModifierRepeated]
// CHECK:STDERR:        ·    ╰────────────┤ `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
private private class DuplicatePrivate;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ abstract class AbstractDecl;
// CHECK:STDERR:        · ┳━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: `abstract` not allowed on `class` forward declaration, only
// CHECK:STDERR:        ·    ┃        definition
// CHECK:STDERR:        ·    ┃        [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
abstract class AbstractDecl;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+8]]:9
// CHECK:STDERR: ->     │ private protected class TwoAccess;
// CHECK:STDERR:        · ───┬─── ━━━━┳━━━━
// CHECK:STDERR:        ·    │        ┗━━┫ error: `protected` not allowed on declaration with `private`
// CHECK:STDERR:        ·    │           ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·    ╰─────────────┤ `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
private protected class TwoAccess;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ base class BaseDecl;
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `base` not allowed on `class` forward declaration, only definition
// CHECK:STDERR:        ·      ┃        [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
base class BaseDecl;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:10
// CHECK:STDERR: ->     │ abstract abstract class TwoAbstract { }
// CHECK:STDERR:        · ────┬─── ━━━━┳━━━
// CHECK:STDERR:        ·     │        ┗━━┫ error: `abstract` repeated on declaration [ModifierRepeated]
// CHECK:STDERR:        ·     ╰─────────────┤ `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
abstract abstract class TwoAbstract { }

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+22]]:19
// CHECK:STDERR: ->     │ protected virtual base class Virtual {}
// CHECK:STDERR:        ·           ───┬─── ━━┳━
// CHECK:STDERR:        ·              │      ┗━━┫ error: `base` not allowed on declaration with `virtual`
// CHECK:STDERR:        ·              │         ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·              ╰───────────┤ `virtual` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ protected virtual base class Virtual {}
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `protected` not allowed; requires class scope
// CHECK:STDERR:        ·        ┃        [ModifierProtectedNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:11
// CHECK:STDERR: ->     │ protected virtual base class Virtual {}
// CHECK:STDERR:        ·           ━━━┳━━━
// CHECK:STDERR:        ·              ┗━━┫ error: `virtual` not allowed on `class` declaration
// CHECK:STDERR:        ·                 ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
protected virtual base class Virtual {}

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+8]]:10
// CHECK:STDERR: ->     │ abstract protected class WrongOrder { }
// CHECK:STDERR:        · ────┬─── ━━━━┳━━━━
// CHECK:STDERR:        ·     │        ┗━━┫ error: `protected` must appear before `abstract`
// CHECK:STDERR:        ·     │           ┃        [ModifierMustAppearBefore]
// CHECK:STDERR:        ·     ╰─────────────┤ `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
abstract protected class WrongOrder { }

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+8]]:10
// CHECK:STDERR: ->     │ abstract base class AbstractAndBase {}
// CHECK:STDERR:        · ────┬─── ━━┳━
// CHECK:STDERR:        ·     │      ┗━━┫ error: `base` not allowed on declaration with `abstract`
// CHECK:STDERR:        ·     │         ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·     ╰───────────┤ `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
abstract base class AbstractAndBase {}

abstract class AbstractWithDefinition {
  // CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:30
  // CHECK:STDERR: ->     │   abstract fn F(unused self) {}
  // CHECK:STDERR:        ·                              ┳
  // CHECK:STDERR:        ·                              ┗━━┫ error: definition of `abstract` function
  // CHECK:STDERR:        ·                                 ┃        [DefinedAbstractFunction]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  abstract fn F(unused self) {}
  abstract fn G(self);
}
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+8]]:42
// CHECK:STDERR: ->     │ fn AbstractWithDefinition.G(unused self) {
// CHECK:STDERR:        ·                                          ┳
// CHECK:STDERR:        ·                                          ┗━━┫ error: definition of `abstract`
// CHECK:STDERR:        ·                                             ┃        function
// CHECK:STDERR:        ·                                             ┃        [DefinedAbstractFunction]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn AbstractWithDefinition.G(unused self) {
}

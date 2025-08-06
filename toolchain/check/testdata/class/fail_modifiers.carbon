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

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `private` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: private private class DuplicatePrivate;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR: abstract class AbstractDecl;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract class AbstractDecl;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `protected` not allowed on declaration with `private` [ModifierNotAllowedWith]
// CHECK:STDERR: private protected class TwoAccess;
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private protected class TwoAccess;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected class TwoAccess;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `base` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR: base class BaseDecl;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base class BaseDecl;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `abstract` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: abstract abstract class TwoAbstract { }
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract abstract class TwoAbstract { }
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract abstract class TwoAbstract { }

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:19: error: `base` not allowed on declaration with `virtual` [ModifierNotAllowedWith]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:11: note: `virtual` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:11: error: `virtual` not allowed on `class` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `protected` must appear before `abstract` [ModifierMustAppearBefore]
// CHECK:STDERR: abstract protected class WrongOrder { }
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract protected class WrongOrder { }
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract protected class WrongOrder { }

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base class AbstractAndBase {}

abstract class AbstractWithDefinition {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:31: error: definition of `abstract` function [DefinedAbstractFunction]
  // CHECK:STDERR:   abstract fn F[self: Self]() {}
  // CHECK:STDERR:                               ^
  // CHECK:STDERR:
  abstract fn F[self: Self]() {}
  abstract fn G[self: Self]();
}
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:43: error: definition of `abstract` function [DefinedAbstractFunction]
// CHECK:STDERR: fn AbstractWithDefinition.G[self: Self]() {
// CHECK:STDERR:                                           ^
// CHECK:STDERR:
fn AbstractWithDefinition.G[self: Self]() {
}

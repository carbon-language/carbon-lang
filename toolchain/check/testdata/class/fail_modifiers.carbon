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

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:9-15: error: `private` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-7: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
private private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-8: error: `abstract` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR:
abstract class AbstractDecl;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:9-17: error: `protected` not allowed on declaration with `private` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-7: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
private protected class TwoAccess;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-4: error: `base` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR:
base class BaseDecl;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:10-17: error: `abstract` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-8: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
abstract abstract class TwoAbstract { }

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:19-22: error: `base` not allowed on declaration with `virtual` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11-17: note: `virtual` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1-9: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:11-17: error: `virtual` not allowed on `class` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:10-18: error: `protected` must appear before `abstract` [ModifierMustAppearBefore]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-8: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
abstract protected class WrongOrder { }

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:10-13: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-8: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
abstract base class AbstractAndBase {}

abstract class AbstractWithDefinition {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:30: error: definition of `abstract` function [DefinedAbstractFunction]
  // CHECK:STDERR:
  abstract fn F(unused self) {}
  abstract fn G(self);
}
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:42: error: definition of `abstract` function [DefinedAbstractFunction]
// CHECK:STDERR:
fn AbstractWithDefinition.G(unused self) {
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `private`.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:18: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Previously appeared here.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^
private abstract private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `private`.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:14: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^
private base protected class TwoAccess {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Previously appeared here.
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR: ^
abstract abstract class TwoAbstract;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: Modifier not allowed.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:24: On this declaration.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:                        ^
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: Modifier must appear earlier.
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Before this modifier.
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR: ^
abstract protected class WrongOrder;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR: ^
abstract base class AbstractAndBase {}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc27: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_modifiers.carbon" {
// CHECK:STDOUT:   class_decl @DuplicatePrivate, ()
// CHECK:STDOUT:   %DuplicatePrivate: type = class_type @DuplicatePrivate
// CHECK:STDOUT:   class_decl @TwoAccess, ()
// CHECK:STDOUT:   %TwoAccess: type = class_type @TwoAccess
// CHECK:STDOUT:   class_decl @TwoAbstract, ()
// CHECK:STDOUT:   %TwoAbstract: type = class_type @TwoAbstract
// CHECK:STDOUT:   class_decl @Virtual, ()
// CHECK:STDOUT:   %Virtual: type = class_type @Virtual
// CHECK:STDOUT:   class_decl @WrongOrder, ()
// CHECK:STDOUT:   %WrongOrder: type = class_type @WrongOrder
// CHECK:STDOUT:   class_decl @AbstractAndBase, ()
// CHECK:STDOUT:   %AbstractAndBase: type = class_type @AbstractAndBase
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DuplicatePrivate;
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAccess {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAbstract;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Virtual {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongOrder;
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractAndBase {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }

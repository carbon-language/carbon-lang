// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:18: ERROR: `private` repeated on declaration.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR:                  ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `private` previously appeared here.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
private abstract private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:14: ERROR: `protected` not allowed on declaration with `private`.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `private` previously appeared here.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^~~~~~~
private base protected class TwoAccess {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: `abstract` repeated on declaration.
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR: ^~~~~~~~
abstract abstract class TwoAbstract;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:1: ERROR: `protected` not allowed on `class` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:11: ERROR: `virtual` not allowed on `class` declaration.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:19: ERROR: `base` not allowed on declaration with `virtual`.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:11: `virtual` previously appeared here.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: `protected` must appear before `abstract`.
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR: ^~~~~~~~
abstract protected class WrongOrder;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR: ^~~~~~~~
abstract base class AbstractAndBase {}

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc27: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
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
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %DuplicatePrivate: type = class_type @DuplicatePrivate [template]
// CHECK:STDOUT:   %TwoAccess: type = class_type @TwoAccess [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %TwoAbstract: type = class_type @TwoAbstract [template]
// CHECK:STDOUT:   %Virtual: type = class_type @Virtual [template]
// CHECK:STDOUT:   %WrongOrder: type = class_type @WrongOrder [template]
// CHECK:STDOUT:   %AbstractAndBase: type = class_type @AbstractAndBase [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.DuplicatePrivate = %DuplicatePrivate.decl, .TwoAccess = %TwoAccess.decl, .TwoAbstract = %TwoAbstract.decl, .Virtual = %Virtual.decl, .WrongOrder = %WrongOrder.decl, .AbstractAndBase = %AbstractAndBase.decl} [template]
// CHECK:STDOUT:   %DuplicatePrivate.decl = class_decl @DuplicatePrivate, ()
// CHECK:STDOUT:   %TwoAccess.decl = class_decl @TwoAccess, ()
// CHECK:STDOUT:   %TwoAbstract.decl = class_decl @TwoAbstract, ()
// CHECK:STDOUT:   %Virtual.decl = class_decl @Virtual, ()
// CHECK:STDOUT:   %WrongOrder.decl = class_decl @WrongOrder, ()
// CHECK:STDOUT:   %AbstractAndBase.decl = class_decl @AbstractAndBase, ()
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

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:18: ERROR: `private` repeated on declaration.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR:                  ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `private` previously appeared here.
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private abstract private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:14: ERROR: `protected` not allowed on declaration with `private`.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `private` previously appeared here.
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private base protected class TwoAccess {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: ERROR: `abstract` repeated on declaration.
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract abstract class TwoAbstract;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: ERROR: `protected` not allowed on `class` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:11: ERROR: `virtual` not allowed on `class` declaration.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:19: ERROR: `base` not allowed on declaration with `virtual`.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:11: `virtual` previously appeared here.
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: ERROR: `protected` must appear before `abstract`.
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract protected class WrongOrder;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base class AbstractAndBase {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `extern` not allowed on `class` declaration that provides a definition.
// CHECK:STDERR: extern class ExternDefined {}
// CHECK:STDERR: ^~~~~~
extern class ExternDefined {}

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
// CHECK:STDOUT:   %ExternDefined: type = class_type @ExternDefined [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .DuplicatePrivate = %DuplicatePrivate.decl
// CHECK:STDOUT:     .TwoAccess = %TwoAccess.decl
// CHECK:STDOUT:     .TwoAbstract = %TwoAbstract.decl
// CHECK:STDOUT:     .Virtual = %Virtual.decl
// CHECK:STDOUT:     .WrongOrder = %WrongOrder.decl
// CHECK:STDOUT:     .AbstractAndBase = %AbstractAndBase.decl
// CHECK:STDOUT:     .ExternDefined = %ExternDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %DuplicatePrivate.decl: type = class_decl @DuplicatePrivate [template = constants.%DuplicatePrivate] {}
// CHECK:STDOUT:   %TwoAccess.decl: type = class_decl @TwoAccess [template = constants.%TwoAccess] {}
// CHECK:STDOUT:   %TwoAbstract.decl: type = class_decl @TwoAbstract [template = constants.%TwoAbstract] {}
// CHECK:STDOUT:   %Virtual.decl: type = class_decl @Virtual [template = constants.%Virtual] {}
// CHECK:STDOUT:   %WrongOrder.decl: type = class_decl @WrongOrder [template = constants.%WrongOrder] {}
// CHECK:STDOUT:   %AbstractAndBase.decl: type = class_decl @AbstractAndBase [template = constants.%AbstractAndBase] {}
// CHECK:STDOUT:   %ExternDefined.decl: type = class_decl @ExternDefined [template = constants.%ExternDefined] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DuplicatePrivate;
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAccess {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%TwoAccess
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAbstract;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Virtual {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Virtual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongOrder;
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractAndBase {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractAndBase
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExternDefined {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ExternDefined
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:18: error: `private` repeated on declaration
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR:                  ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here
// CHECK:STDERR: private abstract private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private abstract private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:14: error: `protected` not allowed on declaration with `private`
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here
// CHECK:STDERR: private base protected class TwoAccess {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private base protected class TwoAccess {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `abstract` repeated on declaration
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here
// CHECK:STDERR: abstract abstract class TwoAbstract;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract abstract class TwoAbstract;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: error: `protected` not allowed on `class` declaration at file scope, `protected` is only allowed on class members
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:11: error: `virtual` not allowed on `class` declaration
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:19: error: `base` not allowed on declaration with `virtual`
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:11: note: `virtual` previously appeared here
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `protected` must appear before `abstract`
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here
// CHECK:STDERR: abstract protected class WrongOrder;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract protected class WrongOrder;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `base` not allowed on declaration with `abstract`
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base class AbstractAndBase {}

abstract class AbstractWithDefinition {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:19: error: definition of `abstract` function
  // CHECK:STDERR:   abstract fn F() {}
  // CHECK:STDERR:                   ^
  // CHECK:STDERR:
  abstract fn F() {}
  abstract fn G();
}
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:31: error: definition of `abstract` function
// CHECK:STDERR: fn AbstractWithDefinition.G() {
// CHECK:STDERR:                               ^
fn AbstractWithDefinition.G() {
}

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DuplicatePrivate: type = class_type @DuplicatePrivate [template]
// CHECK:STDOUT:   %TwoAccess: type = class_type @TwoAccess [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %TwoAbstract: type = class_type @TwoAbstract [template]
// CHECK:STDOUT:   %Virtual: type = class_type @Virtual [template]
// CHECK:STDOUT:   %WrongOrder: type = class_type @WrongOrder [template]
// CHECK:STDOUT:   %AbstractAndBase: type = class_type @AbstractAndBase [template]
// CHECK:STDOUT:   %AbstractWithDefinition: type = class_type @AbstractWithDefinition [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DuplicatePrivate [private] = %DuplicatePrivate.decl
// CHECK:STDOUT:     .TwoAccess [private] = %TwoAccess.decl
// CHECK:STDOUT:     .TwoAbstract = %TwoAbstract.decl
// CHECK:STDOUT:     .Virtual = %Virtual.decl
// CHECK:STDOUT:     .WrongOrder = %WrongOrder.decl
// CHECK:STDOUT:     .AbstractAndBase = %AbstractAndBase.decl
// CHECK:STDOUT:     .AbstractWithDefinition = %AbstractWithDefinition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DuplicatePrivate.decl: type = class_decl @DuplicatePrivate [template = constants.%DuplicatePrivate] {} {}
// CHECK:STDOUT:   %TwoAccess.decl: type = class_decl @TwoAccess [template = constants.%TwoAccess] {} {}
// CHECK:STDOUT:   %TwoAbstract.decl: type = class_decl @TwoAbstract [template = constants.%TwoAbstract] {} {}
// CHECK:STDOUT:   %Virtual.decl: type = class_decl @Virtual [template = constants.%Virtual] {} {}
// CHECK:STDOUT:   %WrongOrder.decl: type = class_decl @WrongOrder [template = constants.%WrongOrder] {} {}
// CHECK:STDOUT:   %AbstractAndBase.decl: type = class_decl @AbstractAndBase [template = constants.%AbstractAndBase] {} {}
// CHECK:STDOUT:   %AbstractWithDefinition.decl: type = class_decl @AbstractWithDefinition [template = constants.%AbstractWithDefinition] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DuplicatePrivate;
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAccess {
// CHECK:STDOUT:   %.loc27: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%TwoAccess
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAbstract;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Virtual {
// CHECK:STDOUT:   %.loc53: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Virtual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongOrder;
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractAndBase {
// CHECK:STDOUT:   %.loc71: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractAndBase
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractWithDefinition {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %.loc80: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractWithDefinition
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_method_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_method_modifiers.carbon

class FinalClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: error: `abstract` not allowed; requires `abstract` class scope [ModifierAbstractNotAllowed]
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract fn Abstract[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: error: `virtual` not allowed; requires `abstract` or `base` class scope [ModifierVirtualNotAllowed]
  // CHECK:STDERR:   virtual fn Virtual[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-14]]:1: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn Virtual[self: Self]();
}

abstract class AbstractClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default fn Default[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn Default[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+4]]:3: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   final fn Final[self: Self]();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final fn Final[self: Self]();
}

base class BaseClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: error: `abstract` not allowed; requires `abstract` class scope [ModifierAbstractNotAllowed]
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR: base class BaseClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract fn Abstract[self: Self]();
}

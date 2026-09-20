// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/fail_method_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/fail_method_modifiers.carbon

class FinalClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+3]]:3-10: error: `abstract` not allowed; requires `abstract` class scope [ModifierAbstractNotAllowed]
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-3]]:1-18: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR:
  abstract fn Abstract(self);

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+3]]:3-9: error: `virtual` not allowed; requires `abstract` or `base` class scope [ModifierVirtualNotAllowed]
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-8]]:1-18: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR:
  virtual fn Virtual(self);
}

abstract class AbstractClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+2]]:3-9: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:
  default fn Default(self);

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+2]]:3-7: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:
  final fn Final(self);
}

base class BaseClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+3]]:3-10: error: `abstract` not allowed; requires `abstract` class scope [ModifierAbstractNotAllowed]
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-3]]:1-22: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR:
  abstract fn Abstract(self);
}

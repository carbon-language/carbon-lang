// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_base_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_base_modifiers.carbon

base class B {}

class C1 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+2]]:3-9: error: `private` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  private extend base: B;
}

class C2 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3-10: error: `abstract` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+2]]:3-19: error: missing `extend` before `base` declaration [BaseMissingExtend]
  // CHECK:STDERR:
  abstract base: B;
}

class C3 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+2]]:10-16: error: `default` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  extend default base: B;
}

class C4 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:10-15: error: `extend` repeated on declaration [ModifierRepeated]
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+2]]:3-8: note: `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:
  extend extend base: B;
}

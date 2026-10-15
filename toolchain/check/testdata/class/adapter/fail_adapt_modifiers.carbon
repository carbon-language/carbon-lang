// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_modifiers.carbon

class B {}

class C1 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+2]]:3-9: error: `private` not allowed on `adapt` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  private adapt B;
}

class C2 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+2]]:3-10: error: `abstract` not allowed on `adapt` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  abstract adapt B;
}

class C3 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+2]]:3-9: error: `default` not allowed on `adapt` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  default adapt B;
}

class C4 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+3]]:10-15: error: `extend` repeated on declaration [ModifierRepeated]
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+2]]:3-8: note: `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:
  extend extend adapt B;
}

class C5 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+2]]:3-6: error: `base` not allowed on `adapt` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  base adapt B;
}

class C6 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+2]]:10-13: error: `base` not allowed on `adapt` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:
  extend base adapt B;
}

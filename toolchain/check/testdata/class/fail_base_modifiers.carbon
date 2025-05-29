// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_modifiers.carbon

base class B {}

class C1 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: error: `private` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   private extend base: B;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private extend base: B;
}

class C2 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+8]]:3: error: `abstract` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: error: missing `extend` before `base` declaration [BaseMissingExtend]
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract base: B;
}

class C3 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:10: error: `default` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   extend default base: B;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  extend default base: B;
}

class C4 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+7]]:10: error: `extend` repeated on declaration [ModifierRepeated]
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: note: `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend extend base: B;
}

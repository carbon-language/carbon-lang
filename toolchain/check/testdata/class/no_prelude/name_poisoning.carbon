// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/name_poisoning.carbon

// --- fail_extend_poison_class_members.carbon

library "[[@TEST_NAME]]";

base class B {}

class C {
  // CHECK:STDERR: fail_extend_poison_class_members.carbon:[[@LINE+3]]:16: error: name `B` used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:   extend base: B;
  // CHECK:STDERR:                ^
  extend base: B;

  // CHECK:STDERR: fail_extend_poison_class_members.carbon:[[@LINE+4]]:6: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   fn B();
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  fn B();
}

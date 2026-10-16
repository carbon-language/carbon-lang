// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_redefinition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_redefinition.carbon

class Class {
  fn F();
  fn H();
  fn I() {}
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+3]]:1-13: error: redefinition of `class Class` [RedeclRedef]
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-7]]:1-13: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR:
class Class {
  fn G();
  fn H();
  fn I() {}
}

fn Class.F() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+3]]:10: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-17]]:1-13: note: no declaration of `G` in this scope [QualifiedDeclScopeEntity]
// CHECK:STDERR:
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+3]]:1-14: error: redefinition of `fn I` [RedeclRedef]
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-19]]:3-10: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR:
fn Class.I() {}

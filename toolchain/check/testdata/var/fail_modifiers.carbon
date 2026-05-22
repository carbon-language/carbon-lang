// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected var b: () = ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected var b: () = ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `protected` not allowed on declaration with `private` [ModifierNotAllowedWith]
// CHECK:STDERR: private protected var c: () = ();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private protected var c: () = ();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected var c: () = ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:11: error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: protected protected var d: () = ();
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected protected var d: () = ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected protected var d: () = ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected protected var d: () = ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: abstract var e: () = ();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract var e: () = ();

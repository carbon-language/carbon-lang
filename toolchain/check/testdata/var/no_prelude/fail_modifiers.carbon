// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected var b: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected var b: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `protected` not allowed on declaration with `private` [ModifierNotAllowedWith]
// CHECK:STDERR: private protected var c: ();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private protected var c: ();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected var c: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:11: error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected protected var d: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: error: `abstract` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: abstract var e: ();
// CHECK:STDERR: ^~~~~~~~
abstract var e: ();

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c [private] = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_19.2: type = converted %.loc15_19.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   %.loc24_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_27.2: type = converted %.loc24_27.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var c
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var
// CHECK:STDOUT:   %.loc37_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc37_29.2: type = converted %.loc37_29.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc42_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc42_18.2: type = converted %.loc42_18.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:

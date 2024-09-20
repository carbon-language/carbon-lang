// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed on `var` declaration at file scope, `protected` is only allowed on class members
// CHECK:STDERR: protected var b: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected var b: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `protected` not allowed on declaration with `private`
// CHECK:STDERR: private protected var c: ();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `private` previously appeared here
// CHECK:STDERR: private protected var c: ();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected var c: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `protected` not allowed on `var` declaration at file scope, `protected` is only allowed on class members
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:11: error: `protected` repeated on declaration
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `protected` previously appeared here
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected protected var d: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: error: `abstract` not allowed on `var` declaration
// CHECK:STDERR: abstract var e: ();
// CHECK:STDERR: ^~~~~~~~
abstract var e: ();

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c [private] = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_19.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc15_19.2: type = converted %.loc15_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref %.1 = var b
// CHECK:STDOUT:   %b: ref %.1 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc24_27.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc24_27.2: type = converted %.loc24_27.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %c.var: ref %.1 = var c
// CHECK:STDOUT:   %c: ref %.1 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc37_29.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc37_29.2: type = converted %.loc37_29.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc42_18.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc42_18.2: type = converted %.loc42_18.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref %.1 = var e
// CHECK:STDOUT:   %e: ref %.1 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:11: error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected protected var d: ();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected protected var d: ();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: abstract var e: ();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract var e: ();

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .b [protected] = %b
// CHECK:STDOUT:     .c [private] = %c
// CHECK:STDOUT:     .d [protected] = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:     %.loc15_11: %empty_tuple.type = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %.loc15_19.1: type = splice_block %.loc15_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc15_19.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_19.3: type = converted %.loc15_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %empty_tuple.type = binding_pattern c
// CHECK:STDOUT:     %.loc24_19: %empty_tuple.type = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var c
// CHECK:STDOUT:   %.loc24_27.1: type = splice_block %.loc24_27.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc24_27.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc24_27.3: type = converted %.loc24_27.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc37_21: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc37_29.1: type = splice_block %.loc37_29.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc37_29.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc37_29.3: type = converted %.loc37_29.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc43_10: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc43_18.1: type = splice_block %.loc43_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc43_18.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc43_18.3: type = converted %.loc43_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_modifiers.carbon

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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
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
// CHECK:STDOUT:     %b.patt: %pattern_type = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc19_19.1: type = splice_block %.loc19_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc19_19.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_19.3: type = converted %.loc19_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc28_27.1: type = splice_block %.loc28_27.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc28_27.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc28_27.3: type = converted %.loc28_27.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc41_29.1: type = splice_block %.loc41_29.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc41_29.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc41_29.3: type = converted %.loc41_29.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc47_18.1: type = splice_block %.loc47_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc47_18.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc47_18.3: type = converted %.loc47_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:

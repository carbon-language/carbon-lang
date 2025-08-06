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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_namespace_conflict.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_namespace_conflict.carbon

namespace A;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+7]]:5: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: var A: ();
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-5]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var A: ();

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+7]]:5: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: var A: () = ();
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-14]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var A: () = ();

// CHECK:STDOUT: --- fail_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc15: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt.loc24: %pattern_type = binding_pattern A [concrete]
// CHECK:STDOUT:     %A.var_patt.loc24: %pattern_type = var_pattern %A.patt.loc24 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.var.loc24: ref %empty_tuple.type = var %A.var_patt.loc24 [concrete]
// CHECK:STDOUT:   %.loc24_9.1: type = splice_block %.loc24_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc24_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc24_9.3: type = converted %.loc24_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc24: ref %empty_tuple.type = bind_name A, %A.var.loc24 [concrete = %A.var.loc24]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt.loc33: %pattern_type = binding_pattern A [concrete]
// CHECK:STDOUT:     %A.var_patt.loc33: %pattern_type = var_pattern %A.patt.loc33 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.var.loc33: ref %empty_tuple.type = var %A.var_patt.loc33 [concrete]
// CHECK:STDOUT:   %.loc33_9.1: type = splice_block %.loc33_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc33_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc33_9.3: type = converted %.loc33_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc33: ref %empty_tuple.type = bind_name A, %A.var.loc33 [concrete = %A.var.loc33]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc33_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc33_14.2: init %empty_tuple.type = tuple_init () to file.%A.var.loc33 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc33_1: init %empty_tuple.type = converted %.loc33_14.1, %.loc33_14.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%A.var.loc33, %.loc33_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

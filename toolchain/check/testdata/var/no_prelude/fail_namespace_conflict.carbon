// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_namespace_conflict.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_namespace_conflict.carbon

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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc11: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt.loc20: %empty_tuple.type = binding_pattern A
// CHECK:STDOUT:     %.loc20_1: %empty_tuple.type = var_pattern %A.patt.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.var.loc20: ref %empty_tuple.type = var A
// CHECK:STDOUT:   %.loc20_9.1: type = splice_block %.loc20_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc20_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc20_9.3: type = converted %.loc20_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc20: ref %empty_tuple.type = bind_name A, %A.var.loc20
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt.loc29: %empty_tuple.type = binding_pattern A
// CHECK:STDOUT:     %.loc29_1: %empty_tuple.type = var_pattern %A.patt.loc29
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.var.loc29: ref %empty_tuple.type = var A
// CHECK:STDOUT:   %.loc29_9.1: type = splice_block %.loc29_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc29_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc29_9.3: type = converted %.loc29_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc29: ref %empty_tuple.type = bind_name A, %A.var.loc29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc29_14.2: init %empty_tuple.type = tuple_init () to file.%A.var.loc29 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_1: init %empty_tuple.type = converted %.loc29_14.1, %.loc29_14.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%A.var.loc29, %.loc29_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

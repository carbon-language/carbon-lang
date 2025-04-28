// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_not_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_not_constant.carbon

var a: () = ();
var b: ()* = &a;

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+4]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias c = *b;
// CHECK:STDERR:           ^~
// CHECK:STDERR:
alias c = *b;

// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc11_9.1: type = splice_block %.loc11_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc11_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_9.3: type = converted %.loc11_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %ptr = binding_pattern b
// CHECK:STDOUT:     %.loc12_1: %ptr = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %ptr = var b
// CHECK:STDOUT:   %.loc12_10.1: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %.loc12_9: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_10.2: type = converted %.loc12_9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %ptr: type = ptr_type %.loc12_10.2 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr = bind_name b, %b.var
// CHECK:STDOUT:   %b.ref: ref %ptr = name_ref b, %b
// CHECK:STDOUT:   %.loc18_12: %ptr = bind_value %b.ref
// CHECK:STDOUT:   %.loc18_11: ref %empty_tuple.type = deref %.loc18_12
// CHECK:STDOUT:   %c: <error> = bind_alias c, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_14.2: init %empty_tuple.type = tuple_init () to file.%a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_1: init %empty_tuple.type = converted %.loc11_14.1, %.loc11_14.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, file.%a
// CHECK:STDOUT:   %addr: %ptr = addr_of %a.ref
// CHECK:STDOUT:   assign file.%b.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/no_prelude/fail_assign_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/no_prelude/fail_assign_empty.carbon

// CHECK:STDERR: fail_assign_empty.carbon:[[@LINE+4]]:16: error: cannot initialize tuple of 1 element from tuple with 0 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((),) = ();
// CHECK:STDERR:                ^~
// CHECK:STDERR:
var x: ((),) = ();

// CHECK:STDOUT: --- fail_assign_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc15_1: %tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type = var x
// CHECK:STDOUT:   %.loc15_12.1: type = splice_block %.loc15_12.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc15_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: %tuple.type = tuple_literal (%.loc15_10)
// CHECK:STDOUT:     %.loc15_12.3: type = converted %.loc15_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_12.4: type = converted %.loc15_12.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/no_prelude/empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/no_prelude/empty.carbon

var x: () = ();
var y: () = x;

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc11_1: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc11_9.1: type = splice_block %.loc11_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc11_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_9.3: type = converted %.loc11_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %empty_tuple.type = binding_pattern y
// CHECK:STDOUT:     %.loc12_1: %empty_tuple.type = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var y
// CHECK:STDOUT:   %.loc12_9.1: type = splice_block %.loc12_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_9.3: type = converted %.loc12_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_14.2: init %empty_tuple.type = tuple_init () to file.%x.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_1: init %empty_tuple.type = converted %.loc11_14.1, %.loc11_14.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_1
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_13: init %empty_tuple.type = tuple_init () to file.%y.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_1: init %empty_tuple.type = converted %x.ref, %.loc12_13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc12_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

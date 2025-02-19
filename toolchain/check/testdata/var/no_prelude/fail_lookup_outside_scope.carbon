// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_lookup_outside_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_lookup_outside_scope.carbon

fn Main() {
  var x: ();
}

// CHECK:STDERR: fail_lookup_outside_scope.carbon:[[@LINE+4]]:13: error: name `x` not found [NameNotFound]
// CHECK:STDERR: var y: () = x;
// CHECK:STDERR:             ^
// CHECK:STDERR:
var y: () = x;

// CHECK:STDOUT: --- fail_lookup_outside_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .x = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %empty_tuple.type = binding_pattern y
// CHECK:STDOUT:     %.loc19_1: %empty_tuple.type = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var y
// CHECK:STDOUT:   %.loc19_9.1: type = splice_block %.loc19_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc19_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_9.3: type = converted %.loc19_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc12_3: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc12_11.1: type = splice_block %.loc12_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_11.3: type = converted %.loc12_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

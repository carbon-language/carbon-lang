// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_not_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_not_constant.carbon

fn F() {
  var a: () = ();
  var b: ()* = &a;
  // CHECK:STDERR: fail_not_constant.carbon:[[@LINE+3]]:13: error: alias initializer must be a name reference
  // CHECK:STDERR:   alias c = *b;
  // CHECK:STDERR:             ^~
  alias c = *b;
}

// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc12_11.2: type = converted %.loc12_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_16.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.2: init %.1 = tuple_init () to %a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_17: init %.1 = converted %.loc12_16.1, %.loc12_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %a.var, %.loc12_17
// CHECK:STDOUT:   %.loc13_11: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_12.1: type = converted %.loc13_11, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_12.2: type = ptr_type %.1 [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref %.2 = var b
// CHECK:STDOUT:   %b: ref %.2 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref %.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc13_16: %.2 = addr_of %a.ref
// CHECK:STDOUT:   assign %b.var, %.loc13_16
// CHECK:STDOUT:   %b.ref: ref %.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc17_14: %.2 = bind_value %b.ref
// CHECK:STDOUT:   %.loc17_13: ref %.1 = deref %.loc17_14
// CHECK:STDOUT:   %c: <error> = bind_alias c, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

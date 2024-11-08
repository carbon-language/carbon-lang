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
  // CHECK:STDERR: fail_not_constant.carbon:[[@LINE+3]]:13: error: alias initializer must be a name reference [AliasRequiresNameRef]
  // CHECK:STDERR:   alias c = *b;
  // CHECK:STDERR:             ^~
  alias c = *b;
}

// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type %empty_tuple.type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_11.2: type = converted %.loc12_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.2: init %empty_tuple.type = tuple_init () to %a.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_17: init %empty_tuple.type = converted %.loc12_16.1, %.loc12_16.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc12_17
// CHECK:STDOUT:   %.loc13_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_12.1: type = converted %.loc13_11, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc13_12.2: type = ptr_type %empty_tuple.type [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref %.1 = var b
// CHECK:STDOUT:   %b: ref %.1 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %.loc13_16: %.1 = addr_of %a.ref
// CHECK:STDOUT:   assign %b.var, %.loc13_16
// CHECK:STDOUT:   %b.ref: ref %.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc17_14: %.1 = bind_value %b.ref
// CHECK:STDOUT:   %.loc17_13: ref %empty_tuple.type = deref %.loc17_14
// CHECK:STDOUT:   %c: <error> = bind_alias c, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/no_prelude/fail_explicit_self_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/no_prelude/fail_explicit_self_param.carbon

// CHECK:STDERR: fail_explicit_self_param.carbon:[[@LINE+3]]:6: error: `self` can only be declared in an implicit parameter list
// CHECK:STDERR: fn F(self: ());
// CHECK:STDERR:      ^~~~~~~~
fn F(self: ());

fn Run() {
  F(());
}

// CHECK:STDOUT: --- fail_explicit_self_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc14_13.2: type = converted %.loc14_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %self.param: %.1 = param self, runtime_param0
// CHECK:STDOUT:     %self: %.1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self: %.1);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc17_6: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc17_4: %.1 = converted %.loc17_6, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref(%.loc17_4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

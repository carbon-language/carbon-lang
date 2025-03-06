// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/no_prelude/fail_explicit_self_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/no_prelude/fail_explicit_self_param.carbon

// CHECK:STDERR: fail_explicit_self_param.carbon:[[@LINE+4]]:6: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
// CHECK:STDERR: fn F(self: ());
// CHECK:STDERR:      ^~~~~~~~
// CHECK:STDERR:
fn F(self: ());

fn Run() {
  F(());
}

// CHECK:STDOUT: --- fail_explicit_self_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc15_13.1: type = splice_block %.loc15_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_13.3: type = converted %.loc15_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_6.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_6.2: %empty_tuple.type = converted %.loc18_6.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc18_6.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

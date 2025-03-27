// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_local_in_namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_local_in_namespace.carbon

namespace NS;

fn F() -> {} {
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+8]]:9: error: name `NS` not found [NameNotFound]
  // CHECK:STDERR:   alias NS.a = {};
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+4]]:16: error: alias initializer must be a name reference [AliasRequiresNameRef]
  // CHECK:STDERR:   alias NS.a = {};
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  alias NS.a = {};
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+4]]:10: error: member name `a` not found in `NS` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return NS.a;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return NS.a;
}

// CHECK:STDOUT: --- fail_local_in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_12.2: type = converted %.loc13_12.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_17: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_12: <error> = bind_alias <none>, <error> [concrete = <error>]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [concrete = file.%NS]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

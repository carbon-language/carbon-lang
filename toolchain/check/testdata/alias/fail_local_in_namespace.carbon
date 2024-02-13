// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

fn F() -> bool {
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+3]]:9: ERROR: Name `NS` not found.
  // CHECK:STDERR:   alias NS.a = false;
  // CHECK:STDERR:         ^~
  alias NS.a = false;
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+3]]:10: ERROR: Name `a` not found.
  // CHECK:STDERR:   return NS.a;
  // CHECK:STDERR:          ^~~~
  return NS.a;
}

// CHECK:STDOUT: --- fail_local_in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc7, .F = %F} [template]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_16: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_9: bool = bind_template_name <invalid>, %.loc13_16 [template = constants.%.1]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%.loc7 [template = file.%.loc7]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

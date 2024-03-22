// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

fn F() -> bool {
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+8]]:9: ERROR: Name `NS` not found.
  // CHECK:STDERR:   alias NS.a = false;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+4]]:16: ERROR: Alias initializer must be a name reference.
  // CHECK:STDERR:   alias NS.a = false;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_16: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_9: <error> = bind_alias <invalid>, <error> [template = <error>]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

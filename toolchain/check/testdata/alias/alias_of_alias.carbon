// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias a = bool;
// CHECK:STDERR: alias_of_alias.carbon:[[@LINE+6]]:11: ERROR: Cannot implicitly convert from `type` to `bool`.
// CHECK:STDERR: alias b = not a;
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: alias_of_alias.carbon:[[@LINE+3]]:11: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias b = not a;
// CHECK:STDERR:           ^~~~~
alias b = not a;
// CHECK:STDERR: alias_of_alias.carbon:[[@LINE+3]]:11: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias c = not b;
// CHECK:STDERR:           ^~~~~
alias c = not b;
let d: bool = c;

// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a.ref.loc7, .b = %b.ref.loc14, .c = %c.ref.loc18} [template]
// CHECK:STDOUT:   %a.ref.loc7: type = name_ref a, bool [template = bool]
// CHECK:STDOUT:   %a.ref.loc14: type = name_ref a, %a.ref.loc7 [template = bool]
// CHECK:STDOUT:   %.loc14: <error> = not <error> [template = <error>]
// CHECK:STDOUT:   %b.ref.loc14: <error> = name_ref b, %.loc14 [template = <error>]
// CHECK:STDOUT:   %b.ref.loc18: <error> = name_ref b, %b.ref.loc14 [template = <error>]
// CHECK:STDOUT:   %.loc18: <error> = not <error> [template = <error>]
// CHECK:STDOUT:   %c.ref.loc18: <error> = name_ref c, %.loc18 [template = <error>]
// CHECK:STDOUT:   %c.ref.loc19: <error> = name_ref c, %c.ref.loc18 [template = <error>]
// CHECK:STDOUT:   %d: bool = bind_name d, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

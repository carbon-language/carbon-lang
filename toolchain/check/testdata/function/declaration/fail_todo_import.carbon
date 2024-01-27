// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TODO: ResolveIfImportRefUnused for FunctionDecl`.

// --- a.carbon

library "a" api;

fn F();

// --- b.carbon

library "b" api;

import library "a";

fn Run() {
  F();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %import_ref, .Run = %Run} [template]
// CHECK:STDOUT:   %import_ref: <error> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, file.%import_ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- other_fn.carbon

package Other library "fn" api;

fn F() {}

// --- other_fn_conflict.carbon

package Other library "fn_conflict" api;

fn F() {}

// --- other_fn2.carbon

package Other library "fn2" api;

fn F2() {}

// --- main_other_ns.carbon

library "other_ns" api;

namespace Other;

// ============================================================================
// Test files
// ============================================================================

// --- fail_main_use_other.carbon
// TODO: Once name lookup works, nothing should fail here.

library "use_other" api;

import Other library "fn";
import Other library "fn2";

fn Run() {
  // CHECK:STDERR: fail_main_use_other.carbon:[[@LINE+3]]:3: ERROR: Name `F` not found.
  // CHECK:STDERR:   Other.F();
  // CHECK:STDERR:   ^~~~~~~
  Other.F();
  // CHECK:STDERR: fail_main_use_other.carbon:[[@LINE+3]]:3: ERROR: Name `F2` not found.
  // CHECK:STDERR:   Other.F2();
  // CHECK:STDERR:   ^~~~~~~~
  Other.F2();
}

// --- main_unused_other_ambiguous.carbon

library "unused_other_ambiguous" api;

import Other library "fn";
import Other library "fn_conflict";

// --- fail_main_use_other_ambiguous.carbon
// TODO: Once name lookup works, this should be ambiguous.

library "use_other_ambiguous" api;

import Other library "fn";
import Other library "fn_conflict";

fn Run() {
  // CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+3]]:3: ERROR: Name `F` not found.
  // CHECK:STDERR:   Other.F();
  // CHECK:STDERR:   ^~~~~~~
  Other.F();
}

// --- fail_main_namespace_conflict.carbon

library "namespace_conflict" api;

import library "other_ns";
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: main_other_ns.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: namespace Other;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
import Other library "fn";

fn Other.F() {}

// --- fail_main_reopen_other.carbon

library "reopen_other" api;

import Other library "fn";

// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE+6]]:11: ERROR: Imported packages cannot be used for declarations.
// CHECK:STDERR: namespace Other;
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE-5]]:1: Package imported here.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
namespace Other;

// This is not diagnosed after the diagnostic on `namespace Other;`.
fn Other.G() {}

// --- fail_main_add_to_other.carbon

library "add_to_other" api;

import Other library "fn";

// CHECK:STDERR: fail_main_add_to_other.carbon:[[@LINE+6]]:4: ERROR: Imported packages cannot be used for declarations.
// CHECK:STDERR: fn Other.G() {}
// CHECK:STDERR:    ^~~~~
// CHECK:STDERR: fail_main_add_to_other.carbon:[[@LINE-5]]:1: Package imported here.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
fn Other.G() {}

// CHECK:STDOUT: --- other_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F2 = %F2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2: <function> = fn_decl @F2 [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_other_ns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_use_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref.loc12: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   %Other.ref.loc16: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F2.ref: <error> = name_ref F2, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_unused_other_ambiguous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_use_other_ambiguous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %Other: <namespace> = namespace %import_ref, [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_reopen_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_add_to_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

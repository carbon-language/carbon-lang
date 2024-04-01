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

// --- other_fn_extern.carbon

package Other library "fn_extern" api;

extern fn F();

// --- other_fn_conflict.carbon

package Other library "fn_conflict" api;

fn F(x: i32) {}

// --- other_fn2.carbon

package Other library "fn2" api;

fn F2() {}

// --- main_other_ns.carbon

library "other_ns" api;

namespace Other;

// ============================================================================
// Test files
// ============================================================================

// --- main_use_other.carbon

library "use_other" api;

import Other library "fn";
import Other library "fn2";

fn Run() {
  Other.F();
  Other.F2();
}

// --- fail_main_use_other_extern.carbon

library "use_other_extern" api;

import Other library "fn";
// CHECK:STDERR: fail_main_use_other_extern.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import Other library "fn_extern";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: other_fn_extern.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_main_use_other_extern.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: other_fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
import Other library "fn_extern";

fn Run() {
  // CHECK:STDERR: fail_main_use_other_extern.carbon:[[@LINE+4]]:3: In name lookup for `F`.
  // CHECK:STDERR:   Other.F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Other.F();
}

// --- main_unused_other_ambiguous.carbon

library "unused_other_ambiguous" api;

import Other library "fn";
import Other library "fn_conflict";

// --- fail_main_use_other_ambiguous.carbon

library "use_other_ambiguous" api;

import Other library "fn";
// CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import Other library "fn_conflict";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: other_fn_conflict.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn F(x: i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: other_fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
import Other library "fn_conflict";

fn Run() {
  // CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+4]]:3: In name lookup for `F`.
  // CHECK:STDERR:   Other.F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Other.F();
}

// --- fail_main_namespace_conflict.carbon

library "namespace_conflict" api;

import library "other_ns";
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE-4]]:1: In import.
// CHECK:STDERR: import library "other_ns";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: main_other_ns.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: namespace Other;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other library "fn";

// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+10]]:1: ERROR: Only one library can declare function F without `extern`.
// CHECK:STDERR: fn Other.F() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: other_fn.carbon:4:1: Previously declared here.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn Other.F() {}

// --- fail_main_reopen_other.carbon

library "reopen_other" api;

import Other library "fn";

// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE+7]]:11: ERROR: Imported packages cannot be used for declarations.
// CHECK:STDERR: namespace Other;
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE-5]]:1: Package imported here.
// CHECK:STDERR: import Other library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
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
// CHECK:STDOUT: --- other_fn_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %x.loc4_6.1: i32 = param x
// CHECK:STDOUT:     @F.%x: i32 = bind_name x, %x.loc4_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: i32) {
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
// CHECK:STDOUT: --- main_use_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%F]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir2, inst+1, used [template = imports.%F2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref.loc8: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%import_ref.1 [template = imports.%F]
// CHECK:STDOUT:   %.loc8: init () = call %F.ref()
// CHECK:STDOUT:   %Other.ref.loc9: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F2.ref: <function> = name_ref F2, file.%import_ref.2 [template = imports.%F2]
// CHECK:STDOUT:   %.loc9: init () = call %F2.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_use_other_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%F.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir2, inst+1, used [template = imports.%F.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%import_ref.1 [template = imports.%F.1]
// CHECK:STDOUT:   %.loc24: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F.2();
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
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%F.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir2, inst+3, used [template = imports.%F.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%import_ref.1 [template = imports.%F.1]
// CHECK:STDOUT:   %.loc24: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%x: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %Other: <namespace> = namespace %import_ref.1, [template] {}
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir2, inst+1, used [template = imports.%F]
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
// CHECK:STDOUT:   %.loc13: <namespace> = namespace [template] {}
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

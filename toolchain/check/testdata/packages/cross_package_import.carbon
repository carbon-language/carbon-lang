// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/cross_package_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/cross_package_import.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- other_fn.carbon

package Other library "[[@TEST_NAME]]";

fn F() {}

// --- other_fn_extern.carbon

package Other library "[[@TEST_NAME]]";

extern fn F();

// --- other_fn_conflict.carbon

package Other library "[[@TEST_NAME]]";

fn F(unused x: ()) {}

// --- other_fn2.carbon

package Other library "[[@TEST_NAME]]";

fn F2() {}

// --- other_fn_use.carbon

package Other library "[[@TEST_NAME]]";

import library "other_fn";

fn G() { F(); }

// --- main_other_ns.carbon

library "[[@TEST_NAME]]";

namespace Other;

// ============================================================================
// Test files
// ============================================================================

// --- main_use_other.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";
import Other library "other_fn2";

fn Run() {
  Other.F();
  Other.F2();
}

// --- fail_todo_main_use_other_extern.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: imported from: fail_todo_main_use_other_extern.carbon:[[@LINE+4]]:1 [InImport]
// CHECK:STDERR: other_fn_extern.carbon:4:1-14: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_todo_main_use_other_extern.carbon:[[@LINE+2]]:1 [InImport]
// CHECK:STDERR: other_fn.carbon:4:1-8: note: name is previously declared here [NameDeclPrevious]
import Other library "other_fn";
import Other library "other_fn_extern";

fn Run() {
  // CHECK:STDERR: fail_todo_main_use_other_extern.carbon:[[@LINE+2]]:3-9: note: in name lookup for `F` [InNameLookup]
  // CHECK:STDERR:
  Other.F();
}

// --- main_unused_other_ambiguous.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";
import Other library "other_fn_conflict";

// --- fail_main_use_other_ambiguous.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: imported from: fail_main_use_other_ambiguous.carbon:[[@LINE+4]]:1 [InImport]
// CHECK:STDERR: other_fn_conflict.carbon:4:1-20: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_main_use_other_ambiguous.carbon:[[@LINE+2]]:1 [InImport]
// CHECK:STDERR: other_fn.carbon:4:1-8: note: name is previously declared here [NameDeclPrevious]
import Other library "other_fn";
import Other library "other_fn_conflict";

fn Run() {
  // CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+2]]:3-9: note: in name lookup for `F` [InNameLookup]
  // CHECK:STDERR:
  Other.F();
}

// --- fail_main_namespace_conflict.carbon

library "[[@TEST_NAME]]";

import library "main_other_ns";
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+4]]:1-32: error: duplicate name `Other` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_main_namespace_conflict.carbon:[[@LINE-2]]:1 [InImport]
// CHECK:STDERR: main_other_ns.carbon:4:1-16: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
import Other library "other_fn";

// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+4]]:1-14: error: redeclaration of `fn F` is redundant [RedeclRedundant]
// CHECK:STDERR: imported from: fail_main_namespace_conflict.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: other_fn.carbon:4:1-8: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn Other.F() {}

// --- fail_main_reopen_other.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";

// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE+3]]:11-15: error: duplicate name `Other` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE-3]]:1-32: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
namespace Other;

// This is not diagnosed after the diagnostic on `namespace Other;`.
fn Other.G() {}

// --- fail_main_reopen_other_nested.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";

// CHECK:STDERR: fail_main_reopen_other_nested.carbon:[[@LINE+3]]:11-15: error: imported packages cannot be used for declarations [QualifiedDeclOutsidePackage]
// CHECK:STDERR: fail_main_reopen_other_nested.carbon:[[@LINE-3]]:1-32: note: package imported here [QualifiedDeclOutsidePackageSource]
// CHECK:STDERR:
namespace Other.Nested;

// This is not diagnosed after the diagnostic on `namespace Other;`.
fn Other.Nested.F() {}

// --- fail_main_add_to_other.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";

// CHECK:STDERR: fail_main_add_to_other.carbon:[[@LINE+3]]:4-8: error: imported packages cannot be used for declarations [QualifiedDeclOutsidePackage]
// CHECK:STDERR: fail_main_add_to_other.carbon:[[@LINE-3]]:1-32: note: package imported here [QualifiedDeclOutsidePackageSource]
// CHECK:STDERR:
fn Other.G() {}

// --- fail_use_other_fn_use.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn_use";

// CHECK:STDERR: fail_use_other_fn_use.carbon:[[@LINE+2]]:13-19: error: member name `F` not found in `Other` [MemberNameNotFoundInInstScope]
// CHECK:STDERR:
fn UseF() { Other.F(); }

// CHECK:STDOUT: --- other_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_17.3: type = converted %.loc4_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [concrete = constants.%F2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.F: %F.type = import_ref Other//other_fn, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = imports.%Other.F
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "other_fn.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_other_ns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_use_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .F = %Other.F
// CHECK:STDOUT:     .F2 = %Other.F2
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.F: %F.type = import_ref Other//other_fn, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Other.F2: %F2.type = import_ref Other//other_fn2, F2, loaded [concrete = constants.%F2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref.loc8: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   %Other.ref.loc9: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F2.ref: %F2.type = name_ref F2, imports.%Other.F2 [concrete = constants.%F2]
// CHECK:STDOUT:   %F2.call: init %empty_tuple.type = call %F2.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "other_fn.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2 [from "other_fn2.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_main_use_other_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .F = %Other.F
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn_extern
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.F: %F.type = import_ref Other//other_fn, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "other_fn.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_unused_other_ambiguous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn_conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_use_other_ambiguous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .F = %Other.F
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn_conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.F: %F.type = import_ref Other//other_fn, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "other_fn.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.2a8c71.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.1be9f8.1: %F.type.2a8c71.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.2a8c71.2: type = fn_type @F.loc15 [concrete]
// CHECK:STDOUT:   %F.1be9f8.2: %F.type.2a8c71.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Other: <namespace> = import_ref Main//main_other_ns, Other, loaded
// CHECK:STDOUT:   %Other: <namespace> = namespace %Main.Other, [concrete] {
// CHECK:STDOUT:     .F = %Other.F
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.F: %F.type.2a8c71.1 = import_ref Other//other_fn, F, loaded [concrete = constants.%F.1be9f8.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %F.decl: %F.type.2a8c71.2 = fn_decl @F.loc15 [concrete = constants.%F.1be9f8.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1 [from "other_fn.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc15() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_reopen_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .G = file.%G.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [concrete] {
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_reopen_other_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .Nested = file.%Nested
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Nested: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_add_to_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .G = file.%G.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_other_fn_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UseF.type: type = fn_type @UseF [concrete]
// CHECK:STDOUT:   %UseF: %UseF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .F = <poisoned>
// CHECK:STDOUT:     import Other//other_fn_use
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .UseF = %UseF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %UseF.decl: %UseF.type = fn_decl @UseF [concrete = constants.%UseF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

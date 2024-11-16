// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/cross_package_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/cross_package_import.carbon

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

fn F(x: ()) {}

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

// CHECK:STDERR: fail_todo_main_use_other_extern.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: other_fn_extern.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_main_use_other_extern.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: other_fn.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
import Other library "other_fn";
import Other library "other_fn_extern";

fn Run() {
  // CHECK:STDERR: fail_todo_main_use_other_extern.carbon:[[@LINE+4]]:3: note: in name lookup for `F` [InNameLookup]
  // CHECK:STDERR:   Other.F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Other.F();
}

// --- main_unused_other_ambiguous.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";
import Other library "other_fn_conflict";

// --- fail_main_use_other_ambiguous.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: other_fn_conflict.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn F(x: ()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: other_fn.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
import Other library "other_fn";
import Other library "other_fn_conflict";

fn Run() {
  // CHECK:STDERR: fail_main_use_other_ambiguous.carbon:[[@LINE+4]]:3: note: in name lookup for `F` [InNameLookup]
  // CHECK:STDERR:   Other.F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Other.F();
}

// --- fail_main_namespace_conflict.carbon

library "[[@TEST_NAME]]";

import library "main_other_ns";
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+8]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: import Other library "other_fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE-4]]:1: in import [InImport]
// CHECK:STDERR: main_other_ns.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: namespace Other;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other library "other_fn";

// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE+8]]:1: error: redeclaration of `fn F` is redundant [RedeclRedundant]
// CHECK:STDERR: fn Other.F() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_main_namespace_conflict.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: other_fn.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn Other.F() {}

// --- fail_main_reopen_other.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";

// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE+7]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: namespace Other;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_main_reopen_other.carbon:[[@LINE-5]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: import Other library "other_fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
namespace Other;

// This is not diagnosed after the diagnostic on `namespace Other;`.
fn Other.G() {}

// --- fail_main_reopen_other_nested.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";

// CHECK:STDERR: fail_main_reopen_other_nested.carbon:[[@LINE+7]]:11: error: imported packages cannot be used for declarations [QualifiedDeclOutsidePackage]
// CHECK:STDERR: namespace Other.Nested;
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: fail_main_reopen_other_nested.carbon:[[@LINE-5]]:1: note: package imported here [QualifiedDeclOutsidePackageSource]
// CHECK:STDERR: import Other library "other_fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
namespace Other.Nested;

// This is not diagnosed after the diagnostic on `namespace Other;`.
fn Other.Nested.F() {}

// --- fail_main_add_to_other.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn";

// CHECK:STDERR: fail_main_add_to_other.carbon:[[@LINE+7]]:4: error: imported packages cannot be used for declarations [QualifiedDeclOutsidePackage]
// CHECK:STDERR: fn Other.G() {}
// CHECK:STDERR:    ^~~~~
// CHECK:STDERR: fail_main_add_to_other.carbon:[[@LINE-5]]:1: note: package imported here [QualifiedDeclOutsidePackageSource]
// CHECK:STDERR: import Other library "other_fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
fn Other.G() {}

// --- fail_use_other_fn_use.carbon

library "[[@TEST_NAME]]";

import Other library "other_fn_use";

// CHECK:STDERR: fail_use_other_fn_use.carbon:[[@LINE+3]]:13: error: name `F` not found [NameNotFound]
// CHECK:STDERR: fn UseF() { Other.F(); }
// CHECK:STDERR:             ^~~~~~~
fn UseF() { Other.F(); }

// CHECK:STDOUT: --- other_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_10.2: type = converted %.loc4_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_fn2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {} {}
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
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %F.type = import_ref Other//other_fn, inst+1, loaded [template = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = imports.%import_ref
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
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
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .F = %import_ref.1
// CHECK:STDOUT:     .F2 = %import_ref.2
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %F.type = import_ref Other//other_fn, inst+1, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.2: %F2.type = import_ref Other//other_fn2, inst+1, loaded [template = constants.%F2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref.loc8: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.1 [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   %Other.ref.loc9: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F2.ref: %F2.type = name_ref F2, imports.%import_ref.2 [template = constants.%F2]
// CHECK:STDOUT:   %F2.call: init %empty_tuple.type = call %F2.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_main_use_other_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .F = %import_ref.1
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn_extern
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %F.type = import_ref Other//other_fn, inst+1, loaded [template = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.1 [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_unused_other_ambiguous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn_conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_use_other_ambiguous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .F = %import_ref.1
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:     import Other//other_fn_conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %F.type = import_ref Other//other_fn, inst+1, loaded [template = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.1 [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref Main//main_other_ns, inst+1, loaded
// CHECK:STDOUT:   %Other: <namespace> = namespace %import_ref.1, [template] {
// CHECK:STDOUT:     .F = file.%F.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_reopen_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .G = file.%G.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .Nested = file.%Nested
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Nested: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
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
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .G = file.%G.decl
// CHECK:STDOUT:     import Other//other_fn
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
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
// CHECK:STDOUT:   %UseF.type: type = fn_type @UseF [template]
// CHECK:STDOUT:   %UseF: %UseF.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     import Other//other_fn_use
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .UseF = %UseF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %UseF.decl: %UseF.type = fn_decl @UseF [template = constants.%UseF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

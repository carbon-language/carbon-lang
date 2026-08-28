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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/extern_library.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/extern_library.carbon

// --- extern_library.carbon

library "[[@TEST_NAME]]";

extern library "extern_library_owner" fn F();

// --- extern_library_owner.carbon

library "[[@TEST_NAME]]";

import library "extern_library";

extern fn F();

// --- fail_extern_library_nonowner.carbon

library "[[@TEST_NAME]]";

import library "extern_library";

// CHECK:STDERR: fail_extern_library_nonowner.carbon:[[@LINE+4]]:1-14: error: declaration in library "extern_library_nonowner" doesn't match `extern library` declaration [ExternLibraryIncorrect]
// CHECK:STDERR: imported from: fail_extern_library_nonowner.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: extern_library.carbon:4:1-45: note: previously declared with `extern library` here [ExternLibraryExpected]
// CHECK:STDERR:
extern fn F();

// --- fail_extern_library_nonextern.carbon

library "[[@TEST_NAME]]";

import library "extern_library";

// CHECK:STDERR: fail_extern_library_nonextern.carbon:[[@LINE+4]]:1-7: error: redeclarations of `fn F` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: imported from: fail_extern_library_nonextern.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: extern_library.carbon:4:1-45: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn F();

// --- fail_extern_library_redecl.carbon

library "[[@TEST_NAME]]";

import library "extern_library";

// CHECK:STDERR: fail_extern_library_redecl.carbon:[[@LINE+4]]:1-45: error: declaration in library "extern_library_redecl" doesn't match `extern library` declaration [ExternLibraryIncorrect]
// CHECK:STDERR: imported from: fail_extern_library_redecl.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: extern_library.carbon:4:1-45: note: previously declared with `extern library` here [ExternLibraryExpected]
// CHECK:STDERR:
extern library "extern_library_owner" fn F();

// --- extern_library_copy.carbon

library "[[@TEST_NAME]]";

extern library "extern_library_owner" fn F();

// --- fail_extern_library_collision.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: imported from: fail_extern_library_collision.carbon:[[@LINE+5]]:1 [InImport]
// CHECK:STDERR: extern_library_copy.carbon:4:1-45: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_extern_library_collision.carbon:[[@LINE+3]]:1 [InImport]
// CHECK:STDERR: extern_library.carbon:4:1-45: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
import library "extern_library";
import library "extern_library_copy";

// --- extern_library_mismatch.carbon

library "[[@TEST_NAME]]";

extern library "extern_library_owner" fn F();

// --- fail_extern_library_mismatch_owner.carbon

library "[[@TEST_NAME]]";

import library "extern_library_mismatch"

// CHECK:STDERR: fail_extern_library_mismatch_owner.carbon:[[@LINE+4]]:1-6: error: `import` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
// CHECK:STDERR: fail_extern_library_mismatch_owner.carbon:[[@LINE-4]]:1-40: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR:
extern fn F();

// --- fail_extern_self_library.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_extern_self_library.carbon:[[@LINE+2]]:1-36: error: `extern library` cannot specify the current library [ExternLibraryIsCurrentLibrary]
// CHECK:STDERR:
extern library "extern_self_library" fn F();

// --- extern_of_import.carbon

library "[[@TEST_NAME]]";

fn F();

// --- fail_extern_of_import_redecl.carbon

library "[[@TEST_NAME]]";

import library "extern_of_import";

// CHECK:STDERR: fail_extern_of_import_redecl.carbon:[[@LINE+4]]:1-41: error: redeclarations of `fn F` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: imported from: fail_extern_of_import_redecl.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: extern_of_import.carbon:4:1-7: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
extern library "extern_of_import" fn F();

// --- fail_extern_library_on_definition.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_extern_library_on_definition.carbon:[[@LINE+2]]:1-37: error: a library cannot be provided for an `extern` modifier on a definition [ExternLibraryOnDefinition]
// CHECK:STDERR:
extern library "extern_library_owner" fn F() {}

// CHECK:STDOUT: --- extern_library.carbon
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
// CHECK:STDOUT: --- extern_library_owner.carbon
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
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_library_nonowner.carbon
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
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_library_nonextern.carbon
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
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_library_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = invalid
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F;
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_library_copy.carbon
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
// CHECK:STDOUT: --- fail_extern_library_collision.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F = import_ref Main//extern_library, F, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_library_mismatch.carbon
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
// CHECK:STDOUT: --- fail_extern_library_mismatch_owner.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_self_library.carbon
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
// CHECK:STDOUT: --- extern_of_import.carbon
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
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_of_import_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = invalid
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "extern_of_import.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_library_on_definition.carbon
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
// CHECK:STDOUT: extern fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

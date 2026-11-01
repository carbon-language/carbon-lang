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

// CHECK:STDERR:        ╭─┤ fail_extern_library_nonowner.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ extern fn F();
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: declaration in library "extern_library_nonowner" doesn't match
// CHECK:STDERR:        ·    ┃        `extern library` declaration
// CHECK:STDERR:        ·    ┃        [ExternLibraryIncorrect]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ extern_library.carbon
// CHECK:STDERR:        │ extern library "extern_library_owner" fn F();
// CHECK:STDERR:        · ──────────────────────┬──────────────────────
// CHECK:STDERR:        ·                       ╰──┤ previously declared with `extern library` here
// CHECK:STDERR:        ·                          │ [ExternLibraryExpected]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern fn F();

// --- fail_extern_library_nonextern.carbon

library "[[@TEST_NAME]]";

import library "extern_library";

// CHECK:STDERR:        ╭─┤ fail_extern_library_nonextern.carbon:[[@LINE+12]]:1
// CHECK:STDERR: ->     │ fn F();
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: redeclarations of `fn F` must match use of `extern`
// CHECK:STDERR:        ·       ┃        [RedeclExternMismatch]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ extern_library.carbon
// CHECK:STDERR:        │ extern library "extern_library_owner" fn F();
// CHECK:STDERR:        · ──────────────────────┬──────────────────────
// CHECK:STDERR:        ·                       ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F();

// --- fail_extern_library_redecl.carbon

library "[[@TEST_NAME]]";

import library "extern_library";

// CHECK:STDERR:        ╭─┤ fail_extern_library_redecl.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ extern library "extern_library_owner" fn F();
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: declaration in library "extern_library_redecl" doesn't match
// CHECK:STDERR:        ·    ┃        `extern library` declaration
// CHECK:STDERR:        ·    ┃        [ExternLibraryIncorrect]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ extern_library.carbon
// CHECK:STDERR:        │ extern library "extern_library_owner" fn F();
// CHECK:STDERR:        · ──────────────────────┬──────────────────────
// CHECK:STDERR:        ·                       ╰──┤ previously declared with `extern library` here
// CHECK:STDERR:        ·                          │ [ExternLibraryExpected]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern library "extern_library_owner" fn F();

// --- extern_library_copy.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╷ ╭── imported from: fail_extern_library_collision.carbon [InImport]
// CHECK:STDERR:        ├─┤ extern_library_copy.carbon:[[@LINE+12]]:1
// CHECK:STDERR: ->     │ extern library "extern_library_owner" fn F();
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                  ┗━━┫ error: duplicate name `F` being declared in the same scope
// CHECK:STDERR:        ·                     ┃        [NameDeclDuplicate]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ extern_library.carbon
// CHECK:STDERR:        │ extern library "extern_library_owner" fn F();
// CHECK:STDERR:        · ──────────────────────┬──────────────────────
// CHECK:STDERR:        ·                       ╰──┤ name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern library "extern_library_owner" fn F();

// --- fail_extern_library_collision.carbon

library "[[@TEST_NAME]]";

import library "extern_library";
import library "extern_library_copy";

// --- extern_library_mismatch.carbon

library "[[@TEST_NAME]]";

extern library "extern_library_owner" fn F();

// --- fail_extern_library_mismatch_owner.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_extern_library_mismatch_owner.carbon:[[@LINE+15]]:41
// CHECK:STDERR: ->     │ import library "extern_library_mismatch"
// CHECK:STDERR:        ·                                         ┳
// CHECK:STDERR:        ·                                         ┗━━┫ error: `import` declarations must
// CHECK:STDERR:        ·                                            ┃        end with a `;`
// CHECK:STDERR:        ·                                            ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_extern_library_mismatch_owner.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ import library "extern_library_mismatch"
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·                 ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import library "extern_library_mismatch"

extern fn F();

// --- fail_extern_self_library.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_extern_self_library.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ extern library "extern_self_library" fn F();
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                  ┗━━┫ error: `extern library` cannot specify the current library
// CHECK:STDERR:        ·                     ┃        [ExternLibraryIsCurrentLibrary]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern library "extern_self_library" fn F();

// --- extern_of_import.carbon

library "[[@TEST_NAME]]";

fn F();

// --- fail_extern_of_import_redecl.carbon

library "[[@TEST_NAME]]";

import library "extern_of_import";

// CHECK:STDERR:        ╭─┤ fail_extern_of_import_redecl.carbon:[[@LINE+12]]:1
// CHECK:STDERR: ->     │ extern library "extern_of_import" fn F();
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                  ┗━━┫ error: redeclarations of `fn F` must match use of `extern`
// CHECK:STDERR:        ·                     ┃        [RedeclExternMismatch]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ extern_of_import.carbon
// CHECK:STDERR:        │ fn F();
// CHECK:STDERR:        · ───┬───
// CHECK:STDERR:        ·    ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern library "extern_of_import" fn F();

// --- fail_extern_library_on_definition.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_extern_library_on_definition.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ extern library "extern_library_owner" fn F() {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: a library cannot be provided for an `extern` modifier on a
// CHECK:STDERR:        ·    ┃        definition
// CHECK:STDERR:        ·    ┃        [ExternLibraryOnDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern library "extern_library_owner" fn F() {}

// CHECK:STDOUT: --- extern_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_modifiers.carbon

// --- fail_extend_package.carbon

// CHECK:STDERR:        ╭─┤ fail_extend_package.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ extend package ExtendPackage;
// CHECK:STDERR:        · ━━━┳━━
// CHECK:STDERR:        ·    ┗━━┫ error: `extend` not allowed on `package` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extend package ExtendPackage;

// --- fail_export_package.carbon

// CHECK:STDERR:        ╭─┤ fail_export_package.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ export package ExportPackage;
// CHECK:STDERR:        · ━━━┳━━
// CHECK:STDERR:        ·    ┗━━┫ error: `export` not allowed on `package` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
export package ExportPackage;

// --- fail_final_package.carbon

// CHECK:STDERR:        ╭─┤ fail_final_package.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ final package ExtendPackage;
// CHECK:STDERR:        · ━━━┳━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·    ┗━━┫ error: library's API previously provided by `fail_extend_package.carbon`
// CHECK:STDERR:        ·       ┃        [DuplicateLibraryApi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_final_package.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ final package ExtendPackage;
// CHECK:STDERR:        · ━━┳━━
// CHECK:STDERR:        ·   ┗━━┫ error: `final` not allowed on `package` declaration
// CHECK:STDERR:        ·      ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
final package ExtendPackage;

// --- fail_virtual_package.carbon

// CHECK:STDERR:        ╭─┤ fail_virtual_package.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ virtual package VirtualPackage;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `virtual` not allowed on `package` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
virtual package VirtualPackage;

// --- fail_private_package.carbon

// CHECK:STDERR:        ╭─┤ fail_private_package.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ private package PrivatePackage;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `private` not allowed on `package` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
private package PrivatePackage;

// --- fail_export_library.carbon

// CHECK:STDERR:        ╭─┤ fail_export_library.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ export library "export_library";
// CHECK:STDERR:        · ━━━┳━━
// CHECK:STDERR:        ·    ┗━━┫ error: `export` not allowed on `library` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
export library "export_library";

// --- fail_import_modifiers.carbon

// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+13]]:1
// CHECK:STDERR: ->     │ impl import ImplImport;
// CHECK:STDERR:        · ━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: imported API 'ImplImport' not found [ImportNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl import ImplImport;
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `impl` not allowed on `import` declaration
// CHECK:STDERR:        ·      ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl import ImplImport;

// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+13]]:1
// CHECK:STDERR: ->     │ extend import ExtendImport;
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: imported API 'ExtendImport' not found [ImportNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ extend import ExtendImport;
// CHECK:STDERR:        · ━━━┳━━
// CHECK:STDERR:        ·    ┗━━┫ error: `extend` not allowed on `import` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extend import ExtendImport;

// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ virtual import VirtualImport;
// CHECK:STDERR:        · ━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━
// CHECK:STDERR:        ·               ┗━━┫ error: imported API 'VirtualImport' not found
// CHECK:STDERR:        ·                  ┃        [ImportNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ virtual import VirtualImport;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `virtual` not allowed on `import` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
virtual import VirtualImport;

// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+13]]:1
// CHECK:STDERR: ->     │ base import BaseImport;
// CHECK:STDERR:        · ━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: imported API 'BaseImport' not found [ImportNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ base import BaseImport;
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `base` not allowed on `import` declaration
// CHECK:STDERR:        ·      ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
base import BaseImport;

// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ private import PrivateImport;
// CHECK:STDERR:        · ━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━
// CHECK:STDERR:        ·               ┗━━┫ error: imported API 'PrivateImport' not found
// CHECK:STDERR:        ·                  ┃        [ImportNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_import_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ private import PrivateImport;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `private` not allowed on `import` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
private import PrivateImport;

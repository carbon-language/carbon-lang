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

// CHECK:STDERR: fail_extend_package.carbon:[[@LINE+2]]:1-6: error: `extend` not allowed on `package` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
extend package ExtendPackage;

// --- fail_export_package.carbon

// CHECK:STDERR: fail_export_package.carbon:[[@LINE+2]]:1-6: error: `export` not allowed on `package` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
export package ExportPackage;

// --- fail_final_package.carbon

// CHECK:STDERR: fail_final_package.carbon:[[@LINE+4]]:1-28: error: library's API previously provided by `fail_extend_package.carbon` [DuplicateLibraryApi]
// CHECK:STDERR:
// CHECK:STDERR: fail_final_package.carbon:[[@LINE+2]]:1-5: error: `final` not allowed on `package` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
final package ExtendPackage;

// --- fail_virtual_package.carbon

// CHECK:STDERR: fail_virtual_package.carbon:[[@LINE+2]]:1-7: error: `virtual` not allowed on `package` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
virtual package VirtualPackage;

// --- fail_private_package.carbon

// CHECK:STDERR: fail_private_package.carbon:[[@LINE+2]]:1-7: error: `private` not allowed on `package` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
private package PrivatePackage;

// --- fail_export_library.carbon

// CHECK:STDERR: fail_export_library.carbon:[[@LINE+2]]:1-6: error: `export` not allowed on `library` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
export library "export_library";

// --- fail_import_modifiers.carbon

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:1-23: error: imported API 'ImplImport' not found [ImportNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+2]]:1-4: error: `impl` not allowed on `import` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
impl import ImplImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:1-27: error: imported API 'ExtendImport' not found [ImportNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+2]]:1-6: error: `extend` not allowed on `import` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
extend import ExtendImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:1-29: error: imported API 'VirtualImport' not found [ImportNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+2]]:1-7: error: `virtual` not allowed on `import` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
virtual import VirtualImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:1-23: error: imported API 'BaseImport' not found [ImportNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+2]]:1-4: error: `base` not allowed on `import` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
base import BaseImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:1-29: error: imported API 'PrivateImport' not found [ImportNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+2]]:1-7: error: `private` not allowed on `import` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
private import PrivateImport;

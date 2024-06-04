// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/fail_modifiers.carbon

// --- fail_export_package.carbon

// CHECK:STDERR: fail_export_package.carbon:[[@LINE+4]]:1: ERROR: `export` not allowed on `package` declaration.
// CHECK:STDERR: export package ExportPackage;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export package ExportPackage;

// --- fail_extend_package.carbon

// CHECK:STDERR: fail_extend_package.carbon:[[@LINE+4]]:1: ERROR: `extend` not allowed on `package` declaration.
// CHECK:STDERR: extend package ExtendPackage;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
extend package ExtendPackage;

// --- fail_virtual_package.carbon

// CHECK:STDERR: fail_virtual_package.carbon:[[@LINE+4]]:1: ERROR: `virtual` not allowed on `package` declaration.
// CHECK:STDERR: virtual package VirtualPackage;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual package VirtualPackage;

// --- fail_private_package.carbon

// CHECK:STDERR: fail_private_package.carbon:[[@LINE+4]]:1: ERROR: `private` not allowed on `package` declaration.
// CHECK:STDERR: private package PrivatePackage;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private package PrivatePackage;

// --- fail_export_library.carbon

// CHECK:STDERR: fail_export_library.carbon:[[@LINE+4]]:1: ERROR: `export` not allowed on `library` declaration.
// CHECK:STDERR: export library "export_library";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export library "export_library";

// --- fail_import_modifiers.carbon

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+8]]:1: ERROR: `impl` not allowed on `import` declaration.
// CHECK:STDERR: impl import ImplImport;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:6: ERROR: Imported API 'ImplImport' not found.
// CHECK:STDERR: impl import ImplImport;
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR:
impl import ImplImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+8]]:1: ERROR: `extend` not allowed on `import` declaration.
// CHECK:STDERR: extend import ExtendImport;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:8: ERROR: Imported API 'ExtendImport' not found.
// CHECK:STDERR: extend import ExtendImport;
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR:
extend import ExtendImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+8]]:1: ERROR: `virtual` not allowed on `import` declaration.
// CHECK:STDERR: virtual import VirtualImport;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:9: ERROR: Imported API 'VirtualImport' not found.
// CHECK:STDERR: virtual import VirtualImport;
// CHECK:STDERR:         ^~~~~~
// CHECK:STDERR:
virtual import VirtualImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+8]]:1: ERROR: `base` not allowed on `import` declaration.
// CHECK:STDERR: base import BaseImport;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+4]]:6: ERROR: Imported API 'BaseImport' not found.
// CHECK:STDERR: base import BaseImport;
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR:
base import BaseImport;

// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+7]]:1: ERROR: `private` not allowed on `import` declaration.
// CHECK:STDERR: private import PrivateImport;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_modifiers.carbon:[[@LINE+3]]:9: ERROR: Imported API 'PrivateImport' not found.
// CHECK:STDERR: private import PrivateImport;
// CHECK:STDERR:         ^~~~~~
private import PrivateImport;

// CHECK:STDOUT: --- fail_export_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_virtual_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_private_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ImplImport = %ImplImport
// CHECK:STDOUT:     .ExtendImport = %ExtendImport
// CHECK:STDOUT:     .VirtualImport = %VirtualImport
// CHECK:STDOUT:     .BaseImport = %BaseImport
// CHECK:STDOUT:     .PrivateImport = %PrivateImport
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplImport: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExtendImport: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %VirtualImport: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BaseImport: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PrivateImport: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:

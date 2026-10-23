// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/export.carbon

// --- fail_package.carbon

package Pkg;

// CHECK:STDERR: fail_package.carbon:[[@LINE+2]]:15-17: error: `export` cannot be used when importing a package [ExportImportPackage]
// CHECK:STDERR:
export import Bar;

// --- default.carbon

package Pkg;

export import library default;

// --- named.carbon

package Pkg;

export import library "lib";

// --- fail_order.carbon

package Pkg;

export import library "a";

class C;

// CHECK:STDERR: fail_order.carbon:[[@LINE+3]]:1-6: error: `import` declarations must come after the `package` declaration (if present) and before any other entities in the file [ImportTooLate]
// CHECK:STDERR: fail_order.carbon:[[@LINE-3]]:1-5: note: first declaration is here [FirstDecl]
// CHECK:STDERR:
export import library "b";

// --- fail_in_default_library.carbon

// CHECK:STDERR: fail_in_default_library.carbon:[[@LINE+2]]:1-6: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR:
export import library "lib";

// --- fail_in_impl.carbon

impl package Pkg;

// CHECK:STDERR: fail_in_impl.carbon:[[@LINE+2]]:1-6: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR:
export import library "lib";

// CHECK:STDOUT: - filename: fail_package.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Bar'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: default.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   │ │ ╭─DefaultLibrary 'default'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: named.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"lib"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_order.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"a"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';'
// CHECK:STDOUT:   │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   ├─ClassDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_in_default_library.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"lib"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_in_impl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ExportModifier 'export'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"lib"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

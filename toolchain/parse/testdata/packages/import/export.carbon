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

// CHECK:STDERR: fail_package.carbon:[[@LINE+4]]:15: error: `export` cannot be used when importing a package [ExportImportPackage]
// CHECK:STDERR: export import Bar;
// CHECK:STDERR:               ^~~
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

// CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:1: error: `import` declarations must come after the `package` declaration (if present) and before any other entities in the file [ImportTooLate]
// CHECK:STDERR: export import library "b";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_order.carbon:[[@LINE-5]]:1: note: first declaration is here [FirstDecl]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
export import library "b";

// --- fail_in_default_library.carbon

// CHECK:STDERR: fail_in_default_library.carbon:[[@LINE+4]]:1: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR: export import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export import library "lib";

// --- fail_in_impl.carbon

impl package Pkg;

// CHECK:STDERR: fail_in_impl.carbon:[[@LINE+4]]:1: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR: export import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export import library "lib";

// CHECK:STDOUT: - filename: fail_package.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: default.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:         {kind: 'DefaultLibrary', text: 'default'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: named.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_order.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"a"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_default_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ExportModifier', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

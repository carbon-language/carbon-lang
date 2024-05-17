// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_package.carbon

package Pkg api;

// CHECK:STDERR: fail_package.carbon:[[@LINE+4]]:15: ERROR: `export` cannot be used when importing a package.
// CHECK:STDERR: export import Bar;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
export import Bar;

// --- default.carbon

package Pkg api;

export import library default;

// --- named.carbon

package Pkg api;

export import library "lib";

// --- fail_order.carbon

package Pkg api;

export import library "a";

class C;

// CHECK:STDERR: fail_order.carbon:[[@LINE+7]]:8: ERROR: `import` directives must come after the `package` directive (if present) and before any other entities in the file.
// CHECK:STDERR: export import library "b";
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_order.carbon:[[@LINE-5]]:1: First declaration is here.
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
export import library "b";

// --- fail_in_default_library.carbon

// CHECK:STDERR: fail_in_default_library.carbon:[[@LINE+4]]:1: ERROR: `export` is only allowed in API files.
// CHECK:STDERR: export import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export import library "lib";

// --- fail_in_impl.carbon

package Pkg impl;

// CHECK:STDERR: fail_in_impl.carbon:[[@LINE+3]]:1: ERROR: `export` is only allowed in API files.
// CHECK:STDERR: export import library "lib";
// CHECK:STDERR: ^~~~~~
export import library "lib";

// CHECK:STDOUT: - filename: fail_package.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: default.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:         {kind: 'DefaultLibrary', text: 'default'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: named.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_order.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"a"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_default_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'ImportExport', text: 'export'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

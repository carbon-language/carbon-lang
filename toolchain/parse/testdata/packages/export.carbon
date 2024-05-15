// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- lib.carbon

library "lib" api;

// --- name.carbon

package Pkg api;

export Foo;

// --- qual_name.carbon

package Pkg api;

export Foo.Bar;

// --- fail_expr.carbon

package Pkg api;

// CHECK:STDERR: fail_expr.carbon:[[@LINE+4]]:8: ERROR: `export` introducer should be followed by a name.
// CHECK:STDERR: export ();
// CHECK:STDERR:        ^
// CHECK:STDERR:
export ();

// --- fail_package_in_name.carbon

package Pkg api;

// CHECK:STDERR: fail_package_in_name.carbon:[[@LINE+4]]:8: ERROR: `export` introducer should be followed by a name.
// CHECK:STDERR: export package.Bar;
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
export package.Bar;

// --- fail_keyword_only

package Pkg api;

// CHECK:STDERR: fail_keyword_only:[[@LINE+4]]:1: ERROR: `export` introducer should be followed by a name.
// CHECK:STDERR: export
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export

// --- fail_no_name.carbon

package Pkg api;

// CHECK:STDERR: fail_no_name.carbon:[[@LINE+4]]:8: ERROR: `export` introducer should be followed by a name.
// CHECK:STDERR: export ;
// CHECK:STDERR:        ^
// CHECK:STDERR:
export ;

// --- fail_no_semi.carbon

package Pkg api;

export Foo
// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+4]]:1: ERROR: `export` declarations must end with a `;`.
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_incomplete_qual_name.carbon

package Pkg api;

// CHECK:STDERR: fail_incomplete_qual_name.carbon:[[@LINE+4]]:12: ERROR: Expected identifier after `.`.
// CHECK:STDERR: export Foo.;
// CHECK:STDERR:            ^
// CHECK:STDERR:
export Foo.;

// --- fail_incomplete_qual_name2.carbon

package Pkg api;

// CHECK:STDERR: fail_incomplete_qual_name2.carbon:[[@LINE+4]]:8: ERROR: `export` introducer should be followed by a name.
// CHECK:STDERR: export .Bar;
// CHECK:STDERR:        ^
// CHECK:STDERR:
export .Bar;

// --- after_decl.carbon

package Pkg api;

class C;

export Foo;

// --- fail_before_import.carbon

package Pkg api;

export Foo;

// CHECK:STDERR: fail_before_import.carbon:[[@LINE+7]]:1: ERROR: `import` directives must come after the `package` directive (if present) and before any other entities in the file.
// CHECK:STDERR: import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_before_import.carbon:[[@LINE-5]]:1: First declaration is here.
// CHECK:STDERR: export Foo;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import library "lib";

// --- fail_in_default_library.carbon

// CHECK:STDERR: fail_in_default_library.carbon:[[@LINE+4]]:1: ERROR: `export` is only allowed in API files.
// CHECK:STDERR: export Foo;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export Foo;

// --- fail_in_impl.carbon

package Pkg impl;

// CHECK:STDERR: fail_in_impl.carbon:[[@LINE+3]]:1: ERROR: `export` is only allowed in API files.
// CHECK:STDERR: export Foo;
// CHECK:STDERR: ^~~~~~
export Foo;

// CHECK:STDOUT: - filename: lib.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: qual_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:       {kind: 'QualifiedName', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_package_in_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_keyword_only
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: 'export', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: 'Foo', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_incomplete_qual_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'QualifiedName', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_incomplete_qual_name2.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: after_decl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_before_import.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_default_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Pkg'},
// CHECK:STDOUT:       {kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDirective', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

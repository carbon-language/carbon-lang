// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/export.carbon

// --- lib.carbon

library "[[@TEST_NAME]]";

// --- name.carbon

package Pkg;

export Foo;

// --- qual_name.carbon

package Pkg;

export Foo.Bar;

// --- fail_expr.carbon

package Pkg;

// CHECK:STDERR: fail_expr.carbon:[[@LINE+4]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: export ();
// CHECK:STDERR:        ^
// CHECK:STDERR:
export ();

// --- fail_package_in_name.carbon

package Pkg;

// CHECK:STDERR: fail_package_in_name.carbon:[[@LINE+4]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: export package.Bar;
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
export package.Bar;

// --- fail_keyword_only

package Pkg;

// CHECK:STDERR: fail_keyword_only:[[@LINE+4]]:1: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: export
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export

// --- fail_no_name.carbon

package Pkg;

// CHECK:STDERR: fail_no_name.carbon:[[@LINE+4]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: export ;
// CHECK:STDERR:        ^
// CHECK:STDERR:
export ;

// --- fail_no_semi.carbon

package Pkg;

export Foo
// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+4]]:1: error: `export` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_incomplete_qual_name.carbon

package Pkg;

// CHECK:STDERR: fail_incomplete_qual_name.carbon:[[@LINE+4]]:12: error: `.` should be followed by a name [ExpectedDeclNameAfterPeriod]
// CHECK:STDERR: export Foo.;
// CHECK:STDERR:            ^
// CHECK:STDERR:
export Foo.;

// --- fail_incomplete_qual_name2.carbon

package Pkg;

// CHECK:STDERR: fail_incomplete_qual_name2.carbon:[[@LINE+4]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: export .Bar;
// CHECK:STDERR:        ^
// CHECK:STDERR:
export .Bar;

// --- after_decl.carbon

package Pkg;

class C;

export Foo;

// --- fail_before_import.carbon

package Pkg;

export Foo;

// CHECK:STDERR: fail_before_import.carbon:[[@LINE+7]]:1: error: `import` declarations must come after the `package` declaration (if present) and before any other entities in the file [ImportTooLate]
// CHECK:STDERR: import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_before_import.carbon:[[@LINE-5]]:1: note: first declaration is here [FirstDecl]
// CHECK:STDERR: export Foo;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import library "lib";

// --- fail_in_default_library.carbon

// CHECK:STDERR: fail_in_default_library.carbon:[[@LINE+4]]:1: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR: export Foo;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export Foo;

// --- fail_in_impl.carbon

impl package Pkg;

// CHECK:STDERR: fail_in_impl.carbon:[[@LINE+4]]:1: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR: export Foo;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export Foo;

// CHECK:STDOUT: - filename: lib.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"lib"'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: qual_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_package_in_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_keyword_only
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: 'export', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: 'Foo', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_incomplete_qual_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_incomplete_qual_name2.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: after_decl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_before_import.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:     {kind: 'ImportDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_default_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_in_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierPackageName', text: 'Pkg'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExportIntroducer', text: 'export'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ExportDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

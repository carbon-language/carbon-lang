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

// CHECK:STDERR: fail_expr.carbon:[[@LINE+2]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
export ();

// --- fail_package_in_name.carbon

package Pkg;

// CHECK:STDERR: fail_package_in_name.carbon:[[@LINE+2]]:8-14: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
export package.Bar;

// --- fail_keyword_only

package Pkg;

// CHECK:STDERR: fail_keyword_only:[[@LINE+2]]:1-6: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
export

// --- fail_no_name.carbon

package Pkg;

// CHECK:STDERR: fail_no_name.carbon:[[@LINE+2]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
export ;

// --- fail_no_semi.carbon

package Pkg;

export Foo
// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+2]]:1: error: `export` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:

// --- fail_incomplete_qual_name.carbon

package Pkg;

// CHECK:STDERR: fail_incomplete_qual_name.carbon:[[@LINE+2]]:12: error: `.` should be followed by a name [ExpectedDeclNameAfterPeriod]
// CHECK:STDERR:
export Foo.;

// --- fail_incomplete_qual_name2.carbon

package Pkg;

// CHECK:STDERR: fail_incomplete_qual_name2.carbon:[[@LINE+2]]:8: error: `export` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
export .Bar;

// --- after_decl.carbon

package Pkg;

class C;

export Foo;

// --- fail_before_import.carbon

package Pkg;

export Foo;

// CHECK:STDERR: fail_before_import.carbon:[[@LINE+3]]:1-6: error: `import` declarations must come after the `package` declaration (if present) and before any other entities in the file [ImportTooLate]
// CHECK:STDERR: fail_before_import.carbon:[[@LINE-3]]:1-6: note: first declaration is here [FirstDecl]
// CHECK:STDERR:
import library "lib";

// --- fail_in_default_library.carbon

// CHECK:STDERR: fail_in_default_library.carbon:[[@LINE+2]]:1-6: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR:
export Foo;

// --- fail_in_impl.carbon

impl package Pkg;

// CHECK:STDERR: fail_in_impl.carbon:[[@LINE+2]]:1-6: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR:
export Foo;

// CHECK:STDOUT: - filename: lib.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"lib"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─ExportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: qual_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Bar'
// CHECK:STDOUT:   ├─ExportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_expr.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─InvalidParse '(' has_error
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_package_in_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─InvalidParse 'package' has_error
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_keyword_only
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─InvalidParse '' has_error
// CHECK:STDOUT:   ├─ExportDecl 'export' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─ExportDecl 'Foo' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_incomplete_qual_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_incomplete_qual_name2.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─InvalidParse '.' has_error
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: after_decl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   ├─ClassDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─ExportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_before_import.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─ExportDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_in_default_library.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_in_impl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Pkg'
// CHECK:STDOUT:   ├─PackageDecl ';'
// CHECK:STDOUT:   │ ╭─ExportIntroducer 'export'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─ExportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

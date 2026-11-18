// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/cpp_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/cpp_inline.carbon

// --- fail_bad_syntax.carbon

// CHECK:STDERR:        ╭─┤ fail_bad_syntax.carbon:[[@LINE+7]]:18
// CHECK:STDERR: ->     │ import Cpp inline;
// CHECK:STDERR:        ·                  ┳
// CHECK:STDERR:        ·                  ┗━━┫ error: expected string literal after `inline`
// CHECK:STDERR:        ·                     ┃        [ExpectedStringAfterInline]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp inline;
// CHECK:STDERR:        ╭─┤ fail_bad_syntax.carbon:[[@LINE+7]]:18
// CHECK:STDERR: ->     │ import Cpp inline library "foo.h";
// CHECK:STDERR:        ·                  ┳
// CHECK:STDERR:        ·                  ┗━━┫ error: expected string literal after `inline`
// CHECK:STDERR:        ·                     ┃        [ExpectedStringAfterInline]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp inline library "foo.h";

// TODO: We might want to allow this as a "preprocessed source" syntax.
// CHECK:STDERR:        ╭─┤ fail_bad_syntax.carbon:[[@LINE+7]]:27
// CHECK:STDERR: ->     │ import Cpp library "foo.h" inline "bar";
// CHECK:STDERR:        ·                           ┳
// CHECK:STDERR:        ·                           ┗━━┫ error: `import` declarations must end with a `;`
// CHECK:STDERR:        ·                              ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp library "foo.h" inline "bar";

// CHECK:STDERR:        ╭─┤ fail_bad_syntax.carbon:[[@LINE+7]]:23
// CHECK:STDERR: ->     │ import library "foo.h" inline "bar";
// CHECK:STDERR:        ·                       ┳
// CHECK:STDERR:        ·                       ┗━━┫ error: `import` declarations must end with a `;`
// CHECK:STDERR:        ·                          ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import library "foo.h" inline "bar";

// --- fail_no_semi_at_eof.carbon

// CHECK:STDERR:        ╭─┤ fail_no_semi_at_eof.carbon:[[@LINE+7]]:18
// CHECK:STDERR: ->     │ import Cpp inline
// CHECK:STDERR:        ·                  ┳
// CHECK:STDERR:        ·                  ┗━━┫ error: expected string literal after `inline`
// CHECK:STDERR:        ·                     ┃        [ExpectedStringAfterInline]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp inline

// --- fail_package_inline.carbon

// CHECK:STDERR:        ╭─┤ fail_package_inline.carbon:[[@LINE+7]]:15
// CHECK:STDERR: ->     │ package NotCpp inline "int n;";
// CHECK:STDERR:        ·               ┳
// CHECK:STDERR:        ·               ┗━━┫ error: `package` declarations must end with a `;`
// CHECK:STDERR:        ·                  ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
package NotCpp inline "int n;";

// --- fail_library_inline.carbon

// CHECK:STDERR:        ╭─┤ fail_library_inline.carbon:[[@LINE+7]]:16
// CHECK:STDERR: ->     │ library "foo.h" inline "int n;";
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: `library` declarations must end with a `;`
// CHECK:STDERR:        ·                   ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library "foo.h" inline "int n;";

// --- wrong_package.carbon

// We reject this in check.
import NotCpp inline "bar";

// --- valid_syntax.carbon

import Cpp inline "int m;";

import Cpp inline '''
// C++ comment.
int n;
''';

// --- inline_cpp.carbon

inline Cpp '''
int n;
''';

// --- fail_inline_cpp_syntax_errors.carbon

// CHECK:STDERR:        ╭─┤ fail_inline_cpp_syntax_errors.carbon:[[@LINE+6]]:7
// CHECK:STDERR: ->     │ inline;
// CHECK:STDERR:        ·       ┳
// CHECK:STDERR:        ·       ┗━━┫ error: expected `Cpp` after `inline` [ExpectedCppAfterInline]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
inline;

// CHECK:STDERR:        ╭─┤ fail_inline_cpp_syntax_errors.carbon:[[@LINE+6]]:7
// CHECK:STDERR: ->     │ inline '''
// CHECK:STDERR:        ·       ┳
// CHECK:STDERR:        ·       ┗━━┫ error: expected `Cpp` after `inline` [ExpectedCppAfterInline]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
inline '''
int n;
''';

// Recovery skips to the `;` even when tokens follow the closing `'''` on its
// line, so the `;` ends the declaration rather than starting an empty one.
// CHECK:STDERR:        ╭─┤ fail_inline_cpp_syntax_errors.carbon:[[@LINE+6]]:7
// CHECK:STDERR: ->     │ inline '''
// CHECK:STDERR:        ·       ┳
// CHECK:STDERR:        ·       ┗━━┫ error: expected `Cpp` after `inline` [ExpectedCppAfterInline]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
inline '''
int n;
''' + "more";

// CHECK:STDERR:        ╭─┤ fail_inline_cpp_syntax_errors.carbon:[[@LINE+7]]:11
// CHECK:STDERR: ->     │ inline Cpp;
// CHECK:STDERR:        ·           ┳
// CHECK:STDERR:        ·           ┗━━┫ error: expected string literal after `inline Cpp`
// CHECK:STDERR:        ·              ┃        [ExpectedStringAfterInlineCpp]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
inline Cpp;

// CHECK:STDERR:        ╭─┤ fail_inline_cpp_syntax_errors.carbon:[[@LINE+7]]:11
// CHECK:STDERR: ->     │ inline Cpp x;
// CHECK:STDERR:        ·           ┳
// CHECK:STDERR:        ·           ┗━━┫ error: expected string literal after `inline Cpp`
// CHECK:STDERR:        ·              ┃        [ExpectedStringAfterInlineCpp]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
inline Cpp x;


// CHECK:STDOUT: - filename: fail_bad_syntax.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─CppPackageName 'Cpp'
// CHECK:STDOUT:   │ │ ╭─InlineImportBody ';' has_error
// CHECK:STDOUT:   │ ├─InlineImportSpecifier 'inline'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─CppPackageName 'Cpp'
// CHECK:STDOUT:   │ │ ╭─InlineImportBody 'library' has_error
// CHECK:STDOUT:   │ ├─InlineImportSpecifier 'inline'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─CppPackageName 'Cpp'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"foo.h"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"foo.h"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_semi_at_eof.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─CppPackageName 'Cpp'
// CHECK:STDOUT:   │ │ ╭─InlineImportBody '' has_error
// CHECK:STDOUT:   │ ├─InlineImportSpecifier 'inline'
// CHECK:STDOUT:   ├─ImportDecl 'inline' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_package_inline.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'NotCpp'
// CHECK:STDOUT:   ├─PackageDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_library_inline.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"foo.h"'
// CHECK:STDOUT:   ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: wrong_package.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'NotCpp'
// CHECK:STDOUT:   │ │ ╭─InlineImportBody '"bar"'
// CHECK:STDOUT:   │ ├─InlineImportSpecifier 'inline'
// CHECK:STDOUT:   ├─ImportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: valid_syntax.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─CppPackageName 'Cpp'
// CHECK:STDOUT:   │ │ ╭─InlineImportBody '"int m;"'
// CHECK:STDOUT:   │ ├─InlineImportSpecifier 'inline'
// CHECK:STDOUT:   ├─ImportDecl ';'
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─CppPackageName 'Cpp'
// CHECK:STDOUT:   │ │ ╭─InlineImportBody ''''
// CHECK:STDOUT: // C++ comment.
// CHECK:STDOUT: int n;
// CHECK:STDOUT: ''''
// CHECK:STDOUT:   │ ├─InlineImportSpecifier 'inline'
// CHECK:STDOUT:   ├─ImportDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: inline_cpp.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   │ ├─CppNameExpr 'Cpp'
// CHECK:STDOUT:   │ ├─InlineImportBody ''''
// CHECK:STDOUT: int n;
// CHECK:STDOUT: ''''
// CHECK:STDOUT:   ├─InlineCppDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_inline_cpp_syntax_errors.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   ├─InlineCppDecl ';' has_error
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   ├─InlineCppDecl ';' has_error
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   ├─InlineCppDecl ';' has_error
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   │ ├─CppNameExpr 'Cpp'
// CHECK:STDOUT:   ├─InlineCppDecl ';' has_error
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   │ ├─CppNameExpr 'Cpp'
// CHECK:STDOUT:   ├─InlineCppDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

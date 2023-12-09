// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A();

// CHECK:STDERR: fail_after_decl_repeated.carbon:[[@LINE+6]]:1: ERROR: `import` directives must come after the `package` directive (if present) and before any other entities in the file.
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_after_decl_repeated.carbon:[[@LINE-5]]:1: First declaration is here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~
import B;
// Note this is still invalid, but doesn't warn because it's sequential.
import C;

// CHECK:STDERR: fail_after_decl_repeated.carbon:[[@LINE+6]]:1: ERROR: The `package` directive must be the first non-comment line.
// CHECK:STDERR: package D;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_after_decl_repeated.carbon:[[@LINE-15]]:1: First non-comment line is here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~
package D;

fn E();

// CHECK:STDERR: fail_after_decl_repeated.carbon:[[@LINE+6]]:1: ERROR: `import` directives must come after the `package` directive (if present) and before any other entities in the file.
// CHECK:STDERR: import F;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fail_after_decl_repeated.carbon:[[@LINE-25]]:1: First declaration is here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~
import F;
// Note this is still invalid, but doesn't warn because it's sequential.
import G;

// CHECK:STDOUT: - filename: fail_after_decl_repeated.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'A'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'E'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

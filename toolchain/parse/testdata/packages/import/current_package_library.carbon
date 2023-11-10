// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- default.carbon

import library default;

// --- named.carbon

import library "Shapes";

// CHECK:STDOUT: - filename: default.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:         {kind: 'DefaultLibrary', text: 'default'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: named.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"Shapes"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

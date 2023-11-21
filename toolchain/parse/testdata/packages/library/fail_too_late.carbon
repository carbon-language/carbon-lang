// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

import Geometry;

// CHECK:STDERR: fail_too_late.carbon:[[@LINE+6]]:1: ERROR: The `library` directive must be the first non-comment line.
// CHECK:STDERR: library "Shapes" api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_too_late.carbon:[[@LINE-5]]:1: First non-comment line is here.
// CHECK:STDERR: import Geometry;
// CHECK:STDERR: ^~~~~~
library "Shapes" api;

// CHECK:STDOUT: - filename: fail_too_late.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Geometry'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

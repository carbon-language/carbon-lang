// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

import Geometry library "Shapes";

// CHECK:STDOUT: - filename: library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'Name', text: 'Geometry'},
// CHECK:STDOUT:         {kind: 'Literal', text: '"Shapes"'},
// CHECK:STDOUT:       {kind: 'Library', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

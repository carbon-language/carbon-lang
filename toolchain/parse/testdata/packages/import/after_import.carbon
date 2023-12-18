// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

import A;
import B;

// CHECK:STDOUT: - filename: after_import.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'A'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

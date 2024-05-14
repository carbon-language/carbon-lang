// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

import A;

// CHECK:STDERR: fail_after_import.carbon:[[@LINE+6]]:1: ERROR: The `package` directive must be the first non-comment line.
// CHECK:STDERR: package B api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_after_import.carbon:[[@LINE-5]]:1: First non-comment line is here.
// CHECK:STDERR: import A;
// CHECK:STDERR: ^~~~~~
package B api;

// CHECK:STDOUT: - filename: fail_after_import.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImportIntroducer', text: 'import'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'A'},
// CHECK:STDOUT:     {kind: 'ImportDirective', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

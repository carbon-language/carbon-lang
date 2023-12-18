// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Geometry library "Shapes" impl;

// CHECK:STDOUT: - filename: impl_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageName', text: 'Geometry'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"Shapes"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:     {kind: 'PackageDirective', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

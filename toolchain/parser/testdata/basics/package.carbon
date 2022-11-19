// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:   {kind: 'PackageEnd', text: ';'},
// CHECK:STDOUT: {kind: 'PackageDirective', text: 'package', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:   {kind: 'PackageEnd', text: ';'},
// CHECK:STDOUT: {kind: 'PackageDirective', text: 'package', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:     {kind: 'Literal', text: '"Shapes"'},
// CHECK:STDOUT:   {kind: 'PackageLibrary', text: 'library', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:   {kind: 'PackageEnd', text: ';'},
// CHECK:STDOUT: {kind: 'PackageDirective', text: 'package', subtree_size: 6},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:     {kind: 'Literal', text: '"Shapes"'},
// CHECK:STDOUT:   {kind: 'PackageLibrary', text: 'library', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:   {kind: 'PackageEnd', text: ';'},
// CHECK:STDOUT: {kind: 'PackageDirective', text: 'package', subtree_size: 6},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
package Geometry api;
package Geometry impl;
package Geometry library "Shapes" api;
package Geometry library "Shapes" impl;

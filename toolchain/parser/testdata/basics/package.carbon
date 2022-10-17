// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 3, kind: 'PackageDirective', text: 'package', subtree_size: 4, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {node_index: 1, kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:   {node_index: 2, kind: 'PackageEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 7, kind: 'PackageDirective', text: 'package', subtree_size: 4, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {node_index: 5, kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:   {node_index: 6, kind: 'PackageEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 13, kind: 'PackageDirective', text: 'package', subtree_size: 6, children: [
// CHECK:STDOUT:   {node_index: 8, kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {node_index: 10, kind: 'PackageLibrary', text: 'library', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 9, kind: 'Literal', text: '"Shapes"'}]},
// CHECK:STDOUT:   {node_index: 11, kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:   {node_index: 12, kind: 'PackageEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 19, kind: 'PackageDirective', text: 'package', subtree_size: 6, children: [
// CHECK:STDOUT:   {node_index: 14, kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {node_index: 16, kind: 'PackageLibrary', text: 'library', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 15, kind: 'Literal', text: '"Shapes"'}]},
// CHECK:STDOUT:   {node_index: 17, kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT:   {node_index: 18, kind: 'PackageEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 20, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
package Geometry api;
package Geometry impl;
package Geometry library "Shapes" api;
package Geometry library "Shapes" impl;

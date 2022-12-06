// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {kind: 'PackageImpl', text: 'impl'},
// CHECK:STDOUT: {kind: 'PackageDirective', text: ';', subtree_size: 4},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

package Geometry impl;

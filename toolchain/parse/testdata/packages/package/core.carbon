// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/core.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/core.carbon

// --- core.carbon

package Core;

// --- core_with_library.carbon

package Core library "prelude";

// CHECK:STDOUT: - filename: core.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'CorePackageName', text: 'Core'},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: core_with_library.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'CorePackageName', text: 'Core'},
// CHECK:STDOUT:         {kind: 'LibraryName', text: '"prelude"'},
// CHECK:STDOUT:       {kind: 'LibrarySpecifier', text: 'library', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'PackageDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

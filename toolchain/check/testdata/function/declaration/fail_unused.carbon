// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_unused.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_unused.carbon

// CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:13: error: `unused` maker cannot be used on a declaration that is not a definition [UnusedMarkerInDeclaration]
// CHECK:STDERR: fn F(unused x: i32);
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn F(unused x: i32);

// CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:21: error: `unused` maker cannot be used on a declaration that is not a definition [UnusedMarkerInDeclaration]
// CHECK:STDERR: fn G(x: i32, unused y: i32);
// CHECK:STDERR:                     ^
// CHECK:STDERR:
fn G(x: i32, unused y: i32);

// CHECK:STDERR: fail_unused.carbon:[[@LINE+12]]:13: error: semantics TODO: `Support patterns besides bindings in parameter list` [SemanticsTodo]
// CHECK:STDERR: fn H(unused (x: i32, y: i32));
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_unused.carbon:[[@LINE+8]]:14: error: `unused` maker cannot be used on a declaration that is not a definition [UnusedMarkerInDeclaration]
// CHECK:STDERR: fn H(unused (x: i32, y: i32));
// CHECK:STDERR:              ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:22: error: `unused` maker cannot be used on a declaration that is not a definition [UnusedMarkerInDeclaration]
// CHECK:STDERR: fn H(unused (x: i32, y: i32));
// CHECK:STDERR:                      ^
// CHECK:STDERR:
fn H(unused (x: i32, y: i32));

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/struct_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/struct_literal.carbon

// --- test.carbon

// A struct literal's field designators are tight to their names and spaced
// after commas.
//
// TODO: The braces are still laid out as a block (the parse tree's
// `StructLiteralStart` can distinguish them); value-brace layout is future
// work.
var origin: auto = {.x = 0,.y = 0};

var type_like: {.x: i32, .y: i32} = origin;

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A struct literal's field designators are tight to their names and spaced
// CHECK:STDOUT: // after commas.
// CHECK:STDOUT: //
// CHECK:STDOUT: // TODO: The braces are still laid out as a block (the parse tree's
// CHECK:STDOUT: // `StructLiteralStart` can distinguish them); value-brace layout is future
// CHECK:STDOUT: // work.
// CHECK:STDOUT: var origin: auto = {
// CHECK:STDOUT:   .x = 0, .y = 0
// CHECK:STDOUT: };
// CHECK:STDOUT:
// CHECK:STDOUT: var type_like: {
// CHECK:STDOUT:   .x: i32, .y: i32
// CHECK:STDOUT: } = origin;

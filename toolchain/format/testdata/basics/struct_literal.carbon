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

// A struct literal, struct type literal, or struct pattern without a trailing
// comma stays inline with no spaces inside `{...}`.
var origin: auto = { .x = 0,.y = 0 };

var type_like: { .x: i32, .y: i32 } = origin;

var ptr: { .x: i32 }* = &origin;

var empty_ptr: { }* = nullptr;

var {.x = ox: i32, .y = oy: i32} = origin;

fn Make() -> {.x: i32} {
  return {.x = {.a = 1}.a};
}

// A trailing comma expands a struct literal, struct type literal, or struct
// pattern to one field per line, while nested structs without trailing commas
// stay inline.
var multiline: { .a: i32, .b: {.x: i32, .y: i32}, } = {
  .a = 1, .b = { .x = 2, .y = 3 },
};

var { .a = ma: i32, .b = mb: auto, } = multiline;

var empty_commented: auto = {
  // Empty struct with a comment stays multiline.
};

// `choice` alternatives format one per line.
choice EmptyChoice {}

choice Color { Red, Green, Blue }

choice Option(T: type) { None, Some(x: T, y: T), }

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A struct literal, struct type literal, or struct pattern without a trailing
// CHECK:STDOUT: // comma stays inline with no spaces inside `{...}`.
// CHECK:STDOUT: var origin: auto = {.x = 0, .y = 0};
// CHECK:STDOUT:
// CHECK:STDOUT: var type_like: {.x: i32, .y: i32} = origin;
// CHECK:STDOUT:
// CHECK:STDOUT: var ptr: {.x: i32}* = &origin;
// CHECK:STDOUT:
// CHECK:STDOUT: var empty_ptr: {}* = nullptr;
// CHECK:STDOUT:
// CHECK:STDOUT: var {.x = ox: i32, .y = oy: i32} = origin;
// CHECK:STDOUT:
// CHECK:STDOUT: fn Make() -> {.x: i32} {
// CHECK:STDOUT:   return {.x = {.a = 1}.a};
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // A trailing comma expands a struct literal, struct type literal, or struct
// CHECK:STDOUT: // pattern to one field per line, while nested structs without trailing commas
// CHECK:STDOUT: // stay inline.
// CHECK:STDOUT: var multiline: {
// CHECK:STDOUT:   .a: i32,
// CHECK:STDOUT:   .b: {.x: i32, .y: i32},
// CHECK:STDOUT: } = {
// CHECK:STDOUT:   .a = 1,
// CHECK:STDOUT:   .b = {.x = 2, .y = 3},
// CHECK:STDOUT: };
// CHECK:STDOUT:
// CHECK:STDOUT: var {
// CHECK:STDOUT:   .a = ma: i32,
// CHECK:STDOUT:   .b = mb: auto,
// CHECK:STDOUT: } = multiline;
// CHECK:STDOUT:
// CHECK:STDOUT: var empty_commented: auto = {
// CHECK:STDOUT:   // Empty struct with a comment stays multiline.
// CHECK:STDOUT: };
// CHECK:STDOUT:
// CHECK:STDOUT: // `choice` alternatives format one per line.
// CHECK:STDOUT: choice EmptyChoice {}
// CHECK:STDOUT:
// CHECK:STDOUT: choice Color {
// CHECK:STDOUT:   Red,
// CHECK:STDOUT:   Green,
// CHECK:STDOUT:   Blue
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: choice Option(T: type) {
// CHECK:STDOUT:   None,
// CHECK:STDOUT:   Some(x: T, y: T),
// CHECK:STDOUT: }

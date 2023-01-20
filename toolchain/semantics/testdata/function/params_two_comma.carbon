// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/function/params_two_comma.carbon
// CHECK:STDERR: 1.	Parser stack:
// CHECK:STDERR: 	0.	DeclarationLoop @ 25:1: token 0 : Fn
// CHECK:STDERR: 	1.	FunctionAfterParameterList @ 25:1: token 0 : Fn
// CHECK:STDERR: 	cursor	position_ @ 25:22: token 10 : Comma
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3ad3)[0x55a6e27a4ad3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf181e)[0x55a6e27a281e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3eca)[0x55a6e27a4eca]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7f4ef2d6df90]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8accc)[0x7f4ef2dbcccc]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x12)[0x7f4ef2d6def2]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xd3)[0x7f4ef2d58472]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xc83e2)[0x55a6e27793e2]
// CHECK:STDERR: CHECK failure at toolchain/parser/parser.cpp:166: PositionIs(kind): Required CloseParen, found Comma

fn Foo(a: i32, b: i32,) {}

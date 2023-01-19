// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/var/lookup.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionIntroducer
// CHECK:STDERR: 	1.	DeclaredName
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: 	0.	block0
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf29f3)[0x55c6c3fcb9f3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf073e)[0x55c6c3fc973e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf2dea)[0x55c6c3fcbdea]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7efe10f0ef90]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8accc)[0x7efe10f5dccc]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x12)[0x7efe10f0eef2]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xd3)[0x7efe10ef9472]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xc7302)[0x55c6c3fa0302]
// CHECK:STDERR: CHECK failure at toolchain/semantics/semantics_node_stack.cpp:57: entry.node_id.is_valid(): Expected valid node_id on ParameterList

fn Main() {
  var x: i32 = 0;
  x;
}

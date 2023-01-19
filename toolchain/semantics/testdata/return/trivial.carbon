// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/return/trivial.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionIntroducer
// CHECK:STDERR: 	1.	DeclaredName
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: 	0.	block0
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf29f3)[0x5563c81299f3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf073e)[0x5563c812773e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf2dea)[0x5563c8129dea]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7f1cb8c42f90]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8accc)[0x7f1cb8c91ccc]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x12)[0x7f1cb8c42ef2]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xd3)[0x7f1cb8c2d472]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xc7302)[0x5563c80fe302]
// CHECK:STDERR: CHECK failure at toolchain/semantics/semantics_node_stack.cpp:57: entry.node_id.is_valid(): Expected valid node_id on ParameterList

fn Main() {
  return;
}

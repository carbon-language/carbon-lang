// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/basics/bad_name_lookup.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionDefinitionStart
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: 	0.	block0
// CHECK:STDERR: 	1.	block1
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xea6e3)[0x55c23d7126e3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xe842e)[0x55c23d71042e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xeaada)[0x55c23d712ada]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3daa0)[0x7fd848c3daa0]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8957c)[0x7fd848c8957c]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(raise+0x12)[0x7fd848c3da02]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xcf)[0x7fd848c28469]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xbeff2)[0x55c23d6e6ff2]
// CHECK:STDERR: CHECK failure at toolchain/semantics/semantics_parse_tree_handler.cpp:157: node_id.is_valid(): Invalid PopWithResult on NameReference

fn Main() {
  x;
}

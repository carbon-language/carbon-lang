// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/var/global_decl_with_init.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	VariableIntroducer
// CHECK:STDERR: 	1.	PatternBinding -> node0
// CHECK:STDERR: 	2.	Literal -> node2
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: 	0.	block0
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xe9803)[0x555df7747803]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xe754e)[0x555df774554e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xe9bfa)[0x555df7747bfa]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3daa0)[0x7f5f6163daa0]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8957c)[0x7f5f6168957c]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(raise+0x12)[0x7f5f6163da02]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xcf)[0x7f5f61628469]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xbe112)[0x555df771c112]
// CHECK:STDERR: FATAL failure at toolchain/semantics/semantics_parse_tree_handler.cpp:579: TODO

var x: i32 = 0;

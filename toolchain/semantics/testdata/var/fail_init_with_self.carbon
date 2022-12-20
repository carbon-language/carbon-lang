// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/var/todo_bad_init_with_self.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionDefinitionStart
// CHECK:STDERR: 	1.	VariableIntroducer
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: 	0.	block0
// CHECK:STDERR: 	1.	block1
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xef323)[0x55d26ef99323]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xed06e)[0x55d26ef9706e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xef71a)[0x55d26ef9971a]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3daa0)[0x7f47b783daa0]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8957c)[0x7f47b788957c]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(raise+0x12)[0x7f47b783da02]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xcf)[0x7f47b7828469]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xc3c32)[0x55d26ef6dc32]
// CHECK:STDERR: FATAL failure at ./toolchain/semantics/semantics_ir.h:105: Need to think more about this case

fn Main() {
  var x: i32 = x;
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/operators/fail_type_mismatch_once.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionIntroducer
// CHECK:STDERR: 	1.	DeclaredName
// CHECK:STDERR: 	2.	ParameterListStart
// CHECK:STDERR: 	3.	ParameterList -> node3
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3ad3)[0x5625b2053ad3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf181e)[0x5625b205181e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3eca)[0x5625b2053eca]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7fce5d9caf90]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9e27c)[0x5625b1ffe27c]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9b768)[0x5625b1ffb768]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x99bc6)[0x5625b1ff9bc6]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x93b93)[0x5625b1ff3b93]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x930ea)[0x5625b1ff30ea]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92ccb)[0x5625b1ff2ccb]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x2718a)[0x7fce5d9b618a]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x85)[0x7fce5d9b6245]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92aa1)[0x5625b1ff2aa1]

fn Main() {
  // The following line has two mismatches, but after the first, it shouldn't
  // keep erroring.
  return 12 + 3.4 + 12;
}

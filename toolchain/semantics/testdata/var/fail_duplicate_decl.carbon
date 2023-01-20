// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/var/fail_duplicate_decl.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionIntroducer
// CHECK:STDERR: 	1.	DeclaredName
// CHECK:STDERR: 	2.	ParameterListStart
// CHECK:STDERR: 	3.	ParameterList -> node3
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3ad3)[0x55e3591d8ad3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf181e)[0x55e3591d681e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3eca)[0x55e3591d8eca]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7ff8942c5f90]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9e27c)[0x55e35918327c]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9b768)[0x55e359180768]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x99bc6)[0x55e35917ebc6]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x93b93)[0x55e359178b93]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x930ea)[0x55e3591780ea]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92ccb)[0x55e359177ccb]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x2718a)[0x7ff8942b118a]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x85)[0x7ff8942b1245]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92aa1)[0x55e359177aa1]


fn Main() {
  var x: i32 = 0;
  var x: i32 = 0;
}

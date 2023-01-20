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
// CHECK:STDERR: 	2.	ParameterListStart
// CHECK:STDERR: 	3.	ParameterList -> node3
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3ad3)[0x555fa36c7ad3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf181e)[0x555fa36c581e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3eca)[0x555fa36c7eca]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7fbba758df90]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9e27c)[0x555fa367227c]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9b768)[0x555fa366f768]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x99bc6)[0x555fa366dbc6]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x93b93)[0x555fa3667b93]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x930ea)[0x555fa36670ea]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92ccb)[0x555fa3666ccb]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x2718a)[0x7fbba757918a]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x85)[0x7fbba7579245]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92aa1)[0x555fa3666aa1]

fn Main() {
  var x: i32 = 0;
  x;
}

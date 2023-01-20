// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/var/decl.carbon
// CHECK:STDERR: 1.	node_stack_:
// CHECK:STDERR: 	0.	FunctionIntroducer
// CHECK:STDERR: 	1.	DeclaredName
// CHECK:STDERR: 	2.	ParameterListStart
// CHECK:STDERR: 	3.	ParameterList -> node3
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3ad3)[0x55bb32bb3ad3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf181e)[0x55bb32bb181e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xf3eca)[0x55bb32bb3eca]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3bf90)[0x7f6e4a57cf90]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9e27c)[0x55bb32b5e27c]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x9b768)[0x55bb32b5b768]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x99bc6)[0x55bb32b59bc6]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x93b93)[0x55bb32b53b93]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x930ea)[0x55bb32b530ea]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92ccb)[0x55bb32b52ccb]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x2718a)[0x7f6e4a56818a]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x85)[0x7f6e4a568245]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0x92aa1)[0x55bb32b52aa1]

fn Main() {
  var x: i32;
}

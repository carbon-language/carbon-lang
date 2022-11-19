// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump parse-tree {{.*}}/toolchain/parser/testdata/if/fail_errors.carbon
// CHECK:STDERR: 1.	Parser stack:
// CHECK:STDERR: 	0.	DeclarationLoop @ 27:1: token 0 : Fn
// CHECK:STDERR: 	1.	FunctionDefinitionFinish @ 27:1: token 0 : Fn
// CHECK:STDERR: 	2.	StatementScopeLoop @ 28:3: token 5 : If
// CHECK:STDERR: 	3.	StatementIfConditionFinish @ 30:3: token 14 : If
// CHECK:STDERR: 	cursor	position_ @ 30:9: token 17 : Identifier
// CHECK:STDERR: Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xda1a3)[0x5559b820d1a3]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xd7e5e)[0x5559b820ae5e]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xda5ca)[0x5559b820d5ca]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x3daa0)[0x7f720583daa0]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(+0x8957c)[0x7f720588957c]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(raise+0x12)[0x7f720583da02]
// CHECK:STDERR: /lib/x86_64-linux-gnu/libc.so.6(abort+0xcf)[0x7f7205828469]
// CHECK:STDERR: ./bazel-bin/toolchain/driver/carbon(+0xaea92)[0x5559b81e1a92]
// CHECK:STDERR: CHECK failure at toolchain/lexer/tokenized_buffer.cpp:709: opening_token_info.kind.IsOpeningSymbol(): The token must be an opening group symbol: If

fn F() {
  if a {}
  if () {}
  if (b c) {}
  if (d)
}

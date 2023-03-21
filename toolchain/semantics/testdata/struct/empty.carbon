// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/struct/empty.carbon
// CHECK:STDERR: 1.	SemanticsNodeStack:
// CHECK:STDERR: 	0.	VariableIntroducer
// CHECK:STDERR: 	1.	DeclaredName
// CHECK:STDERR: 2.	node_block_stack_:
// CHECK:STDERR: 	0.	block<invalid>
// CHECK:STDERR: #0 0x000055d426d18367 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (./bazel-bin/toolchain/driver/carbon+0x3dc367)
// CHECK:STDERR: #1 0x000055d426d1612e llvm::sys::RunSignalHandlers() (./bazel-bin/toolchain/driver/carbon+0x3da12e)
// CHECK:STDERR: #2 0x000055d426d18a8a SignalHandler(int) Signals.cpp:0:0
// CHECK:STDERR: #3 0x00007fb40480df90 (/lib/x86_64-linux-gnu/libc.so.6+0x3bf90)
// CHECK:STDERR: #4 0x00007fb40485cccc __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
// CHECK:STDERR: #5 0x00007fb40480def2 raise ./signal/../sysdeps/posix/raise.c:27:6
// CHECK:STDERR: #6 0x00007fb4047f8472 abort ./stdlib/./stdlib/abort.c:81:7
// CHECK:STDERR: #7 0x000055d426b63352 Carbon::Internal::ExitingStream::Done() (./bazel-bin/toolchain/driver/carbon+0x227352)
// CHECK:STDERR: CHECK failure at toolchain/semantics/semantics_node_stack.cpp:73: entry.node_id.is_valid(): Expected valid id on StructLiteral

var y: {} = {};

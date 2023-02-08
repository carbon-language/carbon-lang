// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump semantics-ir {{.*}}/toolchain/semantics/testdata/basics/builtin_types.carbon
// CHECK:STDERR: #0 0x0000557da8ecd4c3 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (./bazel-bin/toolchain/driver/carbon+0xec4c3)
// CHECK:STDERR: #1 0x0000557da8ecb20e llvm::sys::RunSignalHandlers() (./bazel-bin/toolchain/driver/carbon+0xea20e)
// CHECK:STDERR: #2 0x0000557da8ecd8ba SignalHandler(int) Signals.cpp:0:0
// CHECK:STDERR: #3 0x00007fe81c545f90 (/lib/x86_64-linux-gnu/libc.so.6+0x3bf90)
// CHECK:STDERR: #4 0x00007fe81c594ccc __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
// CHECK:STDERR: #5 0x00007fe81c545ef2 raise ./signal/../sysdeps/posix/raise.c:27:6
// CHECK:STDERR: #6 0x00007fe81c530472 abort ./stdlib/./stdlib/abort.c:81:7
// CHECK:STDERR: #7 0x0000557da8ea1dd2 Carbon::Internal::ExitingStream::Done() (./bazel-bin/toolchain/driver/carbon+0xc0dd2)
// CHECK:STDERR: CHECK failure at toolchain/parser/parse_tree.cpp:29: !verify_error: [
// CHECK:STDERR:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDERR:     {kind: 'DeclaredName', text: 'test_i32'},
// CHECK:STDERR:     {kind: 'Literal', text: 'i32'},
// CHECK:STDERR:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDERR:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDERR:   {kind: 'Literal', text: '0'},
// CHECK:STDERR: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDERR:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDERR:     {kind: 'DeclaredName', text: 'test_f64'},
// CHECK:STDERR:     {kind: 'Literal', text: 'f64'},
// CHECK:STDERR:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDERR:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDERR:   {kind: 'Literal', text: '0.1'},
// CHECK:STDERR: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDERR:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDERR:   {kind: 'DeclaredName', text: 'test_str'},
// CHECK:STDERR:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDERR:   {kind: 'Literal', text: '"Test"'},
// CHECK:STDERR: {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDERR: {kind: 'FileEnd', text: ''},
// CHECK:STDERR: ]
// CHECK:STDERR: ParseTree has 20 nodes and no errors, but TokenizedBuffer has 22 tokens.

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_str: String = "Test";

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump parse-tree {{.*}}/toolchain/parser/testdata/basics/var.carbon
// CHECK:STDERR: #0 0x0000561c2dcc4a13 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (./bazel-bin/toolchain/driver/carbon+0xeca13)
// CHECK:STDERR: #1 0x0000561c2dcc275e llvm::sys::RunSignalHandlers() (./bazel-bin/toolchain/driver/carbon+0xea75e)
// CHECK:STDERR: #2 0x0000561c2dcc4e0a SignalHandler(int) Signals.cpp:0:0
// CHECK:STDERR: #3 0x00007f0941b43f90 (/lib/x86_64-linux-gnu/libc.so.6+0x3bf90)
// CHECK:STDERR: #4 0x00007f0941b92ccc __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
// CHECK:STDERR: #5 0x00007f0941b43ef2 raise ./signal/../sysdeps/posix/raise.c:27:6
// CHECK:STDERR: #6 0x00007f0941b2e472 abort ./stdlib/./stdlib/abort.c:81:7
// CHECK:STDERR: #7 0x0000561c2dc99322 Carbon::Internal::ExitingStream::Done() (./bazel-bin/toolchain/driver/carbon+0xc1322)
// CHECK:STDERR: CHECK failure at toolchain/parser/parse_tree.cpp:29: !verify_error: [
// CHECK:STDERR:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDERR:     {kind: 'DeclaredName', text: 'v'},
// CHECK:STDERR:     {kind: 'Literal', text: 'i32'},
// CHECK:STDERR:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDERR:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDERR:   {kind: 'Literal', text: '0'},
// CHECK:STDERR: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDERR:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDERR:     {kind: 'DeclaredName', text: 'w'},
// CHECK:STDERR:     {kind: 'Literal', text: 'i32'},
// CHECK:STDERR:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDERR: {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDERR:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDERR:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDERR:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDERR:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDERR:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDERR:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDERR:     {kind: 'DeclaredName', text: 's'},
// CHECK:STDERR:     {kind: 'VariableInitializer', text: '='},
// CHECK:STDERR:     {kind: 'Literal', text: '"hello"'},
// CHECK:STDERR:   {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDERR: {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDERR: {kind: 'FileEnd', text: ''},
// CHECK:STDERR: ]
// CHECK:STDERR: ParseTree has 24 nodes and no errors, but TokenizedBuffer has 26 tokens.

// NOTE: Move to its own directory when more tests are added.
var v: i32 = 0;
var w: i32;
fn F() {
  var s: String = "hello";
}

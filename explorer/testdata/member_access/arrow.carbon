// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/explorer/explorer {{.*}}/explorer/testdata/member_access/arrow.carbon
// CHECK:STDERR: #0 0x000055ca03f3d9c3 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (./bazel-bin/explorer/explorer+0x1c09c3)
// CHECK:STDERR: #1 0x000055ca03f3b81e llvm::sys::RunSignalHandlers() (./bazel-bin/explorer/explorer+0x1be81e)
// CHECK:STDERR: #2 0x000055ca03f3ddba SignalHandler(int) Signals.cpp:0:0
// CHECK:STDERR: #3 0x00007fca9243daa0 (/lib/x86_64-linux-gnu/libc.so.6+0x3daa0)
// CHECK:STDERR: #4 0x00007fca9248957c __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
// CHECK:STDERR: #5 0x00007fca9243da02 gsignal ./signal/../sysdeps/posix/raise.c:27:6
// CHECK:STDERR: #6 0x00007fca92428469 abort ./stdlib/./stdlib/abort.c:81:7
// CHECK:STDERR: #7 0x000055ca03f043b2 Carbon::Internal::ExitingStream::Done() (./bazel-bin/explorer/explorer+0x1873b2)
// CHECK:STDERR: FATAL failure at explorer/interpreter/value.cpp:172: field access not allowed for value lval<Allocation(1)>

package Foo api;
class X {
  fn F[addr self: Self*]() -> i32 { return self->n; }
  var n: i32;
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  let p: X* = &v;
  return p->n + p->(X.n) + p->F() + p->(X.F)();
}

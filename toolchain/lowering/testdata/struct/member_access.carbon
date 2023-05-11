// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: Please report issues to https://github.com/carbon-language/carbon-lang/issues and include the crash backtrace.
// CHECK:STDERR: Stack dump:
// CHECK:STDERR: 0.	Program arguments: ./bazel-bin/toolchain/driver/carbon dump llvm-ir toolchain/lowering/testdata/struct/member_access.carbon
// CHECK:STDERR: #0 0x000055c5a92f5ee7 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (./bazel-bin/toolchain/driver/carbon+0x3daee7)
// CHECK:STDERR: #1 0x000055c5a92f3cae llvm::sys::RunSignalHandlers() (./bazel-bin/toolchain/driver/carbon+0x3d8cae)
// CHECK:STDERR: #2 0x000055c5a92f660a SignalHandler(int) Signals.cpp:0:0
// CHECK:STDERR: #3 0x00007f14ff0b6f90 (/lib/x86_64-linux-gnu/libc.so.6+0x3bf90)
// CHECK:STDERR: #4 0x00007f14ff105ccc __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
// CHECK:STDERR: #5 0x00007f14ff0b6ef2 raise ./signal/../sysdeps/posix/raise.c:27:6
// CHECK:STDERR: #6 0x00007f14ff0a1472 abort ./stdlib/./stdlib/abort.c:81:7
// CHECK:STDERR: #7 0x000055c5a9141022 Carbon::Internal::ExitingStream::Done() (./bazel-bin/toolchain/driver/carbon+0x226022)
// CHECK:STDERR: FATAL failure at toolchain/lowering/lowering_handle.cpp:125: TODO: Add support: {kind: RealLiteral, arg0: real0, type: nodeFloatingPointType}

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var z: i32 = y;
  return 0;
}

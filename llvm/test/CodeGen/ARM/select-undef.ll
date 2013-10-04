; RUN: llc < %s -march=arm -mcpu=swift -verify-machineinstrs
define i32 @func(i32 %arg0, i32 %arg1) {
entry:
  %cmp = icmp slt i32 %arg0, 10
  %v = select i1 %cmp, i32 undef, i32 %arg1
  ret i32 %v
}

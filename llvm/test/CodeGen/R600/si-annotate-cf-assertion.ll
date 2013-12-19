; REQUIRES: asserts
; XFAIL: *
; RUN: llc -march=r600 -mcpu=SI -asm-verbose=false < %s | FileCheck %s


define void @test(i32 addrspace(1)* %g, i8 addrspace(3)* %l, i32 %x) nounwind {
; CHECK-LABEL: @test:

entry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.bb
    i32 60, label %sw.bb
  ]

sw.bb:
  unreachable

sw.default:
  unreachable

sw.epilog:
  ret void
}


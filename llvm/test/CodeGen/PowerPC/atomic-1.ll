; RUN: llc < %s -march=ppc32 |  FileCheck %s

define i32 @exchange_and_add(i32* %mem, i32 %val) nounwind {
; CHECK: exchange_and_add:
; CHECK: lwarx
  %tmp = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %mem, i32 %val)
; CHECK: stwcx.
  ret i32 %tmp
}

define i32 @exchange_and_cmp(i32* %mem) nounwind {
; CHECK: exchange_and_cmp:
; CHECK: lwarx
  %tmp = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %mem, i32 0, i32 1)
; CHECK: stwcx.
; CHECK: stwcx.
  ret i32 %tmp
}

define i32 @exchange(i32* %mem, i32 %val) nounwind {
; CHECK: exchange:
; CHECK: lwarx
  %tmp = call i32 @llvm.atomic.swap.i32.p0i32(i32* %mem, i32 1)
; CHECK: stwcx.
  ret i32 %tmp
}

declare i32 @llvm.atomic.load.add.i32.p0i32(i32* nocapture, i32) nounwind

declare i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* nocapture, i32, i32) nounwind

declare i32 @llvm.atomic.swap.i32.p0i32(i32* nocapture, i32) nounwind

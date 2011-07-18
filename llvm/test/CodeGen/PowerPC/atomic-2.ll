; RUN: llc < %s -march=ppc64 | FileCheck %s

define i64 @exchange_and_add(i64* %mem, i64 %val) nounwind {
; CHECK: exchange_and_add:
; CHECK: ldarx
  %tmp = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %mem, i64 %val)
; CHECK: stdcx.
  ret i64 %tmp
}

define i64 @exchange_and_cmp(i64* %mem) nounwind {
; CHECK: exchange_and_cmp:
; CHECK: ldarx
  %tmp = call i64 @llvm.atomic.cmp.swap.i64.p0i64(i64* %mem, i64 0, i64 1)
; CHECK: stdcx.
; CHECK: stdcx.
  ret i64 %tmp
}

define i64 @exchange(i64* %mem, i64 %val) nounwind {
; CHECK: exchange:
; CHECK: ldarx
  %tmp = call i64 @llvm.atomic.swap.i64.p0i64(i64* %mem, i64 1)
; CHECK: stdcx.
  ret i64 %tmp
}

declare i64 @llvm.atomic.load.add.i64.p0i64(i64* nocapture, i64) nounwind

declare i64 @llvm.atomic.cmp.swap.i64.p0i64(i64* nocapture, i64, i64) nounwind

declare i64 @llvm.atomic.swap.i64.p0i64(i64* nocapture, i64) nounwind

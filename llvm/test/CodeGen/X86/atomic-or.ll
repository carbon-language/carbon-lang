; RUN: llc < %s -march=x86-64 | FileCheck %s

; rdar://9692967

define void @do_the_sync(i64* %p, i32 %b) nounwind {
entry:
  %p.addr = alloca i64*, align 8
  store i64* %p, i64** %p.addr, align 8
  %tmp = load i64** %p.addr, align 8
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
; CHECK: lock
; CHECK-NEXT: orq     $2147483648
  %0 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %tmp, i64 2147483648)
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  ret void
}
declare i64 @llvm.atomic.load.or.i64.p0i64(i64* nocapture, i64) nounwind
declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind

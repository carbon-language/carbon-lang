; RUN: llc < %s -march=x86-64 | FileCheck %s

; rdar://9692967

define void @t1(i64* %p, i32 %b) nounwind {
entry:
  %p.addr = alloca i64*, align 8
  store i64* %p, i64** %p.addr, align 8
  %tmp = load i64** %p.addr, align 8
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
; CHECK: t1:
; CHECK: movl    $2147483648, %eax
; CHECK: lock
; CHECK-NEXT: orq %r{{.*}}, (%r{{.*}})
  %0 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %tmp, i64 2147483648)
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  ret void
}

define void @t2(i64* %p, i32 %b) nounwind {
entry:
  %p.addr = alloca i64*, align 8
  store i64* %p, i64** %p.addr, align 8
  %tmp = load i64** %p.addr, align 8
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
; CHECK: t2:
; CHECK-NOT: movl
; CHECK: lock
; CHECK-NEXT: orq $2147483644, (%r{{.*}})
  %0 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %tmp, i64 2147483644)
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  ret void
}

declare i64 @llvm.atomic.load.or.i64.p0i64(i64* nocapture, i64) nounwind
declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind

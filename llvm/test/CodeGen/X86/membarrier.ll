; RUN: llc < %s -march=x86-64 -mattr=-sse -O0
; PR9675

define i32 @t() {
entry:
  %i = alloca i32, align 4
  store i32 1, i32* %i, align 4
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  %0 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %i, i32 1)
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  ret i32 0
}

declare i32 @llvm.atomic.load.sub.i32.p0i32(i32* nocapture, i32) nounwind
declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind

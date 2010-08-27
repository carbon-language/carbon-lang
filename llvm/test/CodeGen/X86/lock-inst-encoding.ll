; RUN: llc -O0 --show-mc-encoding < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: f0:
; CHECK: addq %rax, (%rdi)
; CHECK: # encoding: [0xf0,0x48,0x01,0x07]
; CHECK: ret
define void @f0(i64* %a0) {
  %t0 = and i64 1, 1
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true) nounwind
  %1 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %a0, i64 %t0) nounwind
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true) nounwind
  ret void
}

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind

declare i32 @llvm.atomic.load.and.i32.p0i32(i32* nocapture, i32) nounwind

declare i64 @llvm.atomic.load.add.i64.p0i64(i64* nocapture, i64) nounwind

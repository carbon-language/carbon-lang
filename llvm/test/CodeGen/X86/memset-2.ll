; RUN: llc -mtriple=i386-apple-darwin -mcpu=yonah < %s | FileCheck %s

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind

define fastcc void @t1() nounwind {
entry:
; CHECK: t1:
; CHECK: pxor %xmm0, %xmm0
; CHECK: movups %xmm0, 160
; CHECK: movups %xmm0, 144
; CHECK: movups %xmm0, 128
; CHECK: movups %xmm0, 112
; CHECK: movups %xmm0, 96
; CHECK: movups %xmm0, 80
; CHECK: movups %xmm0, 64
; CHECK: movups %xmm0, 48
; CHECK: movups %xmm0, 32
; CHECK: movups %xmm0, 16
; CHECK: movups %xmm0, 0
; CHECK: movl $0, 184
; CHECK: movl $0, 180
; CHECK: movl $0, 176
  call void @llvm.memset.i32( i8* null, i8 0, i32 188, i32 1 ) nounwind
  unreachable
}

define fastcc void @t2(i8 signext %c) nounwind {
entry:
; CHECK: t2:
; CHECK: calll _memset
  call void @llvm.memset.i32( i8* undef, i8 %c, i32 76, i32 1 ) nounwind
  unreachable
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @t3(i8* nocapture %s, i8 %a) nounwind {
entry:
  tail call void @llvm.memset.p0i8.i32(i8* %s, i8 %a, i32 8, i32 1, i1 false)
  ret void
; CHECK: t3:
; CHECK: imull $16843009
}

define void @t4(i8* nocapture %s, i8 %a) nounwind {
entry:
  tail call void @llvm.memset.p0i8.i32(i8* %s, i8 %a, i32 15, i32 1, i1 false)
  ret void
; CHECK: t4:
; CHECK: imull $16843009
; CHECK-NOT: imul
; CHECK: ret
}

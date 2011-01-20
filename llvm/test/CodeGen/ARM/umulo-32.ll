; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck %s

%umul.ty = type { i32, i1 }

define i32 @func(i32 %a) nounwind {
; CHECK: func
; CHECK: muldi3
  %tmp0 = tail call %umul.ty @llvm.umul.with.overflow.i32(i32 %a, i32 37)
  %tmp1 = extractvalue %umul.ty %tmp0, 0
  %tmp2 = select i1 undef, i32 -1, i32 %tmp1
  ret i32 %tmp2
}

declare %umul.ty @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

; RUN: llc < %s -march=x86 -mcpu=yonah | FileCheck %s

define i32 @t1(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.ctlz.i32( i32 %x )
	ret i32 %tmp
; CHECK: t1:
; CHECK: bsrl
; CHECK: cmov
}

declare i32 @llvm.ctlz.i32(i32) nounwind readnone 

define i32 @t2(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.cttz.i32( i32 %x )
	ret i32 %tmp
; CHECK: t2:
; CHECK: bsfl
; CHECK: cmov
}

declare i32 @llvm.cttz.i32(i32) nounwind readnone 

define i16 @t3(i16 %x, i16 %y) nounwind  {
entry:
        %tmp1 = add i16 %x, %y
	%tmp2 = tail call i16 @llvm.ctlz.i16( i16 %tmp1 )		; <i16> [#uses=1]
	ret i16 %tmp2
; CHECK: t3:
; CHECK: bsrw
; CHECK: cmov
}

declare i16 @llvm.ctlz.i16(i16) nounwind readnone 

; Don't generate the cmovne when the source is known non-zero (and bsr would
; not set ZF).
; rdar://9490949

define i32 @t4(i32 %n) nounwind {
entry:
; CHECK: t4:
; CHECK: bsrl
; CHECK-NOT: cmov
; CHECK: ret
  %or = or i32 %n, 1
  %tmp1 = tail call i32 @llvm.ctlz.i32(i32 %or)
  ret i32 %tmp1
}

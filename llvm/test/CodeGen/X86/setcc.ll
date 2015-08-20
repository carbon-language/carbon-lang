; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; rdar://7329206

; Use sbb x, x to materialize carry bit in a GPR. The value is either
; all 1's or all 0's.

define zeroext i16 @t1(i16 zeroext %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: t1:
; CHECK: seta %al
; CHECK: movzbl %al, %eax
; CHECK: shll $5, %eax
  %0 = icmp ugt i16 %x, 26                        ; <i1> [#uses=1]
  %iftmp.1.0 = select i1 %0, i16 32, i16 0        ; <i16> [#uses=1]
  ret i16 %iftmp.1.0
}

define zeroext i16 @t2(i16 zeroext %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: sbbl %eax, %eax
; CHECK: andl $32, %eax
  %0 = icmp ult i16 %x, 26                        ; <i1> [#uses=1]
  %iftmp.0.0 = select i1 %0, i16 32, i16 0        ; <i16> [#uses=1]
  ret i16 %iftmp.0.0
}

define i64 @t3(i64 %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: t3:
; CHECK: sbbq %rax, %rax
; CHECK: andl $64, %eax
  %0 = icmp ult i64 %x, 18                        ; <i1> [#uses=1]
  %iftmp.2.0 = select i1 %0, i64 64, i64 0        ; <i64> [#uses=1]
  ret i64 %iftmp.2.0
}

@v4 = common global i32 0, align 4

define i32 @t4(i32 %a) {
entry:
; CHECK-LABEL: t4:
; CHECK:  movq    _v4@GOTPCREL(%rip), %rax
; CHECK:  cmpl    $1, (%rax)
; CHECK:  sbbl    %eax, %eax
; CHECK:  andl    $32768, %eax
; CHECK:  leal    65536(%rax,%rax), %eax
  %0 = load i32, i32* @v4, align 4
  %not.tobool = icmp eq i32 %0, 0
  %conv.i = sext i1 %not.tobool to i16
  %call.lobit = lshr i16 %conv.i, 15
  %add.i.1 = add nuw nsw i16 %call.lobit, 1
  %conv4.2 = zext i16 %add.i.1 to i32
  %add = shl nuw nsw i32 %conv4.2, 16
  ret i32 %add
}

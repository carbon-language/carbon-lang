; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define zeroext i16 @t1(i16 zeroext %x) nounwind readnone ssp {
entry:
; CHECK: t1:
; CHECK: seta %al
; CHECK: movzbl %al, %eax
; CHECK: shll $5, %eax
  %0 = icmp ugt i16 %x, 26                        ; <i1> [#uses=1]
  %iftmp.1.0 = select i1 %0, i16 32, i16 0        ; <i16> [#uses=1]
  ret i16 %iftmp.1.0
}


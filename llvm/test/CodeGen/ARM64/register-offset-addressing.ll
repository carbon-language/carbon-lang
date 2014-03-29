; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

define i8 @t1(i16* %a, i64 %b) {
; CHECK: t1
; CHECK: lsl [[REG:x[0-9]+]], x1, #1
; CHECK: ldrb w0, [x0, [[REG]]]
; CHECK: ret
  %tmp1 = getelementptr inbounds i16* %a, i64 %b
  %tmp2 = load i16* %tmp1
  %tmp3 = trunc i16 %tmp2 to i8
  ret i8 %tmp3
}

; RUN: llc < %s -march=x86 -mcpu=yonah | FileCheck %s


; Verify that we don't fail when shift by zero is encountered.

define i64 @test1(<2 x i64> %a) {
entry:
 %c = shl <2 x i64> %a, <i64 0, i64 2>
 %d = extractelement <2 x i64> %c, i32 0
 ret i64 %d
}
; CHECK-LABEL: test1

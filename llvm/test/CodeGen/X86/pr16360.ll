; RUN: llc < %s -mcpu=pentium4 -mtriple=i686-pc-linux | FileCheck %s

define i64 @foo(i32 %sum) {
entry:
  %conv = sext i32 %sum to i64
  %shr = lshr i64 %conv, 2
  %or = or i64 4611686018360279040, %shr
  ret i64 %or
}

; CHECK: foo
; CHECK: shrl $2
; CHECK: orl $-67108864
; CHECK-NOT: movl $-1
; CHECK: movl $1073741823
; CHECK: ret

; PR13504
; RUN: llc -mtriple=i686-- -mcpu=atom < %s | FileCheck %s
; Check that treemap is read before the asm statement.
; CHECK: movl 8(%{{esp|ebp}})
; CHECK: bsfl
; CHECK-NOT: movl 8(%{{esp|ebp}})

define i32 @foo(i32 %treemap) nounwind uwtable {
entry:
  %sub = sub i32 0, %treemap
  %and = and i32 %treemap, %sub
  %0 = tail call i32 asm "bsfl $1,$0\0A\09", "=r,rm,~{dirflag},~{fpsr},~{flags}"(i32 %and) nounwind
  ret i32 %0
}


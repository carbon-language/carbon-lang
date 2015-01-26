; Tests that check our handling of volatile instructions encountered
; when scanning for dependencies
; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

; Check that we can bypass a volatile load when searching
; for dependencies of a non-volatile load
define i32 @test1(i32* nocapture %p, i32* nocapture %q) {
; CHECK-LABEL: test1
; CHECK:      %0 = load volatile i32* %q
; CHECK-NEXT: ret i32 0
entry:
  %x = load i32* %p
  load volatile i32* %q
  %y = load i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; We can not value forward if the query instruction is 
; volatile, this would be (in effect) removing the volatile load
define i32 @test2(i32* nocapture %p, i32* nocapture %q) {
; CHECK-LABEL: test2
; CHECK:      %x = load i32* %p
; CHECK-NEXT: %y = load volatile i32* %p
; CHECK-NEXT: %add = sub i32 %y, %x
entry:
  %x = load i32* %p
  %y = load volatile i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; If the query instruction is itself volatile, we *cannot*
; reorder it even if p and q are noalias
define i32 @test3(i32* noalias nocapture %p, i32* noalias nocapture %q) {
; CHECK-LABEL: test3
; CHECK:      %x = load i32* %p
; CHECK-NEXT: %0 = load volatile i32* %q
; CHECK-NEXT: %y = load volatile i32* %p
entry:
  %x = load i32* %p
  load volatile i32* %q
  %y = load volatile i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; If an encountered instruction is both volatile and ordered, 
; we need to use the strictest ordering of either.  In this 
; case, the ordering prevents forwarding.
define i32 @test4(i32* noalias nocapture %p, i32* noalias nocapture %q) {
; CHECK-LABEL: test4
; CHECK:      %x = load i32* %p
; CHECK-NEXT: %0 = load atomic volatile i32* %q seq_cst 
; CHECK-NEXT: %y = load atomic i32* %p seq_cst
entry:
  %x = load i32* %p
  load atomic volatile i32* %q seq_cst, align 4
  %y = load atomic i32* %p seq_cst, align 4
  %add = sub i32 %y, %x
  ret i32 %add
}

; Value forwarding from a volatile load is perfectly legal
define i32 @test5(i32* nocapture %p, i32* nocapture %q) {
; CHECK-LABEL: test5
; CHECK:      %x = load volatile i32* %p
; CHECK-NEXT: ret i32 0
entry:
  %x = load volatile i32* %p
  %y = load i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}


; RUN: opt -instsimplify < %s -S | FileCheck %s

; CHECK: define i1 @test
define i1 @test(i8* %pq, i8 %B) {
  %q = load i8, i8* %pq, !range !0 ; %q is known nonzero; no known bits
  %A = add nsw i8 %B, %q
  %cmp = icmp eq i8 %A, %B
  ; CHECK: ret i1 false
  ret i1 %cmp
}

; CHECK: define i1 @test2
define i1 @test2(i8 %a, i8 %b) {
  %A = or i8 %a, 2    ; %A[1] = 1
  %B = and i8 %b, -3  ; %B[1] = 0
  %cmp = icmp eq i8 %A, %B ; %A[1] and %B[1] are contradictory.
  ; CHECK: ret i1 false
  ret i1 %cmp
}

!0 = !{ i8 1, i8 5 }

; RUN: llc < %s -mtriple x86_64-apple-darwin11 -O0 | FileCheck %s

%struct.x = type { i64, i64 }
declare %struct.x @f()

define void @test1(i64*) nounwind ssp {
  %2 = tail call %struct.x @f() nounwind
  %3 = extractvalue %struct.x %2, 0
  %4 = add i64 %3, 10
  store i64 %4, i64* %0
  ret void
; CHECK: test1:
; CHECK: callq _f
; CHECK-NEXT: addq	$10, %rax
}

define void @test2(i64*) nounwind ssp {
  %2 = tail call %struct.x @f() nounwind
  %3 = extractvalue %struct.x %2, 1
  %4 = add i64 %3, 10
  store i64 %4, i64* %0
  ret void
; CHECK: test2:
; CHECK: callq _f
; CHECK-NEXT: addq	$10, %rdx
}

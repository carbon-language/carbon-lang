; RUN: llc -march=xcore < %s | FileCheck %s

; CHECK-LABEL: test:
; CHECK-NOT: zext
define void @test(i8* %s1) {
entry:
  %u8 = load i8, i8* %s1, align 1
  %bool = icmp eq i8 %u8, 0
  br label %BB1
BB1:
  br i1 %bool, label %BB1, label %BB2
BB2:
  br i1 %bool, label %BB1, label %BB2
}


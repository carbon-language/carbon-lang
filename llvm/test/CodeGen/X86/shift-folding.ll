; RUN: llc < %s -march=x86 | FileCheck %s

define i32* @test1(i32* %P, i32 %X) {
; CHECK: test1:
; CHECK-NOT: shrl
; CHECK-NOT: shll
; CHECK: ret

entry:
  %Y = lshr i32 %X, 2
  %gep.upgrd.1 = zext i32 %Y to i64
  %P2 = getelementptr i32* %P, i64 %gep.upgrd.1
  ret i32* %P2
}

define i32* @test2(i32* %P, i32 %X) {
; CHECK: test2:
; CHECK: shll $4
; CHECK-NOT: shll
; CHECK: ret

entry:
  %Y = shl i32 %X, 2
  %gep.upgrd.2 = zext i32 %Y to i64
  %P2 = getelementptr i32* %P, i64 %gep.upgrd.2
  ret i32* %P2
}

define i32* @test3(i32* %P, i32 %X) {
; CHECK: test3:
; CHECK-NOT: shrl
; CHECK-NOT: shll
; CHECK: ret

entry:
  %Y = ashr i32 %X, 2
  %P2 = getelementptr i32* %P, i32 %Y
  ret i32* %P2
}

define fastcc i32 @test4(i32* %d) {
; CHECK: test4:
; CHECK-NOT: shrl
; CHECK: ret

entry:
  %tmp4 = load i32* %d
  %tmp512 = lshr i32 %tmp4, 24
  ret i32 %tmp512
}

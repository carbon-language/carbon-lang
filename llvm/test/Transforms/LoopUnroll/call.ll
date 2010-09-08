; RUN: opt < %s -S -loop-unroll | FileCheck %s

@id = internal global i32 0
@val = internal global [4 x i32] zeroinitializer, align 16

; CHECK: @test
define i32 @test(i32 %k) nounwind ssp {
; CHECK-NOT: call i32 @test(i32 %t.06)
; CHECK: call i32 @test(i32 0)
; CHECK-NOT: call i32 @test(i32 %t.06)
; CHECK: call i32 @test(i32 1)
; CHECK-NOT: call i32 @test(i32 %t.06)
; CHECK: call i32 @test(i32 2)
; CHECK-NOT: call i32 @test(i32 %t.06)
; CHECK: call i32 @test(i32 3)
; CHECK-NOT: call i32 @test(i32 %t.06)

bb.nph:
  %0 = load i32* @id, align 4
  %1 = add nsw i32 %0, 1
  store i32 %1, i32* @id, align 4
  %2 = sext i32 %k to i64
  %3 = getelementptr inbounds [4 x i32]* @val, i64 0, i64 %2
  store i32 %1, i32* %3, align 4
  br label %bb

bb:                                               ; preds = %bb2, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb2 ]
  %scevgep = getelementptr [4 x i32]* @val, i64 0, i64 %indvar
  %4 = load i32* %scevgep, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  %t.06 = trunc i64 %indvar to i32
  %6 = tail call i32 @test(i32 %t.06) nounwind
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 4
  br i1 %exitcond, label %bb4, label %bb

bb4:                                              ; preds = %bb2
  %.pre = load i32* @id, align 4
  %7 = add nsw i32 %.pre, -1
  store i32 %7, i32* @id, align 4
  store i32 0, i32* %3, align 4
  ret i32 undef
; CHECK: }
}

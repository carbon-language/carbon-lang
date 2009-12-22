; Test CFG simplify removal of branch instructions.
;
; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @test1() {
        br label %BB1
BB1:            ; preds = %0
        ret void
; CHECK: @test1
; CHECK-NEXT: ret void
}

define void @test2() {
        ret void
BB1:            ; No predecessors!
        ret void
; CHECK: @test2
; CHECK-NEXT: ret void
; CHECK-NEXT: }
}

define void @test3(i1 %T) {
        br i1 %T, label %BB1, label %BB1
BB1:            ; preds = %0, %0
        ret void
; CHECK: @test3
; CHECK-NEXT: ret void
}


define void @test4() {
  br label %return
return:
  ret void
; CHECK: @test4
; CHECK-NEXT: ret void
}
@test4g = global i8* blockaddress(@test4, %return)



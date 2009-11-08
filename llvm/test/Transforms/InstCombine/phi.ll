; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128:n8:16:32:64"

define i32 @test1(i32 %A, i1 %b) {
BB0:
        br i1 %b, label %BB1, label %BB2

BB1:
        ; Combine away one argument PHI nodes
        %B = phi i32 [ %A, %BB0 ]               
        ret i32 %B

BB2:
        ret i32 %A
; CHECK: @test1
; CHECK: BB1:
; CHECK-NEXT: ret i32 %A
}

define i32 @test2(i32 %A, i1 %b) {
BB0:
        br i1 %b, label %BB1, label %BB2

BB1:
        br label %BB2

BB2:
        ; Combine away PHI nodes with same values
        %B = phi i32 [ %A, %BB0 ], [ %A, %BB1 ]         
        ret i32 %B
; CHECK: @test2
; CHECK: BB2:
; CHECK-NEXT: ret i32 %A
}

define i32 @test3(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ %B, %Loop ]
        br i1 %b, label %Loop, label %Exit

Exit:
        ret i32 %B
; CHECK: @test3
; CHECK: Exit:
; CHECK-NEXT: ret i32 %A
}

define i32 @test4(i1 %b) {
BB0:
        ; Loop is unreachable
        ret i32 7

Loop:           ; preds = %L2, %Loop
        ; PHI has same value always.
        %B = phi i32 [ %B, %L2 ], [ %B, %Loop ]         
        br i1 %b, label %L2, label %Loop

L2:             ; preds = %Loop
        br label %Loop
; CHECK: @test4
; CHECK: Loop:
; CHECK-NEXT: br i1 %b
}

define i32 @test5(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ undef, %Loop ]             
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 %B
; CHECK: @test5
; CHECK: Loop:
; CHECK-NEXT: br i1 %b
; CHECK: Exit:
; CHECK-NEXT: ret i32 %A
}

define i32 @test6(i16 %A, i1 %b) {
BB0:
        %X = zext i16 %A to i32              
        br i1 %b, label %BB1, label %BB2

BB1:           
        %Y = zext i16 %A to i32              
        br label %BB2

BB2:           
        ;; Suck casts into phi
        %B = phi i32 [ %X, %BB0 ], [ %Y, %BB1 ]         
        ret i32 %B
; CHECK: @test6
; CHECK: BB2:
; CHECK: zext i16 %A to i32
; CHECK-NEXT: ret i32
}

define i32 @test7(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI is dead.
        %B = phi i32 [ %A, %BB0 ], [ %C, %Loop ]                
        %C = add i32 %B, 123            
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 0
; CHECK: @test7
; CHECK: Loop:
; CHECK-NEXT: br i1 %b
}

define i32* @test8({ i32, i32 } *%A, i1 %b) {
BB0:
        %X = getelementptr { i32, i32 } *%A, i32 0, i32 1
        br i1 %b, label %BB1, label %BB2

BB1:
        %Y = getelementptr { i32, i32 } *%A, i32 0, i32 1
        br label %BB2

BB2:
        ;; Suck GEPs into phi
        %B = phi i32* [ %X, %BB0 ], [ %Y, %BB1 ]
        ret i32* %B
; CHECK: @test8
; CHECK-NOT: phi
; CHECK: BB2:
; CHECK-NEXT: %B = getelementptr 
; CHECK-NEXT: ret i32* %B
}

define i32 @test9(i32* %A, i32* %B) {
entry:
  %c = icmp eq i32* %A, null
  br i1 %c, label %bb1, label %bb

bb:
  %C = load i32* %B, align 1
  br label %bb2

bb1:
  %D = load i32* %A, align 1
  br label %bb2

bb2:
  %E = phi i32 [ %C, %bb ], [ %D, %bb1 ]
  ret i32 %E
; CHECK: @test9
; CHECK:       bb2:
; CHECK-NEXT:        phi i32* [ %B, %bb ], [ %A, %bb1 ]
; CHECK-NEXT:   %E = load i32* %{{[^,]*}}, align 1
; CHECK-NEXT:   ret i32 %E

}

define i32 @test10(i32* %A, i32* %B) {
entry:
  %c = icmp eq i32* %A, null
  br i1 %c, label %bb1, label %bb

bb:
  %C = load i32* %B, align 16
  br label %bb2

bb1:
  %D = load i32* %A, align 32
  br label %bb2

bb2:
  %E = phi i32 [ %C, %bb ], [ %D, %bb1 ]
  ret i32 %E
; CHECK: @test10
; CHECK:       bb2:
; CHECK-NEXT:        phi i32* [ %B, %bb ], [ %A, %bb1 ]
; CHECK-NEXT:   %E = load i32* %{{[^,]*}}, align 16
; CHECK-NEXT:   ret i32 %E
}


; PR1777
declare i1 @test11a()

define i1 @test11() {
entry:
  %a = alloca i32
  %i = ptrtoint i32* %a to i32
  %b = call i1 @test11a()
  br i1 %b, label %one, label %two

one:
  %x = phi i32 [%i, %entry], [%y, %two]
  %c = call i1 @test11a()
  br i1 %c, label %two, label %end

two:
  %y = phi i32 [%i, %entry], [%x, %one]
  %d = call i1 @test11a()
  br i1 %d, label %one, label %end

end:
  %f = phi i32 [ %x, %one], [%y, %two]
  ; Change the %f to %i, and the optimizer suddenly becomes a lot smarter
  ; even though %f must equal %i at this point
  %g = inttoptr i32 %f to i32*
  store i32 10, i32* %g
  %z = call i1 @test11a()
  ret i1 %z
; CHECK: @test11
; CHECK-NOT: phi i32
; CHECK: ret i1 %z
}


define i64 @test12(i1 %cond, i8* %Ptr, i64 %Val) {
entry:
  %tmp41 = ptrtoint i8* %Ptr to i64
  %tmp42 = zext i64 %tmp41 to i128
  br i1 %cond, label %end, label %two

two:
  %tmp36 = zext i64 %Val to i128            ; <i128> [#uses=1]
  %tmp37 = shl i128 %tmp36, 64                    ; <i128> [#uses=1]
  %ins39 = or i128 %tmp42, %tmp37                 ; <i128> [#uses=1]
  br label %end

end:
  %tmp869.0 = phi i128 [ %tmp42, %entry ], [ %ins39, %two ]
  %tmp32 = trunc i128 %tmp869.0 to i64            ; <i64> [#uses=1]
  %tmp29 = lshr i128 %tmp869.0, 64                ; <i128> [#uses=1]
  %tmp30 = trunc i128 %tmp29 to i64               ; <i64> [#uses=1]

  %tmp2 = add i64 %tmp32, %tmp30
  ret i64 %tmp2
; CHECK: @test12
; CHECK-NOT: zext
; CHECK: end:
; CHECK-NEXT: phi i64 [ 0, %entry ], [ %Val, %two ]
; CHECK-NOT: phi
; CHECK: ret i64
}


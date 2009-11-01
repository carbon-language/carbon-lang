; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

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
; CHECK:       bb2:
; CHECK-NEXT:        phi i32* [ %B, %bb ], [ %A, %bb1 ]
; CHECK-NEXT:   %E = load i32* %{{[^,]*}}, align 1
; CHECK-NEXT:   ret i32 %E

}

; RUN: opt < %s -constprop -die -S | FileCheck %s

; This is a basic sanity check for constant propogation.  The add instruction 
; should be eliminated.
define i32 @test1(i1 %B) {
        br i1 %B, label %BB1, label %BB2

BB1:      
        %Val = add i32 0, 0
        br label %BB3

BB2:      
        br label %BB3

BB3:     
; CHECK: @test1
; CHECK: %Ret = phi i32 [ 0, %BB1 ], [ 1, %BB2 ]
        %Ret = phi i32 [ %Val, %BB1 ], [ 1, %BB2 ] 
        ret i32 %Ret
}


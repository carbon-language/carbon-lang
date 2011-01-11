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


; PR6197
define i1 @test2(i8* %f) nounwind {
entry:
  %V = icmp ne i8* blockaddress(@test2, %bb), null
  br label %bb
bb:
  ret i1 %V
  
; CHECK: @test2
; CHECK: ret i1 true
}

define i1 @TNAN() {
; CHECK: @TNAN
; CHECK: ret i1 true
  %A = fcmp uno double 0x7FF8000000000000, 1.000000e+00
  %B = fcmp uno double 1.230000e+02, 1.000000e+00
  %C = or i1 %A, %B
  ret i1 %C
}

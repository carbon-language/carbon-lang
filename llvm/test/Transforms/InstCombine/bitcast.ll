; RUN: opt < %s -instcombine -S | FileCheck %s

; Bitcasts between vectors and scalars are valid.
; PR4487
define i32 @test1(i64 %a) {
        %t1 = bitcast i64 %a to <2 x i32>
        %t2 = bitcast i64 %a to <2 x i32>
        %t3 = xor <2 x i32> %t1, %t2
        %t4 = extractelement <2 x i32> %t3, i32 0
        ret i32 %t4
        
; CHECK: @test1
; CHECK: ret i32 0
}


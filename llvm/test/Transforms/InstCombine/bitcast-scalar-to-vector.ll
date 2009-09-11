; RUN: opt < %s -instcombine -S | grep {ret i32 0}
; PR4487

; Bitcasts between vectors and scalars are valid, despite being ill-advised.

define i32 @test(i64 %a) {
bb20:
        %t1 = bitcast i64 %a to <2 x i32>
        %t2 = bitcast i64 %a to <2 x i32>
        %t3 = xor <2 x i32> %t1, %t2
        %t4 = extractelement <2 x i32> %t3, i32 0
        ret i32 %t4
}


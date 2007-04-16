; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v OK | not grep add

define i64 @test1(i64 %A, i32 %B) {
        %tmp12 = zext i32 %B to i64
        %tmp3 = shl i64 %tmp12, 32
        %tmp5 = add i64 %tmp3, %A
        %tmp6 = and i64 %tmp5, 123
        ret i64 %tmp6
}


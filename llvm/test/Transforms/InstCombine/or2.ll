; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep { or}

define i32 @test1(i32 %b, i32 %c, i32 %d) {
        %tmp3 = and i32 %c, %b
        %tmp4not = xor i32 %b, -1
        %tmp6 = and i32 %d, %tmp4not
        %tmp7 = or i32 %tmp6, %tmp3
        ret i32 %tmp7
}


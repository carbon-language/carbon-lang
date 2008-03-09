; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep ret | count 3
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep call.*bswap

define i1 @test1(i16 %tmp2) {
        %tmp10 = call i16 @llvm.bswap.i16( i16 %tmp2 )          ; <i16> [#uses=1]
        %tmp = icmp eq i16 %tmp10, 1            ; <i1> [#uses=1]
        ret i1 %tmp
}

define i1 @test2(i32 %tmp) {
        %tmp34 = tail call i32 @llvm.bswap.i32( i32 %tmp )              ; <i32> [#uses=1]
        %tmp.upgrd.1 = icmp eq i32 %tmp34, 1            ; <i1> [#uses=1]
        ret i1 %tmp.upgrd.1
}

declare i32 @llvm.bswap.i32(i32)

define i1 @test3(i64 %tmp) {
        %tmp34 = tail call i64 @llvm.bswap.i64( i64 %tmp )              ; <i64> [#uses=1]
        %tmp.upgrd.2 = icmp eq i64 %tmp34, 1            ; <i1> [#uses=1]
        ret i1 %tmp.upgrd.2
}

declare i64 @llvm.bswap.i64(i64)

declare i16 @llvm.bswap.i16(i16)


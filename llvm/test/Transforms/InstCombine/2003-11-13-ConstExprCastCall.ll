; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep call | notcast

declare void @free(i8*)

define void @test(i32* %X) {
        call i32 (...)* bitcast (void (i8*)* @free to i32 (...)*)( i32* %X )            ; <i32>:1 [#uses=0]
        ret void
}


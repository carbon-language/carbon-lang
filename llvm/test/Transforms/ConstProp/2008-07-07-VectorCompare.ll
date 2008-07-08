; RUN: llvm-as < %s | opt -constprop | llvm-dis
; PR2529
define <4 x i32> @test1(i32 %argc, i8** %argv) {
entry:  
        %foo = vicmp slt <4 x i32> undef, <i32 14, i32 undef, i32 undef, i32 undef>
        ret <4 x i32> %foo
}

define <4 x i32> @main(i32 %argc, i8** %argv) {
entry:  
        %foo = vicmp slt <4 x i32> <i32 undef, i32 undef, i32 undef, i32
undef>, <i32 undef, i32 undef, i32 undef, i32 undef>
        ret <4 x i32> %foo
}

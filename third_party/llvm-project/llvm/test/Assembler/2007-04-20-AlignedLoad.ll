; RUN: llvm-as < %s | llvm-dis | grep "align 1024"
; RUN: verify-uselistorder %s

define i32 @test(i32* %arg) {
entry:
        %tmp2 = load i32, i32* %arg, align 1024      ; <i32> [#uses=1]
        ret i32 %tmp2
}

; RUN: llvm-as < %s | llvm-dis | grep "align 1024"
; RUN: verify-uselistorder %s

define void @test(i32* %arg) {
entry:
        store i32 0, i32* %arg, align 1024
        ret void
}

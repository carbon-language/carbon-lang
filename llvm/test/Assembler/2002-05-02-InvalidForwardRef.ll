; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s -preserve-bc-use-list-order
; It looks like the assembler is not forward resolving the function declaraion
; correctly.

define void @test() {
        call void @foo( )
        ret void
}

declare void @foo()

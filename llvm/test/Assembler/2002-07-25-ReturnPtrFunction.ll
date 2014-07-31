; Test that returning a pointer to a function causes the disassembler to print 
; the right thing.
;
; RUN: llvm-as < %s | llvm-dis | llvm-as
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

declare void (i32)* @foo()

define void @test() {
        call void (i32)* ()* @foo( )           ; <%ty*>:1 [#uses=0]
        ret void
}



; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

define i32 @test() {
        ret i32 ashr (i32 ptrtoint (i32 ()* @test to i32), i32 2)
}

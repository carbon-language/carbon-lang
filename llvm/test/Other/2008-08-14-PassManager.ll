; RUN: opt < %s -loop-deletion -loop-index-split -disable-output
; PR2640
define i32 @test1() {
       ret i32 0
}

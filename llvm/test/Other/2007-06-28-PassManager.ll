; RUN:  llvm-as < %s |  opt -analyze -inline -disable-output
; PR 1526
define i32 @test1() {
       ret i32 0;
}

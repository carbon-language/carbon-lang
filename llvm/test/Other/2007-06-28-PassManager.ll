; RUN: opt < %s -analyze -inline -disable-output
; PR1526
; RUN: opt < %s -analyze -indvars -disable-output
; PR1539
define i32 @test1() {
       ret i32 0
}

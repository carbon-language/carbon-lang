; PR 1258
; RUN: llvm-as < %s >/dev/null -f |& grep {Numbered.*does not match}

define i32 @test1(i32 %a, i32 %b) {
entry:
  icmp eq i32 %b, %a              ; <i1>:0 [#uses=1]
  zext i1 %0 to i32               ; <i32>:0 [#uses=1]
  ret i32 %0                      ; Invalid Type for %0
}

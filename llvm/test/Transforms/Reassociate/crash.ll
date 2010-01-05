; RUN: opt -reassociate -disable-output %s


; rdar://7507855
define fastcc i32 @test() nounwind {
entry:
  %cond = select i1 undef, i32 1, i32 -1          ; <i32> [#uses=2]
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %sub889 = sub i32 undef, undef                  ; <i32> [#uses=1]
  %sub891 = sub i32 %sub889, %cond                ; <i32> [#uses=0]
  %add896 = sub i32 0, %cond                      ; <i32> [#uses=0]
  ret i32 undef
}

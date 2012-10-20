; RUN: opt < %s  -loop-vectorize -dce

; Check that we don't fall into an infinite loop.
define void @test() nounwind {
entry:
 br label %for.body

for.body:
 %0 = phi i32 [ 1, %entry ], [ 0, %for.body ]
 br label %for.body
}


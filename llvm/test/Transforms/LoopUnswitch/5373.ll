; RUN: opt < %s -loop-unswitch -stats -disable-output |& grep "3 loop-unswitch - Number of branches unswitched"

define noalias i32* @func_16(i32** %p_18, i32* %p_20) noreturn nounwind ssp {
entry:
  %lnot = icmp eq i32** %p_18, null               ; <i1> [#uses=1]
  %lnot6 = icmp eq i32* %p_20, null               ; <i1> [#uses=1]
  br label %for.body

for.body:                                         ; preds = %cond.end, %entry
  br i1 %lnot, label %cond.end, label %cond.true

cond.true:                                        ; preds = %for.body
  tail call void @f()
  unreachable

cond.end:                                         ; preds = %for.body
  br i1 %lnot6, label %for.body, label %cond.true10

cond.true10:                                      ; preds = %cond.end
  tail call void @f()
  unreachable
}

declare void @f() noreturn

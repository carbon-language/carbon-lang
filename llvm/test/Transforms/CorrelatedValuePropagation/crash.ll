; RUN: opt < %s -correlated-propagation

; PR8161
define void @test1() nounwind ssp {
entry:
  br label %for.end

for.cond.us.us:                                   ; preds = %for.cond.us.us
  %cmp6.i.us.us = icmp sgt i32 1, 0
  %lor.ext.i.us.us = zext i1 %cmp6.i.us.us to i32
  %lor.ext.add.i.us.us = select i1 %cmp6.i.us.us, i32 %lor.ext.i.us.us, i32 undef
  %conv.i.us.us = trunc i32 %lor.ext.add.i.us.us to i16
  %sext.us.us = shl i16 %conv.i.us.us, 8
  %conv6.us.us = ashr i16 %sext.us.us, 8
  %and.us.us = and i16 %conv6.us.us, %and.us.us
  br i1 false, label %for.end, label %for.cond.us.us

for.end:                                          ; preds = %for.cond.us, %for.cond.us.us, %entry
  ret void
}

; PR 8790
define void @test2() nounwind ssp {
entry:
  br label %func_29.exit

sdf.exit.i:
  %l_44.1.mux.i = select i1 %tobool5.not.i, i8 %l_44.1.mux.i, i8 1
  br label %srf.exit.i

srf.exit.i:
  %tobool5.not.i = icmp ne i8 undef, 0
  br i1 %tobool5.not.i, label %sdf.exit.i, label %func_29.exit

func_29.exit:
  ret void
}

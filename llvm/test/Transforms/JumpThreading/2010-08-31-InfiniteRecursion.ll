; RUN: opt < %s -jump-threading -disable-output

define void @test() nounwind ssp {
entry:
  br i1 undef, label %bb269.us.us, label %bb269.us.us.us

bb269.us.us.us:
  %indvar = phi i64 [ %indvar.next, %bb287.us.us.us ], [ 0, %entry ]
  %0 = icmp eq i16 undef, 0
  br i1 %0, label %bb287.us.us.us, label %bb286.us.us.us

bb287.us.us.us:
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 4
  br i1 %exitcond, label %bb288.bb289.loopexit_crit_edge, label %bb269.us.us.us

bb286.us.us.us:
  unreachable

bb269.us.us:
	unreachable

bb288.bb289.loopexit_crit_edge:
  unreachable
}

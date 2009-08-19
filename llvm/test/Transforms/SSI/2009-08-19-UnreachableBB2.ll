; RUN: llvm-as < %s | opt -ssi-everything -disable-output

define void @foo() {
entry:
	%tmp0 = load i64* undef, align 4		; <i64> [#uses=3]
	br i1 undef, label %end_stmt_playback, label %bb16

readJournalHdr.exit:		; No predecessors!
	br label %end_stmt_playback

bb16:		; preds = %bb7
	%tmp1 = icmp slt i64 0, %tmp0		; <i1> [#uses=1]
	br i1 %tmp1, label %bb16, label %bb17

bb17:		; preds = %bb16
	store i64 %tmp0, i64* undef, align 4
	br label %end_stmt_playback

end_stmt_playback:		; preds = %bb17, %readJournalHdr.exit, %bb6, %bb2
	store i64 %tmp0, i64* undef, align 4
	ret void
}

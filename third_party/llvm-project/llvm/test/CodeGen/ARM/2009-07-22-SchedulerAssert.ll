; RUN: llc -mtriple=arm-eabi %s -o /dev/null

	%struct.cli_ac_alt = type { i8, i8*, i16, i16, %struct.cli_ac_alt* }
	%struct.cli_ac_node = type { i8, i8, %struct.cli_ac_patt*, %struct.cli_ac_node**, %struct.cli_ac_node* }
	%struct.cli_ac_patt = type { i16*, i16*, i16, i16, i8, i32, i32, i8*, i8*, i32, i16, i16, i16, i16, %struct.cli_ac_alt**, i8, i16, %struct.cli_ac_patt*, %struct.cli_ac_patt* }
	%struct.cli_bm_patt = type { i8*, i8*, i16, i16, i8*, i8*, i8, %struct.cli_bm_patt*, i16 }
	%struct.cli_matcher = type { i16, i8, i8*, %struct.cli_bm_patt**, i32*, i32, i8, i8, %struct.cli_ac_node*, %struct.cli_ac_node**, %struct.cli_ac_patt**, i32, i32, i32 }

define i32 @cli_ac_addsig(%struct.cli_matcher* nocapture %root, i8* %virname, i8* %hexsig, i32 %sigid, i16 zeroext %parts, i16 zeroext %partno, i16 zeroext %type, i32 %mindist, i32 %maxdist, i8* %offset, i8 zeroext %target) nounwind {
entry:
	br i1 undef, label %bb126, label %bb1

bb1:		; preds = %entry
	br i1 undef, label %cli_calloc.exit.thread, label %cli_calloc.exit

cli_calloc.exit.thread:		; preds = %bb1
	ret i32 -114

cli_calloc.exit:		; preds = %bb1
	br i1 undef, label %bb52, label %bb4

bb4:		; preds = %cli_calloc.exit
	br i1 undef, label %bb.i, label %bb1.i3

bb.i:		; preds = %bb4
	unreachable

bb1.i3:		; preds = %bb4
	br i1 undef, label %bb2.i4, label %cli_strdup.exit

bb2.i4:		; preds = %bb1.i3
	ret i32 -114

cli_strdup.exit:		; preds = %bb1.i3
	br i1 undef, label %cli_calloc.exit54.thread, label %cli_calloc.exit54

cli_calloc.exit54.thread:		; preds = %cli_strdup.exit
	ret i32 -114

cli_calloc.exit54:		; preds = %cli_strdup.exit
	br label %bb45

cli_calloc.exit70.thread:		; preds = %bb45
	unreachable

cli_calloc.exit70:		; preds = %bb45
	br i1 undef, label %bb.i83, label %bb1.i84

bb.i83:		; preds = %cli_calloc.exit70
	unreachable

bb1.i84:		; preds = %cli_calloc.exit70
	br i1 undef, label %bb2.i85, label %bb17

bb2.i85:		; preds = %bb1.i84
	unreachable

bb17:		; preds = %bb1.i84
	br i1 undef, label %bb22, label %bb.nph

bb.nph:		; preds = %bb17
	br label %bb18

bb18:		; preds = %bb18, %bb.nph
	br i1 undef, label %bb18, label %bb22

bb22:		; preds = %bb18, %bb17
	%0 = getelementptr i8, i8* null, i32 10		; <i8*> [#uses=1]
	%1 = bitcast i8* %0 to i16*		; <i16*> [#uses=1]
	%2 = load i16, i16* %1, align 2		; <i16> [#uses=1]
	%3 = add i16 %2, 1		; <i16> [#uses=1]
	%4 = zext i16 %3 to i32		; <i32> [#uses=1]
	%5 = mul i32 %4, 3		; <i32> [#uses=1]
	%6 = add i32 %5, -1		; <i32> [#uses=1]
	%7 = icmp eq i32 %6, undef		; <i1> [#uses=1]
	br i1 %7, label %bb25, label %bb43.preheader

bb43.preheader:		; preds = %bb22
	br i1 undef, label %bb28, label %bb45

bb25:		; preds = %bb22
	unreachable

bb28:		; preds = %bb43.preheader
	unreachable

bb45:		; preds = %bb43.preheader, %cli_calloc.exit54
	br i1 undef, label %cli_calloc.exit70.thread, label %cli_calloc.exit70

bb52:		; preds = %cli_calloc.exit
	unreachable

bb126:		; preds = %entry
	ret i32 -117
}

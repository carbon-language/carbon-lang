; RUN: opt < %s -simplifycfg -S | not grep br
; END.

        %llvm.dbg.anchor.type = type { i32, i32 }
        %llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }

@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; 

@.str = internal constant [4 x i8] c"a.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [6 x i8] c"/tmp/\00", section "llvm.metadata"	; <[6 x i8]*> [#uses=1]
@.str2 = internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind


define void @main() {
entry:
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.14.i19 = icmp eq i32 0, 2		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.14.i19, label %endif.1.i20, label %read_min.exit
endif.1.i20:		; preds = %entry
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.9.i.i = icmp eq i8* null, null		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.9.i.i, label %then.i12.i, label %then.i.i
then.i.i:		; preds = %endif.1.i20
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
then.i12.i:		; preds = %endif.1.i20
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.9.i4.i = icmp eq i8* null, null		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.9.i4.i, label %endif.2.i33, label %then.i5.i
then.i5.i:		; preds = %then.i12.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
endif.2.i33:		; preds = %then.i12.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %loopexit.0.i40, label %no_exit.0.i35
no_exit.0.i35:		; preds = %no_exit.0.i35, %endif.2.i33
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.130.i = icmp slt i32 0, 0		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.130.i, label %loopexit.0.i40.loopexit, label %no_exit.0.i35
loopexit.0.i40.loopexit:		; preds = %no_exit.0.i35
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %loopexit.0.i40
loopexit.0.i40:		; preds = %loopexit.0.i40.loopexit, %endif.2.i33
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.341.i = icmp eq i32 0, 0		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.341.i, label %loopentry.1.i, label %read_min.exit
loopentry.1.i:		; preds = %loopexit.0.i40
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.347.i = icmp sgt i32 0, 0		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.347.i, label %no_exit.1.i41, label %loopexit.2.i44
no_exit.1.i41:		; preds = %endif.5.i, %loopentry.1.i
	%indvar.i42 = phi i32 [ %indvar.next.i, %endif.5.i ], [ 0, %loopentry.1.i ]		; <i32> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.355.i = icmp eq i32 0, 3		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.355.i, label %endif.5.i, label %read_min.exit
endif.5.i:		; preds = %no_exit.1.i41
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.34773.i = icmp sgt i32 0, 0		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%indvar.next.i = add i32 %indvar.i42, 1		; <i32> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.34773.i, label %no_exit.1.i41, label %loopexit.1.i.loopexit
loopexit.1.i.loopexit:		; preds = %endif.5.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
loopexit.2.i44:		; preds = %loopentry.1.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
read_min.exit:		; preds = %no_exit.1.i41, %loopexit.0.i40, %entry
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.23 = icmp eq i32 0, 0		; <i1> [#uses=1]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %tmp.23, label %endif.1, label %then.1
then.1:		; preds = %read_min.exit
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %endif.0.i, label %then.0.i
then.0.i:		; preds = %then.1
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %endif.1.i, label %then.1.i
endif.0.i:		; preds = %then.1
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %endif.1.i, label %then.1.i
then.1.i:		; preds = %endif.0.i, %then.0.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %getfree.exit, label %then.2.i
endif.1.i:		; preds = %endif.0.i, %then.0.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %getfree.exit, label %then.2.i
then.2.i:		; preds = %endif.1.i, %then.1.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
getfree.exit:		; preds = %endif.1.i, %then.1.i
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
endif.1:		; preds = %read_min.exit
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp.27.i = getelementptr i32* null, i32 0		; <i32*> [#uses=0]
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %loopexit.0.i15, label %no_exit.0.i14
no_exit.0.i14:		; preds = %endif.1
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
loopexit.0.i15:		; preds = %endif.1
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %primal_start_artificial.exit, label %no_exit.1.i16
no_exit.1.i16:		; preds = %no_exit.1.i16, %loopexit.0.i15
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 false, label %primal_start_artificial.exit, label %no_exit.1.i16
primal_start_artificial.exit:		; preds = %no_exit.1.i16, %loopexit.0.i15
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
}

; RUN: opt < %s -gvn -disable-output
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
@llvm.dbg.compile_unit298 = external constant %llvm.dbg.compile_unit.type		; <%llvm.dbg.compile_unit.type*> [#uses=1]

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

define i8* @__deregister_frame_info_bases(i8* %begin) {
entry:
	br i1 false, label %bb17, label %bb

bb:		; preds = %entry
	br i1 false, label %bb17, label %bb6.preheader

bb6.preheader:		; preds = %bb
	br label %bb6

bb3:		; preds = %bb6
	br i1 false, label %bb4, label %bb6

bb4:		; preds = %bb3
	br label %out

bb6:		; preds = %bb3, %bb6.preheader
	br i1 false, label %bb14.loopexit, label %bb3

bb8:		; preds = %bb14
	br i1 false, label %bb9, label %bb11

bb9:		; preds = %bb8
	br i1 false, label %bb10, label %bb13

bb10:		; preds = %bb9
	br label %out

bb11:		; preds = %bb8
	br i1 false, label %bb12, label %bb13

bb12:		; preds = %bb11
	br label %out

bb13:		; preds = %bb11, %bb9
	br label %bb14

bb14.loopexit:		; preds = %bb6
	br label %bb14

bb14:		; preds = %bb14.loopexit, %bb13
	br i1 false, label %bb15.loopexit, label %bb8

out:		; preds = %bb12, %bb10, %bb4
	tail call void @llvm.dbg.stoppoint(i32 217, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit298 to { }*))
	br i1 false, label %bb15, label %bb16

bb15.loopexit:		; preds = %bb14
	br label %bb15

bb15:		; preds = %bb15.loopexit, %out
	tail call void @llvm.dbg.stoppoint(i32 217, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit298 to { }*))
	unreachable

bb16:		; preds = %out
	ret i8* null

bb17:		; preds = %bb, %entry
	ret i8* null
}

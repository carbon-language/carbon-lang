
; RUN: opt < %s -licm -loop-index-split -instcombine -disable-output

	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
@"\01LC81" = external constant [4 x i8]		; <[4 x i8]*> [#uses=1]

define fastcc void @hex_dump_internal(i8* %avcl, %struct.FILE* %f, i32 %level, i8* nocapture %buf, i32 %size) nounwind {
entry:
	br i1 false, label %bb4, label %return

bb4:		; preds = %bb30, %entry
	br label %bb6

bb6:		; preds = %bb15, %bb4
	%j.0.reg2mem.0 = phi i32 [ %2, %bb15 ], [ 0, %bb4 ]		; <i32> [#uses=2]
	%0 = icmp slt i32 %j.0.reg2mem.0, 0		; <i1> [#uses=1]
	br i1 %0, label %bb7, label %bb13

bb7:		; preds = %bb6
	br label %bb15

bb13:		; preds = %bb6
	%1 = tail call i32 @fwrite(i8* getelementptr ([4 x i8]* @"\01LC81", i32 0, i32 0), i32 1, i32 3, i8* null) nounwind		; <i32> [#uses=0]
	br label %bb15

bb15:		; preds = %bb13, %bb7
	%2 = add i32 %j.0.reg2mem.0, 1		; <i32> [#uses=2]
	%3 = icmp sgt i32 %2, 15		; <i1> [#uses=1]
	br i1 %3, label %bb30, label %bb6

bb30:		; preds = %bb15
	br i1 false, label %bb4, label %return

return:		; preds = %bb30, %entry
	ret void
}

declare i32 @fwrite(i8* nocapture, i32, i32, i8* nocapture) nounwind

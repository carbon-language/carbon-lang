; RUN: llc -verify-machineinstrs < %s

target triple = "powerpc-unknown-linux-gnu"
	%struct..0anon = type { i32 }
	%struct.A = type { %struct.anon }
	%struct.anon = type <{  }>

define void @bork(%struct.A* %In0P) {
entry:
	%tmp56 = bitcast %struct.A* %In0P to float*		; <float*> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%i.035.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%tmp8 = getelementptr float, float* %tmp56, i32 %i.035.0		; <float*> [#uses=2]
	%tmp101112 = bitcast float* %tmp8 to i8*		; <i8*> [#uses=1]
	%tmp1617 = bitcast float* %tmp8 to i32*		; <i32*> [#uses=1]
	%tmp21 = tail call i32 asm "lwbrx $0, $2, $1", "=r,r,bO,*m"( i8* %tmp101112, i32 0, i32* %tmp1617 )		; <i32> [#uses=0]
	%indvar.next = add i32 %i.035.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 4		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}

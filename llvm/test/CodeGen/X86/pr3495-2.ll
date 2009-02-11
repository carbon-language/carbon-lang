; RUN: llvm-as < %s | llc -march=x86 -relocation-model=pic -disable-fp-elim -stats |& grep {Number of reloads omited}

target triple = "i386-apple-darwin9.6"
	%struct.constraintVCGType = type { i32, i32, i32, i32 }
	%struct.nodeVCGType = type { %struct.constraintVCGType*, i32, i32, i32, %struct.constraintVCGType*, i32, i32, i32 }

define fastcc void @SCC_DFSBelowVCG(%struct.nodeVCGType* %VCG, i32 %net, i32 %label) nounwind {
entry:
	%0 = getelementptr %struct.nodeVCGType* %VCG, i32 %net, i32 5		; <i32*> [#uses=2]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb5, label %bb.nph3

bb.nph3:		; preds = %entry
	%3 = getelementptr %struct.nodeVCGType* %VCG, i32 %net, i32 4		; <%struct.constraintVCGType**> [#uses=1]
	br label %bb

bb:		; preds = %bb3, %bb.nph3
	%s.02 = phi i32 [ 0, %bb.nph3 ], [ %12, %bb3 ]		; <i32> [#uses=2]
	%4 = load %struct.constraintVCGType** %3, align 4		; <%struct.constraintVCGType*> [#uses=1]
	%5 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %5, label %bb1, label %bb3

bb1:		; preds = %bb
	%6 = getelementptr %struct.constraintVCGType* %4, i32 %s.02, i32 0		; <i32*> [#uses=1]
	%7 = load i32* %6, align 4		; <i32> [#uses=2]
	%8 = getelementptr %struct.nodeVCGType* %VCG, i32 %7, i32 7		; <i32*> [#uses=1]
	%9 = load i32* %8, align 4		; <i32> [#uses=1]
	%10 = icmp eq i32 %9, 0		; <i1> [#uses=1]
	br i1 %10, label %bb2, label %bb3

bb2:		; preds = %bb1
	%11 = getelementptr %struct.nodeVCGType* %VCG, i32 %7, i32 4		; <%struct.constraintVCGType**> [#uses=0]
	br label %bb.i

bb.i:		; preds = %bb.i, %bb2
	br label %bb.i

bb3:		; preds = %bb1, %bb
	%12 = add i32 %s.02, 1		; <i32> [#uses=2]
	%13 = load i32* %0, align 4		; <i32> [#uses=1]
	%14 = icmp ugt i32 %13, %12		; <i1> [#uses=1]
	br i1 %14, label %bb, label %bb5

bb5:		; preds = %bb3, %entry
	%15 = getelementptr %struct.nodeVCGType* %VCG, i32 %net, i32 6		; <i32*> [#uses=1]
	store i32 %label, i32* %15, align 4
	ret void
}

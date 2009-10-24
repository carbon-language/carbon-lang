; RUN: opt < %s -loop-rotate -verify-dom-info -verify-loop-info -disable-output

	%struct.relation = type { [4 x i16], i32, [4 x i16], i32, i32 }

define void @findAllPairs() {
entry:
	br i1 false, label %bb139, label %bb10.i44
bb10.i44:		; preds = %entry
	ret void
bb127:		; preds = %bb139
	br label %bb139
bb139:		; preds = %bb127, %entry
	br i1 false, label %bb127, label %bb142
bb142:		; preds = %bb139
	%r91.0.lcssa = phi %struct.relation* [ null, %bb139 ]		; <%struct.relation*> [#uses=0]
	ret void
}


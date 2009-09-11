; RUN: opt < %s -analyze -scalar-evolution -disable-output | grep {count is 2}
; PR3171

	%struct.Foo = type { i32 }
	%struct.NonPod = type { [2 x %struct.Foo] }

define void @_Z3foov() nounwind {
entry:
	%x = alloca %struct.NonPod, align 8		; <%struct.NonPod*> [#uses=2]
	%0 = getelementptr %struct.NonPod* %x, i32 0, i32 0		; <[2 x %struct.Foo]*> [#uses=1]
	%1 = getelementptr [2 x %struct.Foo]* %0, i32 1, i32 0		; <%struct.Foo*> [#uses=1]
	br label %bb1.i

bb1.i:		; preds = %bb2.i, %entry
	%.0.i = phi %struct.Foo* [ %1, %entry ], [ %4, %bb2.i ]		; <%struct.Foo*> [#uses=2]
	%2 = getelementptr %struct.NonPod* %x, i32 0, i32 0, i32 0		; <%struct.Foo*> [#uses=1]
	%3 = icmp eq %struct.Foo* %.0.i, %2		; <i1> [#uses=1]
	br i1 %3, label %_ZN6NonPodD1Ev.exit, label %bb2.i

bb2.i:		; preds = %bb1.i
	%4 = getelementptr %struct.Foo* %.0.i, i32 -1		; <%struct.Foo*> [#uses=1]
	br label %bb1.i

_ZN6NonPodD1Ev.exit:		; preds = %bb1.i
	ret void
}


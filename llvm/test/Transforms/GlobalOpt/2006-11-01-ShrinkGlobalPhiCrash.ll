; RUN: llvm-upgrade < %s | llvm-as | opt -globalopt -disable-output
	%struct._list = type { int*, %struct._list* }
	%struct._play = type { int, int*, %struct._list*, %struct._play* }
%nrow = internal global int 0		; <int*> [#uses=2]

implementation   ; Functions:

void %make_play() {
entry:
	br label %cond_true16.i

cond_true16.i:		; preds = %cond_true16.i, %entry
	%low.0.in.i.0 = phi int* [ %nrow, %entry ], [ null, %cond_true16.i ]		; <int*> [#uses=1]
	%low.0.i = load int* %low.0.in.i.0		; <int> [#uses=0]
	br label %cond_true16.i
}

void %make_wanted() {
entry:
	unreachable
}

void %get_good_move() {
entry:
	ret void
}

void %main() {
entry:
	store int 8, int* %nrow
	tail call void %make_play( )
	ret void
}

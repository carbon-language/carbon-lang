; Loop Simplify should turn phi nodes like X = phi [X, Y]  into just Y, eliminating them.
; RUN: llvm-as < %s | opt -loopsimplify | llvm-dis | grep phi | wc -l | grep 6

%A = weak global [3000000 x int] zeroinitializer		; <[3000000 x int]*> [#uses=1]
%B = weak global [20000 x int] zeroinitializer		; <[20000 x int]*> [#uses=1]
%C = weak global [100 x int] zeroinitializer		; <[100 x int]*> [#uses=1]
%Z = weak global int 0		; <int*> [#uses=2]

implementation   ; Functions:

int %main() {
entry:
	tail call void %__main( )
	br label %loopentry.1

loopentry.1:		; preds = %loopexit.1, %entry
	%indvar20 = phi uint [ 0, %entry ], [ %indvar.next21, %loopexit.1 ]		; <uint> [#uses=1]
	%a.1 = phi int* [ getelementptr ([3000000 x int]* %A, int 0, int 0), %entry ], [ %inc.0, %loopexit.1 ]		; <int*> [#uses=1]
	br label %no_exit.2

no_exit.2:		; preds = %loopexit.2, %no_exit.2, %loopentry.1
	%a.0.4.ph = phi int* [ %a.1, %loopentry.1 ], [ %inc.0, %loopexit.2 ], [ %a.0.4.ph, %no_exit.2 ]		; <int*> [#uses=3]
	%b.1.4.ph = phi int* [ getelementptr ([20000 x int]* %B, int 0, int 0), %loopentry.1 ], [ %inc.1, %loopexit.2 ], [ %b.1.4.ph, %no_exit.2 ]		; <int*> [#uses=3]
	%indvar17 = phi uint [ 0, %loopentry.1 ], [ %indvar.next18, %loopexit.2 ], [ %indvar17, %no_exit.2 ]		; <uint> [#uses=2]
	%indvar = phi uint [ %indvar.next, %no_exit.2 ], [ 0, %loopexit.2 ], [ 0, %loopentry.1 ]		; <uint> [#uses=5]
	%b.1.4.rec = cast uint %indvar to int		; <int> [#uses=1]
	%c.2.4 = getelementptr [100 x int]* %C, int 0, uint %indvar		; <int*> [#uses=1]
	%a.0.4 = getelementptr int* %a.0.4.ph, uint %indvar		; <int*> [#uses=1]
	%b.1.4 = getelementptr int* %b.1.4.ph, uint %indvar		; <int*> [#uses=1]
	%inc.0.rec = add int %b.1.4.rec, 1		; <int> [#uses=2]
	%inc.0 = getelementptr int* %a.0.4.ph, int %inc.0.rec		; <int*> [#uses=2]
	%tmp.13 = load int* %a.0.4		; <int> [#uses=1]
	%inc.1 = getelementptr int* %b.1.4.ph, int %inc.0.rec		; <int*> [#uses=1]
	%tmp.15 = load int* %b.1.4		; <int> [#uses=1]
	%tmp.18 = load int* %c.2.4		; <int> [#uses=1]
	%tmp.16 = mul int %tmp.15, %tmp.13		; <int> [#uses=1]
	%tmp.19 = mul int %tmp.16, %tmp.18		; <int> [#uses=1]
	%tmp.20 = load int* %Z		; <int> [#uses=1]
	%tmp.21 = add int %tmp.19, %tmp.20		; <int> [#uses=1]
	store int %tmp.21, int* %Z
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=2]
	%exitcond = seteq uint %indvar.next, 100		; <bool> [#uses=1]
	br bool %exitcond, label %loopexit.2, label %no_exit.2

loopexit.2:		; preds = %no_exit.2
	%indvar.next18 = add uint %indvar17, 1		; <uint> [#uses=2]
	%exitcond19 = seteq uint %indvar.next18, 200		; <bool> [#uses=1]
	br bool %exitcond19, label %loopexit.1, label %no_exit.2

loopexit.1:		; preds = %loopexit.2
	%indvar.next21 = add uint %indvar20, 1		; <uint> [#uses=2]
	%exitcond22 = seteq uint %indvar.next21, 300		; <bool> [#uses=1]
	br bool %exitcond22, label %return, label %loopentry.1

return:		; preds = %loopexit.1
	ret int undef
}

declare void %__main()

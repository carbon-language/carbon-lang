; RUN: llvm-as < %s | opt -tailduplicate -disable-output

void %ab() {
entry:
	br label %loopentry.5

loopentry.5:
	%poscnt.1 = phi long [ 0, %entry ], [ %tmp.289, %no_exit.5 ]
	%tmp.289 = shr long %poscnt.1, ubyte 1
	br bool false, label %no_exit.5, label %loopexit.5

no_exit.5:
	br label %loopentry.5

loopexit.5:
	ret void
}

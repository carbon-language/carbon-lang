; RUN: llvm-as < %s | opt -tailduplicate -disable-output

implementation

int %sum() {
entry:
	br label %loopentry

loopentry:
	%i.0 = phi int [ 1, %entry ], [ %tmp.3, %loopentry ]
	%tmp.3 = add int %i.0, 1
	br label %loopentry
}

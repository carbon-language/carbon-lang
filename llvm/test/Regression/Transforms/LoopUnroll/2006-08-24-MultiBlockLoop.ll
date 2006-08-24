; RUN: llvm-as < %s | opt -loop-unroll | llvm-dis | grep "bb72.2"

void %vorbis_encode_noisebias_setup() {
entry:
	br label %cond_true.outer

cond_true.outer:
	%indvar1.ph = phi uint [ 0, %entry ], [ %indvar.next2, %bb72 ]
	br label %bb72

bb72:
	%indvar.next2 = add uint %indvar1.ph, 1
	%exitcond3 = seteq uint %indvar.next2, 3
	br bool %exitcond3, label %cond_true138, label %cond_true.outer

cond_true138:
	ret void
}

; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 add | grep preheader.loopexit: 

implementation

void %test() {
loopentry.2.i:
	br bool false, label %no_exit.1.i.preheader, label %loopentry.3.i.preheader

no_exit.1.i.preheader:
	br label %no_exit.1.i

no_exit.1.i:
	br bool false, label %return.i, label %endif.8.i

endif.8.i:
	%inc.1.i = add int 0, 1
	br bool false, label %no_exit.1.i, label %loopentry.3.i.preheader.loopexit

loopentry.3.i.preheader.loopexit:
	br label %loopentry.3.i.preheader

loopentry.3.i.preheader:
	%arg_num.0.i.ph13000 = phi int [ 0, %loopentry.2.i ], [ %inc.1.i, %loopentry.3.i.preheader.loopexit ]
	ret void

return.i:
	ret void
}

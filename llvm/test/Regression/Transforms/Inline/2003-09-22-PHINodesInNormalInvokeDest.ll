; RUN: llvm-as < %s | opt -inline -disable-output
implementation   

int %main() {
entry:
	invoke void %__main( )
			to label %else except label %RethrowExcept

else:
	%i.2 = phi int [ 36, %entry ], [ %i.2, %LJDecisionBB ]
	br label %LJDecisionBB

LJDecisionBB:
	br label %else

RethrowExcept:
	ret int 0
}

void %__main() {
	ret void
}



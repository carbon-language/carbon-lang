; RUN: llvm-as < %s | opt -inline -disable-output

implementation

int %main() {
entry:
	invoke void %__main( )
			to label %Call2Invoke except label %LongJmpBlkPre

Call2Invoke:
	br label %LongJmpBlkPre

LongJmpBlkPre:
	%i.3 = phi uint [ 0, %entry ], [ 0, %Call2Invoke ]		; <uint> [#uses=0]
	ret int 0
}

void %__main() {
	call void %__llvm_getGlobalCtors( )
	call void %__llvm_getGlobalDtors( )
	ret void
}

declare void %__llvm_getGlobalCtors()

declare void %__llvm_getGlobalDtors()

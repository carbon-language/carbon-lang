; RUN: llvm-as < %s | opt -inline -disable-output
implementation

int %main() {
entry:
	invoke void %__main( )
			to label %LongJmpBlkPre except label %LongJmpBlkPre

LongJmpBlkPre:
	%i.3 = phi uint [ 0, %entry ], [ 0, %entry]
	ret int 0
}

void %__main() {
	ret void
}


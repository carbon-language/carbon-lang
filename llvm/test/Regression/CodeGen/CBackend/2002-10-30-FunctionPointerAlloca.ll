; RUN: llvm-as < %s | llc -march=c


	%BitField = type int
        %tokenptr = type %BitField*

implementation

void %test() {
	%pmf1 = alloca %tokenptr (%tokenptr, sbyte*)*
	ret void
}

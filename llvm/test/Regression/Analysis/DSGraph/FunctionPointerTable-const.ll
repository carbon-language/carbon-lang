; FIXME: this should be SHM for bu, but change it for now since besides incompleteness
;        this is working
; RUN: llvm-as < %s | opt -analyze -datastructure-gc -dsgc-dspass=bu -dsgc-check-flags=Y:SHIM && \
; RUN: llvm-as < %s | opt -analyze -datastructure-gc -dsgc-dspass=td -dsgc-check-flags=P1:SHM,P2:SHM

%G = internal constant [2 x int*(int*)*] [ 
  int*(int*)* %callee1, int*(int*)* %callee2
]

implementation

internal int* %callee1(int* %P1) {
	ret int* %P1
}

internal int* %callee2(int* %P2) {
	%X = malloc int
	ret int* %X
}

void %caller(int %callee) {
	%FPP = getelementptr [2 x int*(int*)*]* %G, int 0, int %callee
	%FP = load int*(int*)** %FPP

	%Y = alloca int
	%Z = call int* %FP(int* %Y)
	store int 4, int* %Z
	ret void
}

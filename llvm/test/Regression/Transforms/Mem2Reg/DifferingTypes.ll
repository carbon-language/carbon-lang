; This is a feature test.  Hopefully one day this will be implemented.  The 
; generated code should perform the appropriate masking operations required 
; depending on the endianness of the target...

; RUN: llvm-as < %s | opt -mem2reg | llvm-dis | not grep 'alloca'

implementation

int %testfunc(int %i, sbyte %j) {
	%I = alloca int

	store int %i, int* %I

	%P = cast int* %I to sbyte*
	store sbyte %j, sbyte* %P

	%t = load int* %I
	ret int %t
}

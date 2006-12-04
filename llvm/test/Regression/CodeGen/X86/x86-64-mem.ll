; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin | grep GOTPCREL | wc -l | grep 4 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin | grep rip | wc -l | grep 6 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin | grep movq | wc -l | grep 6 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin | grep leaq | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin -relocation-model=static | grep rip | wc -l | grep 4 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin -relocation-model=static | grep movl | wc -l | grep 2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin -relocation-model=static | grep movq | wc -l | grep 2

%ptr = external global int*
%src = external global [0 x int]
%dst = external global [0 x int]
%lptr = internal global int* null
%ldst = internal global [500 x int] zeroinitializer, align 32
%lsrc = internal global [500 x int] zeroinitializer, align 32
%bsrc = internal global [500000 x int] zeroinitializer, align 32
%bdst = internal global [500000 x int] zeroinitializer, align 32

void %test1() {
	%tmp = load int* getelementptr ([0 x int]* %src, int 0, int 0)
	store int %tmp, int* getelementptr ([0 x int]* %dst, int 0, int 0)
	ret void
}

void %test2() {
	store int* getelementptr ([0 x int]* %dst, int 0, int 0), int** %ptr
	ret void
}

void %test3() {
	store int* getelementptr ([500 x int]* %ldst, int 0, int 0), int** %lptr
	br label %return

return:
	ret void
}

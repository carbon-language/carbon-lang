; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=x86_64-apple-darwin -o %t1 -f
; RUN: grep GOTPCREL %t1 | wc -l | grep 4 
; RUN: grep rip      %t1 | wc -l | grep 6 
; RUN: grep movq     %t1 | wc -l | grep 6 
; RUN: grep leaq     %t1 | wc -l | grep 1 
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -mtriple=x86_64-apple-darwin -relocation-model=static -o %t2 -f
; RUN: grep rip  %t2 | wc -l | grep 4 
; RUN: grep movl %t2 | wc -l | grep 2 
; RUN: grep movq %t2 | wc -l | grep 2

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

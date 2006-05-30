; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep 'subl.*60'
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep 'movdqa.*32'

void %test() {
	tail call void %xx( int 1, int 2, int 3, int 4, int 5, int 6, int 7, <2 x long> cast (<4 x int> < int 4, int 3, int 2, int 1 > to <2 x long>), <2 x long> cast (<4 x int> < int 8, int 7, int 6, int 5 > to <2 x long>), <2 x long> cast (<4 x int> < int 6, int 4, int 2, int 0 > to <2 x long>), <2 x long> cast (<4 x int> < int 8, int 4, int 2, int 1 > to <2 x long>), <2 x long> cast (<4 x int> < int 0, int 1, int 3, int 9 > to <2 x long>) )
	ret void
}

declare void %xx(int, int, int, int, int, int, int, <2 x long>, <2 x long>, <2 x long>, <2 x long>, <2 x long>)

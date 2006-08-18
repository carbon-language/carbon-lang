
; RUN: opt -analyze %s -datastructure-gc --dsgc-abort-if-merged=Y,BVal

implementation

internal void %F1(int* %X) {
	%Y = alloca int
	store int 4, int* %Y
	%BVal = call int* %F2(int* %Y)
	ret void
}

internal int* %F2(int* %A) {
	%B = malloc int
	store int 4, int* %B
	call void %F1(int* %B)
	ret int* %B
}

int %main() {
	%Q = malloc int
	store int 4, int* %Q
	call void %F1(int* %Q)
	ret int 0
}

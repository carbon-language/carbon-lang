; RUN: llvm-as < %s | opt -anders-aa -load-vn -gcse -deadargelim | llvm-dis | grep store | not grep null

; Because the 'internal' function is passed to an external function, we don't
; know what the incoming values will alias.  As such, we cannot do the 
; optimization checked by the 'arg-must-alias.llx' test.

declare void %external(int(int*)*)
%G = internal constant int* null

implementation

internal int %internal(int* %ARG) {
	;;; We *DON'T* know that ARG always points to null!
	store int* %ARG, int** %G
	ret int 0
}

int %foo() {
	call void %external(int(int*)* %internal)
	%V = call int %internal(int* null)
	ret int %V
}

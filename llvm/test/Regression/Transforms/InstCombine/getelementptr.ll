; The %A getelementptr instruction should be eliminated here

; RUN: if as < %s | opt -instcombine -dce | dis | grep getelementptr | grep -v '%C'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int *"foo"(int * %I)
begin
	%A = getelementptr int* %I, uint 17
	store int 23, int* %A

	%B = load int* %A
	store int %B, int* %A, uint 0

	%C = getelementptr int* %A
	ret int* %C
end


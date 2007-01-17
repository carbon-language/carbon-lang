; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | not grep "805306384"


int %main() {
entry:
	%retval = alloca int, align 4		; <int*> [#uses=2]
	%tmp = alloca int, align 4		; <int*> [#uses=2]
	%a = alloca [805306369 x sbyte], align 16		; <[805306369 x sbyte]*> [#uses=0]
	"alloca point" = bitcast int 0 to int		; <int> [#uses=0]
	store int 0, int* %tmp
	%tmp = load int* %tmp		; <int> [#uses=1]
	store int %tmp, int* %retval
	br label %return

return:		; preds = %entry
	%retval = load int* %retval		; <int> [#uses=1]
	ret int %retval
}

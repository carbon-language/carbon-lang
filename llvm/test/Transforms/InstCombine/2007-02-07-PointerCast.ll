;RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep zext

; Make sure the uint isn't removed.  Instcombine in llvm 1.9 was dropping the 
; uint cast which was causing a sign extend. This only affected code with 
; pointers in the high half of memory, so it wasn't noticed much
; compile a kernel though...

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32

%str = internal constant [6 x sbyte] c"%llx\0A\00"

implementation   ; Functions:

declare int %printf(sbyte*, ...)

int %main(int %x, sbyte** %a) {
entry:
        %tmp = getelementptr [6 x sbyte]* %str, int 0, uint 0 
        %tmp1 = load sbyte** %a
	%tmp2 = cast sbyte* %tmp1 to uint		; <uint> [#uses=1]
	%tmp3 = cast uint %tmp2 to long		; <long> [#uses=1]
        %tmp = call int (sbyte*, ...)* %printf( sbyte* %tmp, long %tmp3 )
        ret int 0
}

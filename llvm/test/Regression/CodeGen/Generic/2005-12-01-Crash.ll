; RUN: llvm-as < %s | llc

%str = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%str = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=0]
%str1 = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=0]
%str2 = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=1]
%str = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%str3 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%str4 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%str5 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]

implementation   ; Functions:

void %printArgsNoRet(int %a1, float %a2, sbyte %a3, double %a4, sbyte* %a5, int %a6, float %a7, sbyte %a8, double %a9, sbyte* %a10, int %a11, float %a12, sbyte %a13, double %a14, sbyte* %a15) {
entry:
	%tmp17 = cast sbyte %a13 to int		; <int> [#uses=1]
	%tmp23 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([29 x sbyte]* %str2, int 0, uint 0), int %a11, double 0.000000e+00, int %tmp17, double %a14, int 0 )		; <int> [#uses=0]
	ret void
}

declare int %printf(sbyte*, ...)

declare int %main(int, sbyte**)

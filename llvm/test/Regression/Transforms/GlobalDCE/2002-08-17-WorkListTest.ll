; This testcase tests that a worklist is being used, and that globals can be 
; removed if they are the subject of a constexpr and ConstantPointerRef

; RUN: if as < %s | opt -globaldce | dis | grep global
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%t0 = internal global [4 x sbyte] c"foo\00"
%t1 = internal global [4 x sbyte] c"bar\00"

%s1 = internal global [1 x sbyte*] [sbyte* cast ([4 x sbyte]* %t0 to sbyte*)]
%s2 = internal global [1 x sbyte*] [sbyte* getelementptr ([4 x sbyte]* %t1, uint 0, uint 0 )]

%b = internal global int* %a
%a = internal global int 7


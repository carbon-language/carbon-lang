; Test that: extern int X[]  and int X[] = { 1, 2, 3, 4 } are resolved correctly.
;
; RUN: if as < %s | opt -gcse -instcombine -constprop -dce | dis | grep getelementptr
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
;

%X = uninitialized global int           ; <int*> [#uses=1]
%X = global [4 x int] [ int 1, int 2, int 3, int 4 ]            ; <[4 x int]*> [#uses=1]

implementation   ; Functions:

int %foo(int %x) {
bb1:                                    ;[#uses=0]
        %reg107-idxcast = cast int %x to uint           ; <uint> [#uses=2]
        %reg113 = getelementptr int* %X, uint %reg107-idxcast           ; <int*> [#uses=1]
        %reg120 = getelementptr [4 x int]* %X, uint 0, uint %reg107-idxcast             ; <int*> [#uses=1]
        %reg123 = sub int* %reg113, %reg120             ; <int*> [#uses=1]
        %cast232 = cast int* %reg123 to long            ; <long> [#uses=1]
        %reg234 = div long %cast232, 4          ; <long> [#uses=1]
        %cast108 = cast long %reg234 to int             ; <int> [#uses=1]
        ret int %cast108
}


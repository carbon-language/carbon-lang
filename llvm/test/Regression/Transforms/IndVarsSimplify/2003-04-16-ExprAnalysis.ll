; This is a test case for the expression analysis code, not really indvars.
; It was assuming any constant of int type was a ConstantInteger.
;
; RUN: as < %s | opt -indvars

%X = global int 7

void %test(int %A) {
        br label %Loop
Loop:
        %IV = phi int [%A, %0], [%IVNext, %Loop]
        %IVNext = add int %IV, cast (int* %X to int)
        br label %Loop
}
